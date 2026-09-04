import argparse
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="salome-agent")
    parser.add_argument("request")
    parser.add_argument("--images", default=None, metavar="PATH,PATH,...",
                        help="comma-separated reference image(s) (PNG/JPEG/WEBP/GIF/BMP) "
                             "the generated CAD should match -- e.g. several views/plans "
                             "of the same part; sent to the planner and the geom "
                             "specialist alongside the request")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--no-session", action="store_true",
                        help="planner + specialists + static checks only; "
                             "no connection to a SALOME agent server")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--on-failure", choices=["abort", "continue"], default=None)
    parser.add_argument("--tools", choices=["hints", "all"], default=None,
                        help="geom tool retrieval: 'hints' (default) offers the "
                             "specialist a per-stage subset it widens on demand; "
                             "'all' offers the full geom registry every call")
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-request LLM timeout in seconds (overrides llm.timeout in config)")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v: per-tool-call / per-retry detail; "
                             "-vv: also dump every LLM prompt, response and "
                             "reasoning to the terminal and <run_dir>/llm_trace.jsonl")
    return parser.parse_args(argv)


_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"}


def parse_image_paths(raw: str) -> list[str]:
    """--images is a comma-separated path list; split and drop empty entries
    (a trailing comma, accidental double comma)."""
    return [p.strip() for p in raw.split(",") if p.strip()]


def load_reference_image(path: str) -> str:
    """Read an image file and return it as a data: URI for a multimodal LLM
    message. Raises ValueError on an unsupported type, OSError if unreadable."""
    import base64
    import mimetypes

    mime, _ = mimetypes.guess_type(path)
    if mime not in _IMAGE_MIME_TYPES:
        raise ValueError(f"--images: unsupported or unrecognized image type for {path!r} "
                         f"(expected one of {sorted(_IMAGE_MIME_TYPES)})")
    with open(path, "rb") as f:
        data = f.read()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def load_reference_images(paths: list[str]) -> list[str]:
    return [load_reference_image(p) for p in paths]


def build_llm(config: dict, request_timeout: float | None = None,
              trace_path: str | None = None, run_id: str | None = None):
    from salome_agent.llm import LLMClient

    llm_config = config["llm"]
    # extra_headers: verbatim per-request headers; string values may contain
    # {run_id}, substituted with this run's id (a stable per-conversation token
    # some gateways require).
    extra_headers = {
        k: v.format(run_id=run_id) if isinstance(v, str) else v
        for k, v in (llm_config.get("extra_headers") or {}).items()
    }
    # extra_body: verbatim top-level chat-completions body fields (e.g. a
    # provider's reasoning_effort knob) — unlike extra_headers, these land in
    # the JSON payload, not the HTTP headers (GH #28).
    extra_body = llm_config.get("extra_body") or {}
    return LLMClient(base_url=llm_config["base_url"], api_key=llm_config.get("api_key", ""),
                     model=llm_config["model"], ssl_cert_file=llm_config.get("ssl_cert_file", ""),
                     temperature=llm_config.get("temperature", 0),
                     request_timeout=request_timeout or llm_config.get("timeout", 120),
                     trace_path=trace_path, extra_headers=extra_headers, extra_body=extra_body,
                     json_retries=llm_config.get("json_retries", 2))


def main(argv: list[str] | None = None) -> int:
    import os
    import uuid

    from salome_agent import server_conn
    from salome_agent.config import load_config
    from salome_agent.graph import build_graph
    from salome_agent.logs import log, setup_logging

    args = parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging(args.verbose)

    config = load_config(args.config or "config.json")
    if args.on_failure:
        config["policy"]["on_stage_failure"] = args.on_failure
    if args.tools:
        config["policy"]["tool_selection"] = args.tools
    if args.work_dir:
        config["work_dir"] = args.work_dir

    run_id = uuid.uuid4().hex[:12]
    work_dir = os.path.expanduser(config.get("work_dir") or ".")
    run_dir = os.path.join(work_dir, run_id)
    # run_dir is created by _report_node when it actually has output to write;
    # a run aborted at an interactive gate leaves nothing behind.

    reference_images = None
    if args.images:
        image_paths = parse_image_paths(args.images)
        try:
            reference_images = load_reference_images(image_paths)
        except (OSError, ValueError) as e:
            log.error("%s", e)
            return 1

    log.info("run %s", run_id)
    log.info("request: %s", args.request)
    if reference_images:
        log.info("reference image(s): %s", ", ".join(image_paths))
    log.info("output dir: %s", run_dir)
    if not args.batch:
        log.info("interactive mode: you will be prompted to approve the plan "
                 "and to handle stage failures (pass --batch to disable)")

    if args.no_session:
        log.info("session: --no-session (planner + specialists + static checks only)")
        client = None
        session_handle = {"mode": "none", "pid": None, "salome_port": None,
                          "name_prefix": ""}
    else:
        try:
            client = server_conn.connect(run_id)
        except server_conn.ServerUnavailable as e:
            log.error("%s", e)
            return 1
        session_handle = {"mode": client.mode, "pid": None, "salome_port": None,
                          "name_prefix": client.name_prefix}
        log.info("session: connected to the %s agent server", client.mode)

    trace_path = os.path.join(run_dir, "llm_trace.jsonl") if args.verbose >= 2 else None
    if trace_path:
        log.info("LLM trace: %s", trace_path)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex,
                            "llm_client": build_llm(config, args.timeout, trace_path, run_id),
                            "executor_client": client}}
    try:
        graph = build_graph(batch=args.batch)
        final = graph.invoke({
            "user_request": args.request, "reference_images": reference_images,
            "config": {**config, "batch": args.batch},
            "session": session_handle, "plan": [], "plan_approved": False,
            "stage_idx": 0, "stages": {}, "last_error": None, "history": [],
            "study_ns": "salome_agent_run", "final_script": None, "report": None,
            "fatal": None, "aborted": False, "run_id": run_id,
        }, config=cfg)
        if not args.batch:
            from salome_agent.interactive import resume_interactive

            final = resume_interactive(graph, final, cfg)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    if final.get("__interrupt__"):
        log.warning("run still paused at an interactive gate - aborting")
        return 1

    if final.get("aborted"):
        log.info("run aborted at your request - no result.py / report.md written")
        return 1

    result_path = os.path.join(run_dir, "result.py")
    report_path = os.path.join(run_dir, "report.md")
    hs = server_conn.read_handshake() or {}
    log.info("done")

    bar = "-" * 60
    print(f"\n{bar}")
    print(final.get("report") or "(no report)")
    print(bar)
    print(f"  result script : {result_path}")
    print(f"  report        : {report_path}")
    print(f"  server log    : {hs.get('log_path', '(n/a)')}")
    print(bar)
    return 1 if final.get("fatal") else 0


if __name__ == "__main__":
    sys.exit(main())
