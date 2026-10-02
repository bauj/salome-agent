import argparse
import os
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="salome-agent")
    parser.add_argument("request")
    parser.add_argument("--images", default=None, metavar="PATH,PATH,...",
                        help="comma-separated reference image(s) (PNG/JPEG/WEBP/GIF/BMP) "
                             "the generated CAD should match -- e.g. several views/plans "
                             "of the same part; a vision agent describes them, the "
                             "description (not the images) goes to the planner and "
                             "the geom specialist")
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
                        help="total wall-clock budget per LLM round-trip in seconds; a "
                             "generation still running at the limit is cut, not retried "
                             "(overrides llm.timeout in config)")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v: per-tool-call / per-retry detail; "
                             "-vv: also dump every LLM prompt, response and "
                             "reasoning to the terminal and <run_dir>/llm_trace.jsonl")
    parser.add_argument("--chat", action="store_true",
                        help="after the first result, keep talking to the agent in a "
                             "REPL (needs a SALOME session)")
    return parser.parse_args(argv)


# Recognised by content, not extension: the host's mimetypes table varies
# (.webp only from Python 3.11, .bmp as image/x-ms-bmp on some distros), and
# a mislabelled file would otherwise reach the model as garbage.
_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)
MAX_IMAGES = 10
MAX_IMAGE_BYTES = 20 * 1024 * 1024   # all images together, before base64


def parse_image_paths(raw: str) -> list[str]:
    """--images is a comma-separated path list; split and drop empty entries
    (a trailing comma, accidental double comma)."""
    return [p.strip() for p in raw.split(",") if p.strip()]


def _image_mime(data: bytes) -> str | None:
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    for magic, mime in _IMAGE_SIGNATURES:
        if data.startswith(magic):
            return mime
    return None


def load_reference_image(path: str) -> str:
    """Read an image file and return it as a data: URI for a multimodal LLM
    message. Raises ValueError on an unsupported type, OSError if unreadable."""
    import base64

    with open(path, "rb") as f:
        data = f.read()
    mime = _image_mime(data)
    if mime is None:
        raise ValueError(f"--images: {path!r} is not a PNG, JPEG, WEBP, GIF or BMP image")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def load_reference_images(paths: list[str]) -> list[str]:
    """Raises ValueError on no path at all (a typo like `--images ,` must not
    silently drop the images) and past MAX_IMAGES / MAX_IMAGE_BYTES."""
    if not paths:
        raise ValueError("--images: no image path given")
    if len(paths) > MAX_IMAGES:
        raise ValueError(f"--images: {len(paths)} images, at most {MAX_IMAGES}")
    total = sum(os.path.getsize(p) for p in paths)
    if total > MAX_IMAGE_BYTES:
        raise ValueError(f"--images: {total / (1024 * 1024):.1f} MB of images, "
                         f"at most {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    return [load_reference_image(p) for p in paths]


def check_image_uris(uris) -> list[str]:
    """Reference images that arrive as data: URIs (a GUI's attachments), held
    to the --images rules: base64, recognised by content, MAX_IMAGES /
    MAX_IMAGE_BYTES. Returns them relabelled with the type their content
    shows; ValueError otherwise."""
    import base64
    import binascii

    if not isinstance(uris, list):
        raise ValueError("expected a list of images")
    if len(uris) > MAX_IMAGES:
        raise ValueError(f"{len(uris)} images, at most {MAX_IMAGES}")
    checked, total = [], 0
    for uri in uris:
        head, _, payload = uri.partition(",") if isinstance(uri, str) else ("", "", "")
        if not (head.startswith("data:") and head.endswith(";base64")):
            raise ValueError("an image is not a base64 data: URI")
        try:
            data = base64.b64decode(payload, validate=True)
        except binascii.Error:
            raise ValueError("an image is not valid base64") from None
        mime = _image_mime(data)
        if mime is None:
            raise ValueError("only PNG, JPEG, WEBP, GIF and BMP images can be attached")
        total += len(data)
        checked.append(f"data:{mime};base64,{payload}")
    if total > MAX_IMAGE_BYTES:
        raise ValueError(f"{total / (1024 * 1024):.1f} MB of images, "
                         f"at most {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    return checked


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
    # module_extra_body: {"vision"|"planner"|"geom"|"smesh"|"paravis": {...}} — a body
    # that replaces extra_body for that module's calls, so an expensive knob
    # (reasoning_effort) can sit on the planner alone.
    module_extra_body = llm_config.get("module_extra_body") or {}
    return LLMClient(base_url=llm_config["base_url"], api_key=llm_config.get("api_key", ""),
                     model=llm_config["model"], ssl_cert_file=llm_config.get("ssl_cert_file", ""),
                     temperature=llm_config.get("temperature", 0),
                     request_timeout=request_timeout or llm_config.get("timeout", 120),
                     trace_path=trace_path, extra_headers=extra_headers, extra_body=extra_body,
                     json_retries=llm_config.get("json_retries", 2),
                     module_extra_body=module_extra_body,
                     max_retries=llm_config.get("max_retries", 2))


def main(argv: list[str] | None = None) -> int:
    import uuid

    from salome_agent import server_conn
    from salome_agent.config import load_config
    from salome_agent.logs import log, setup_logging

    args = parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging(args.verbose)

    if args.chat and args.batch:
        log.error("--chat is interactive; it cannot be combined with --batch")
        return 2
    if args.chat and args.no_session:
        log.error("--chat needs a SALOME session; drop --no-session")
        return 2

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
    if args.images is not None:
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
    else:
        try:
            # same condition as runner.py's llm_trace.jsonl: a recorded run
            # always has both trace halves, or neither.
            record_path = (os.path.join(run_dir, "exec_trace.jsonl")
                           if args.verbose >= 2 else None)
            client = server_conn.connect(run_id, record_path=record_path, hold=args.chat)
        except server_conn.ServerUnavailable as e:
            log.error("%s", e)
            return 1
        log.info("session: connected to the %s agent server", client.mode)
        if not args.batch:
            from salome_agent.interactive import offer_fresh_study
            offer_fresh_study(client)

    if args.chat:
        from salome_agent import chat
        from salome_agent.conversation import Conversation

        try:
            conv = Conversation(config, client, run_id, verbose=args.verbose,
                                request_timeout=args.timeout)
            return chat.repl(conv, args.request)
        finally:
            try:
                client.close()
            except Exception:
                pass

    from salome_agent import runner
    from salome_agent.interactive import terminal_gate

    try:
        result = runner.run_request(
            args.request, config, client=client, on_event=lambda ev: None,
            on_gate=terminal_gate, batch=args.batch, verbose=args.verbose,
            run_id=run_id, request_timeout=args.timeout,
            reference_images=reference_images)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    final = result.final_state

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
