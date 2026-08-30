import argparse
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="salome-agent")
    parser.add_argument("request")
    parser.add_argument("--batch", action="store_true")
    session_mode = parser.add_mutually_exclusive_group()
    session_mode.add_argument("--no-session", action="store_true")
    session_mode.add_argument("--live-gui", action="store_true",
                              help="attach to an already-open SALOME GUI (launched with "
                                   "`salome withsession`) so objects appear as stages run")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--on-failure", choices=["abort", "continue"], default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show per-tool-call / per-retry detail")
    return parser.parse_args(argv)


def build_llm(config: dict):
    from salome_agent.llm import LLMClient

    llm_config = config["llm"]
    return LLMClient(base_url=llm_config["base_url"], api_key=llm_config.get("api_key", ""),
                      model=llm_config["model"], temperature=llm_config.get("temperature", 0))


def main(argv: list[str] | None = None) -> int:
    import os
    import uuid

    from salome_agent.config import load_config
    from salome_agent.logs import log, setup_logging
    from salome_agent.session.manager import SessionManager
    from salome_agent.graph import build_graph

    args = parse_args(argv if argv is not None else sys.argv[1:])
    setup_logging(args.verbose)

    config = load_config(args.config or "config.json")
    if args.on_failure:
        config["policy"]["on_stage_failure"] = args.on_failure
    if args.work_dir:
        config["work_dir"] = args.work_dir

    run_id = uuid.uuid4().hex[:12]
    work_dir = os.path.expanduser(config.get("work_dir") or ".")
    run_dir = os.path.join(work_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    executor_log = os.path.join(run_dir, "executor.log")
    gui_log = os.path.join(run_dir, "gui.log")

    log.info("run %s", run_id)
    log.info("request: %s", args.request)
    log.info("output dir: %s", run_dir)
    log.info("config: %s (salome root %s, llm %s)", args.config or "config.json",
             config["salome"]["root"], config["llm"].get("base_url"))
    if not args.batch:
        log.info("interactive mode: you will be prompted to approve the plan "
                 "and to handle stage failures (pass --batch to disable)")

    if args.no_session:
        log.info("session: --no-session (planner + specialists + static checks only)")
    elif args.live_gui:
        log.info("session: --live-gui (objects appear in the Object Browser and 3D "
                 "viewer as stages run; attaches to a running `salome withsession` GUI "
                 "or launches one for you — the window stays open afterward)")
    else:
        log.info("session: starting executor inside SALOME (isolated study, this "
                 "takes ~10-20s; SALOME output goes to executor.log)")

    manager = SessionManager(salome_root=config["salome"]["root"], executor_log=executor_log,
                             gui_log=gui_log)
    try:
        session_handle, client = manager.resolve(args.no_session, args.port,
                                                 live_gui=args.live_gui)
    except (RuntimeError, TimeoutError) as e:
        manager.stop()
        log.error("live GUI session unavailable: %s", e)
        log.error("try launching it yourself: `%s/salome withsession` (on Wayland, "
                  "prefix QT_QPA_PLATFORM=xcb), then retry.", config["salome"]["root"])
        return 1
    if client is not None:
        log.info("session ready: mode=%s port=%s executor_pid=%s",
                 session_handle["mode"], session_handle["port"], session_handle["executor_pid"])
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex,
                            "llm_client": build_llm(config),
                            "executor_client": client}}
    try:
        graph = build_graph(batch=args.batch)
        final = graph.invoke({
            "user_request": args.request, "config": {**config, "batch": args.batch},
            "session": session_handle, "plan": [], "plan_approved": False,
            "stage_idx": 0, "stages": {}, "last_error": None, "history": [],
            "study_ns": "salome_agent_run", "final_script": None, "report": None,
            "run_id": run_id,
        }, config=cfg)
        if not args.batch:
            from salome_agent.interactive import resume_interactive

            final = resume_interactive(graph, final, cfg)
    finally:
        manager.stop()

    if final.get("__interrupt__"):
        log.warning("run still paused at an interactive gate - aborting")
        return 1

    result_path = os.path.join(run_dir, "result.py")
    report_path = os.path.join(run_dir, "report.md")
    log.info("done")

    bar = "-" * 60
    print(f"\n{bar}")
    print(final.get("report") or "(no report)")
    print(bar)
    print(f"  result script : {result_path}")
    print(f"  report        : {report_path}")
    print(f"  executor log  : {executor_log}")
    print(bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
