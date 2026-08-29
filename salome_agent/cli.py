import argparse
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="salome-agent")
    parser.add_argument("request")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--no-session", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--on-failure", choices=["abort", "continue"], default=None)
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from salome_agent.config import load_config
    from salome_agent.session.manager import SessionManager
    from salome_agent.graph import build_graph

    args = parse_args(argv if argv is not None else sys.argv[1:])
    config = load_config(args.config or "config.json")
    if args.on_failure:
        config["policy"]["on_stage_failure"] = args.on_failure

    manager = SessionManager(salome_root=config["salome"]["root"])
    session_handle, client = manager.resolve(args.no_session, args.port)
    try:
        graph = build_graph(batch=args.batch)
        graph.invoke({
            "user_request": args.request, "config": {**config, "batch": args.batch},
            "session": session_handle, "plan": [], "plan_approved": False,
            "stage_idx": 0, "stages": {}, "last_error": None, "history": [],
            "study_ns": "salome_agent_run", "final_script": None, "report": None,
            "run_id": "",
        })
    finally:
        manager.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
