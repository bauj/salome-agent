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


def build_llm(config: dict):
    from salome_agent.llm import LLMClient

    llm_config = config["llm"]
    return LLMClient(base_url=llm_config["base_url"], api_key=llm_config.get("api_key", ""),
                      model=llm_config["model"], temperature=llm_config.get("temperature", 0))


def main(argv: list[str] | None = None) -> int:
    import uuid

    from salome_agent.config import load_config
    from salome_agent.session.manager import SessionManager
    from salome_agent.graph import build_graph

    args = parse_args(argv if argv is not None else sys.argv[1:])
    config = load_config(args.config or "config.json")
    if args.on_failure:
        config["policy"]["on_stage_failure"] = args.on_failure
    if args.work_dir:
        config["work_dir"] = args.work_dir

    manager = SessionManager(salome_root=config["salome"]["root"])
    session_handle, client = manager.resolve(args.no_session, args.port)
    try:
        graph = build_graph(batch=args.batch)
        graph.invoke({
            "user_request": args.request, "config": {**config, "batch": args.batch},
            "session": session_handle, "plan": [], "plan_approved": False,
            "stage_idx": 0, "stages": {}, "last_error": None, "history": [],
            "study_ns": "salome_agent_run", "final_script": None, "report": None,
            "run_id": uuid.uuid4().hex[:12],
        }, config={"configurable": {"thread_id": uuid.uuid4().hex,
                                     "llm_client": build_llm(config),
                                     "executor_client": client}})
    finally:
        manager.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
