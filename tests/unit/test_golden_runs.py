import os

import pytest

from tests.replay import ReplayExecutorClient, ReplayLLM, load_cassette

_GOLDEN = os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden")
_CASSETTES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "cassettes")

_REQUESTS = {
    "vase": "a vase 200mm tall, 80mm wide at the base, hollow with 4mm walls",
    "enclosure": ("a 300x200x150mm enclosure with 5mm walls, a 60mm circular vent "
                  "on the front face, and all vertical edges filleted at 8mm"),
}

# The golden result.py embeds the run id in its header comment, so the
# replay must reuse the id of the live run that produced that golden.
_RUN_IDS = {"vase": "817b5ec87944", "enclosure": "da7ca73a0a12"}


def _has(name):
    return os.path.exists(os.path.join(_CASSETTES, f"{name}.json"))


@pytest.mark.parametrize("name", sorted(_REQUESTS))
def test_replayed_run_reproduces_its_golden_result_py(name, tmp_path):
    if not _has(name):
        pytest.skip(f"cassette {name} not captured yet (see plan Task 0b)")
    cassette = load_cassette(name)
    script = _replay_run(_REQUESTS[name], cassette, str(tmp_path), _RUN_IDS[name])
    with open(os.path.join(_GOLDEN, f"{name}.py")) as f:
        assert script == f.read()


def _replay_run(request, cassette, work_dir, run_id="replay"):
    """Drive the real graph end to end with both streams replayed."""
    import uuid

    from salome_agent.graph import build_graph

    llm = ReplayLLM(cassette)
    client = ReplayExecutorClient(cassette)
    graph = build_graph(batch=True)
    config = {"configurable": {"thread_id": uuid.uuid4().hex,
                               "llm_client": llm, "executor_client": client}}
    state = {
        "run_id": run_id, "user_request": request,
        "config": {"batch": True, "work_dir": work_dir,
                   "policy": {"tool_iteration_cap": 40, "static_retries": 1,
                              "exec_retries": 1, "exec_timeout_s": 120,
                              "tool_selection": "hints", "verify_cadence": "final",
                              "plan_detail": "balanced"}},
        "plan": [], "plan_approved": True, "stage_idx": 0, "plan_timing": None,
        "session": {"mode": "none", "pid": None, "salome_port": None, "name_prefix": ""},
        "study_ns": "salome_agent_run", "stages": {}, "last_error": None, "history": [],
        "final_script": None, "report": None, "fatal": None, "aborted": None,
    }
    out = graph.invoke(state, config=config)
    assert not out.get("fatal"), out.get("fatal")
    return out["final_script"]
