from tests.replay import ReplayExecutorClient, ReplayLLM, load_cassette


def test_replay_drives_the_geom_specialist_to_a_call_log():
    from salome_agent.specialists.geom import run_geom_specialist

    cassette = load_cassette("synthetic_box")
    llm = ReplayLLM(cassette)
    client = ReplayExecutorClient(cassette)
    result = run_geom_specialist(
        {"module": "geom", "intent": "a box", "params": {}, "inputs": []},
        llm, client, iteration_cap=12, verify_cadence="none")
    assert result["result_names"] == ["box_1"]
    assert [c["fn"] for c in result["call_log"]] == ["MakeBoxDXDYDZ"]
    assert llm.exhausted


def test_replay_executor_rejects_a_diverging_call():
    import pytest

    cassette = load_cassette("synthetic_box")
    client = ReplayExecutorClient(cassette)
    client.list_tools()
    with pytest.raises(AssertionError):
        client.call_tool("make_sphere", {"radius": 5})


def test_generated_script_from_a_replayed_call_log():
    from salome_agent.codegen import generate
    from salome_agent.specialists.geom import run_geom_specialist

    cassette = load_cassette("synthetic_box")
    result = run_geom_specialist(
        {"module": "geom", "intent": "a box", "params": {}, "inputs": []},
        ReplayLLM(cassette), ReplayExecutorClient(cassette),
        iteration_cap=12, verify_cadence="none")
    src = generate(result["call_log"], result["result_names"])
    assert src == ("box_1 = geompy.MakeBoxDXDYDZ(10, 20, 30)\n"
                   "geompy.addToStudy(box_1, 'box_1')")
