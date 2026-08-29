from unittest.mock import MagicMock

from salome_agent.nodes.planner import plan


def test_repeated_module_yields_two_stages():
    llm = MagicMock()
    llm.chat.return_value = {
        "stages": [
            {"module": "geom", "intent": "make a 20mm cube", "params": {"size": 20}, "inputs": []},
            {"module": "geom", "intent": "fillet edges 2mm", "params": {"radius": 2}, "inputs": ["cube"]},
            {"module": "smesh", "intent": "mesh with 5mm tets", "params": {"max_size": 5}, "inputs": []},
        ]
    }
    stages = plan("make a 20mm cube, fillet its edges 2mm, mesh it with 5mm tetrahedra", llm)
    assert [s["module"] for s in stages] == ["geom", "geom", "smesh"]
    assert stages[1]["params"]["radius"] == 2  # mm, unconverted


def test_geom_only_request_yields_no_smesh_stage():
    llm = MagicMock()
    llm.chat.return_value = {"stages": [{"module": "geom", "intent": "make a box",
                                          "params": {}, "inputs": []}]}
    stages = plan("make a box", llm)
    assert len(stages) == 1
    assert stages[0]["module"] == "geom"
