import json
import os

import salome_agent.specialists.geom  # noqa: F401 — trigger GEOM_SPEC registration
import salome_agent.specialists.script_modules  # noqa: F401 — trigger SMESH_SPEC/PARAVIS_SPEC registration

_FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden")


def _read(name):
    with open(os.path.join(_FIX, name)) as f:
        return f.read()


def test_list_tools_wire_output_is_byte_identical():
    from executor.geom_tools import list_tools

    got = json.dumps(list_tools(), ensure_ascii=False, indent=2) + "\n"
    assert got == _read("list_tools.json")


def test_list_tools_structure_is_unchanged():
    from executor.geom_tools import list_tools

    assert list_tools() == json.loads(_read("list_tools.json"))


def test_planner_system_prompt_is_byte_identical():
    from salome_agent.nodes.planner import PLANNER_SYSTEM_PROMPT

    assert PLANNER_SYSTEM_PROMPT == _read("planner_prompt.txt")
