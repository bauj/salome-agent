import json
import os
import subprocess

import pytest

from ._salome_root import SALOME_ROOT
from .conftest import _salome_subprocess_env

pytestmark = [pytest.mark.salome,
              pytest.mark.skipif(SALOME_ROOT is None,
                                 reason="set SALOME_ROOT or salome.root in config.json")]


def test_result_py_replays_mesh(agent_client, tmp_path):
    from salome_agent.build_graph import BuildGraph
    from salome_agent.nodes.assembler import render_script

    def ok(module, tool, **args):
        r = agent_client.call_tool(tool, args, module=module)
        assert r["ok"] is True, r.get("error")
        return r

    calls = {"shaper": [], "smesh": []}

    def rec(module, tool, **args):
        r = ok(module, tool, **args)
        calls[module].append({"tool": tool, "tool_args": args, "records": r.get("records", [])})
        return r

    box = rec("shaper", "make_box", dx=10, dy=20, dz=30, center=[5, 10, 15])["name"]
    top = ok("smesh", "get_subshapes", shape=box, sub_type="face",
             where={"sorted_by": "z", "take": "max", "count": 1})["subshapes"][0]["id"]
    rec("smesh", "name_subshapes", shape=box, ids=[top], name="lid")
    m = rec("smesh", "create_mesh", shape=box)["name"]
    rec("smesh", "mesh_tetra", mesh=m, max_size=5)
    rec("smesh", "group_on_geometry", mesh=m, group="lid")
    live = rec("smesh", "compute", mesh=m)

    g = BuildGraph()
    for i, (module, names) in enumerate((("shaper", [box]), ("smesh", [m]))):
        bid = g.add_block(0, f"stage {i + 1}/2: {module}", module, names)
        for c in calls[module]:
            g.append(bid, c["tool"], c["tool_args"], c["records"], 0, module=module)
    script = render_script(g, "replay")
    script += (f"\nimport json\nprint('COUNTS', json.dumps([{m}.NbTetras(), "
               f"{m}_lid.Size()]))\n")
    path = tmp_path / "result.py"
    path.write_text(script)
    out = subprocess.run([f"{SALOME_ROOT}/salome", "shell", "--", "python3", str(path)],
                         env=_salome_subprocess_env(), capture_output=True, text=True,
                         timeout=300)
    line = next(l for l in out.stdout.splitlines() if l.startswith("COUNTS"))
    tetra, lid = json.loads(line.split(" ", 1)[1])
    assert tetra == live["tetra"] and lid == live["groups"]["lid"]
