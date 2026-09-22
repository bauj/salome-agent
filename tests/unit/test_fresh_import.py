import json
import subprocess
import sys
from pathlib import Path


def test_planner_prompt_builds_in_a_fresh_interpreter():
    """MODULES is populated by import side effect; a fresh process must not
    depend on some other module having imported the specialists first.

    This test must run in a subprocess to catch import-order bugs that
    in-process tests would miss due to sys.modules pollution.
    """
    code = "from salome_agent.nodes.planner import PLANNER_SYSTEM_PROMPT; assert PLANNER_SYSTEM_PROMPT"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"Fresh import failed:\nstdout: {r.stdout}\nstderr: {r.stderr}"


def test_cli_path_loads_planner_in_fresh_interpreter():
    """Test that the CLI entry point followed by planner import works in
    a fresh interpreter. CLI importing ensures all dependencies are available.
    """
    code = (
        "import salome_agent.cli; "
        "from salome_agent.nodes.planner import PLANNER_SYSTEM_PROMPT; "
        "assert PLANNER_SYSTEM_PROMPT"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"CLI path failed:\nstdout: {r.stdout}\nstderr: {r.stderr}"


def test_geom_tools_registers_all_tools_in_a_fresh_interpreter():
    """executor.geom.__init__ registers every tool as an import side effect
    spread over 9 submodules; a shared-interpreter suite once passed green
    while a fresh process crashed (session 43). Also guards that the split
    reproduces the pre-split registration order exactly.
    """
    golden = Path(__file__).parent.parent / "fixtures" / "golden" / "list_tools.json"
    expected_names = [t["name"] for t in json.loads(golden.read_text())]
    code = (
        "import json\n"
        "from executor import geom_tools\n"
        "assert len(geom_tools.TOOLS) == 44, len(geom_tools.TOOLS)\n"
        "names = [t['name'] for t in geom_tools.list_tools()]\n"
        "print(json.dumps(names))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"Fresh import failed:\nstdout: {r.stdout}\nstderr: {r.stderr}"
    names = json.loads(r.stdout.strip().splitlines()[-1])
    assert names == expected_names
