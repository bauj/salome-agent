import subprocess
import sys


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
