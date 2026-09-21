#!/usr/bin/env python3
"""Turn a recorded run directory into a committed cassette.

Usage: scripts/make-cassette.py <run_dir> <cassette_name>

Reads <run_dir>/llm_trace.jsonl and <run_dir>/exec_trace.jsonl, keeps only
what replay needs (the provider message per LLM turn, and each executor
call's cmd/payload/response), and writes tests/fixtures/cassettes/<name>.json
plus tests/fixtures/golden/<name>.py from the run's result.py.
"""
import json
import os
import shutil
import sys


def main(run_dir: str, name: str) -> int:
    llm = []
    with open(os.path.join(run_dir, "llm_trace.jsonl")) as f:
        for line in f:
            entry = json.loads(line)
            message = entry["response"].get("message")
            if message is None:
                print(f"skipping a trace line with no message: {entry.get('t')}")
                continue
            # reasoning_content is dropped from history by design, so it is
            # dead weight in a cassette.
            llm.append({k: v for k, v in message.items() if k != "reasoning_content"})
    exec_calls = []
    with open(os.path.join(run_dir, "exec_trace.jsonl")) as f:
        for line in f:
            exec_calls.append(json.loads(line))
    root = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(os.path.join(root, "cassettes"), exist_ok=True)
    os.makedirs(os.path.join(root, "golden"), exist_ok=True)
    with open(os.path.join(root, "cassettes", f"{name}.json"), "w") as f:
        json.dump({"llm": llm, "exec": exec_calls}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    shutil.copyfile(os.path.join(run_dir, "result.py"),
                    os.path.join(root, "golden", f"{name}.py"))
    print(f"{name}: {len(llm)} llm turn(s), {len(exec_calls)} executor call(s)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
