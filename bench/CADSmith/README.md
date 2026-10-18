# CADSmith benchmark

Runs salome-agent on the 95 prompts of the CADSmith benchmark (Barkley, Loghmani, Barati Farimani, arXiv 2603.26512) and scores each result against the benchmark's reference STL.

The dataset is not part of this repo. Clone it next to salome-agent:

```bash
git clone https://github.com/jabarkle/CADSmith.git ../CADSmith
```

`--data` is then `../CADSmith/data/dataset_v2` (the folder with `t*_*.jsonl` and `reference_stls/`).

## Setup

1. Install the scoring dependencies (trimesh, scipy, rtree) into the agent's venv:

   ```bash
   .venv/bin/pip install -e '.[bench]'
   ```

2. Write a config for the model to test: copy `config.example.json` and set `llm.base_url`, `llm.api_key`, `llm.model` (plus any `extra_body`, e.g. `{"reasoning_effort": "high"}`) and `salome.root`. One config per model, e.g. `config_gptoss.json`.

3. Start the headless agent server and leave it running (ready when `~/.salome_agent/server.json` appears):

   ```bash
   SALOME_ROOT=/path/to/your/SALOME scripts/start-headless-server.sh
   ```

   One bench run at a time: never point two runs, a test or a GUI session at the same server.

## Run

```bash
.venv/bin/python bench/CADSmith/cadsmith_bench.py run \
    --data ../CADSmith/data/dataset_v2 --config config_gptoss.json --name gptoss
```

- `--ids T2_004,T3_013` or `--tiers T1,T2` runs a subset; `--timeout` is per prompt (default 1800 s).
- Each prompt goes through the normal CLI, then its `result.py` is rebuilt in a separate SALOME process and exported to STL with its exact volume.
- Output: `~/.local/share/salome-agent/bench/<name>/` (`results.jsonl`, `stl/`, `logs/`). Follow progress with `tail -f ~/.local/share/salome-agent/bench/<name>/progress.txt`; each prompt's line also gives the `tail -f` command for its run.
- Running the same `--name` again resumes: prompts already in `results.jsonl` are skipped.
- The full 95 prompts take hours; try `--tiers T1` first.

## Compare models

Run each model under its own name, then:

```bash
.venv/bin/python bench/CADSmith/cadsmith_bench.py summary gemma4
.venv/bin/python bench/CADSmith/cadsmith_bench.py summary gptoss
```

`summary` prints one table per tier and lists the prompts that did not score. Columns:

- `CD`, `F1`, `IoU`: the paper's metrics (absolute mm, centred, ICP-aligned). Their point-to-point matching penalises big parts even when exact, so do not compare them across parts.
- `CDs`, `F1s`, `d99s`: point-to-surface distances after alignment; `d99sp` the same as placed (no alignment), which also checks the position the prompt asked for.
- `vol%`: exact volume error.

A part counts as a match when its largest point-to-surface distance as placed is at most 0.2 mm.

To look at the parts in 3D, each over its reference:

```bash
.venv/bin/python bench/CADSmith/cadsmith_bench.py view gptoss --data ../CADSmith/data/dataset_v2
```

This writes `<name>/compare.html` and opens it (three.js from a CDN, so it needs a network connection; `--no-open` only writes it). With several names, `view gemma4 gptoss --data ...` puts all parts in one page and a part found in several runs shows the last name given, so put the model you want to see last. Each part gets a verdict from its scores: `match`, `moved` (right shape, placed elsewhere), `differs`, or `failed` (no result). `<name>/notes.json`, e.g. `{"T1_031": {"verdict": "dataset", "note": "..."}}`, overrides it with a reviewed one; reviewed verdicts are also `turned`, `miss` (agent miss) and `dataset` (dataset fault).

## Other commands

- `rescore NAME --data DIR`: recompute every score from the saved STLs (after a change to the scoring).
- `reexport NAME --data DIR --config CFG`: rebuild every run's `result.py` and export its STL again (after a change to `cadsmith_export.py`), then rescore.
- `report NAME`: turn `<name>/report/report.md` (images in `report/img/`) into one self-contained `report.html` with pandoc.

## Known dataset faults

Do not count these as agent misses:

- T1_031: the reference STL is another part.
- T2_019, T3_025: a feature in the reference cuts nothing.
- T2_020: the reference holes sit at a different height from the prompt's.
- T3_019, T3_022: the prompts describe disconnected solids.
- T1_012, T1_013, T2_005, T2_006, T2_007, T2_011: the wording allows more than one reading.
