"""Run salome-agent on the CADSmith benchmark prompts and score each result
against the benchmark's reference STL.

CADSmith: Barkley, Loghmani, Barati Farimani, "CADSmith: Multi-Agent CAD
Generation with Programmatic Geometric Validation", arXiv 2603.26512. The
dataset is not part of this repo: pass the folder holding its t*_*.jsonl
files and reference_stls/ with --data.

usage:
  cadsmith_bench.py run --data DIR --config CFG [--ids T2_001,...] [--tiers T2,T3]
                        [--name NAME] [--timeout S]
  cadsmith_bench.py summary NAME

Each prompt goes through the normal CLI (--batch -vv), one at a time. The
run's result.py is then rebuilt in a separate SALOME process
(cadsmith_export.py), which writes the final geometry to STL with its exact
volume, and the STL is scored against the reference. Prompts without a
reference STL are skipped.

Scores follow the paper's absolute (mm) mode: both meshes moved so their
bounding-box centres are at the origin, the generated one aligned to the
reference by ICP, 10k surface points each; Chamfer distance (mean squared,
both ways, mm^2), F1 at 1 mm, voxel IoU. The paper's IoU compares surface
voxels by grid index, each grid anchored at its own mesh's corner; it is
kept as iou_paper. iou is the filled volume on one shared grid. cd_placed and
f1_placed skip the centring and ICP: the prompts say where the part sits.

Results: ~/.local/share/salome-agent/bench/<name>/results.jsonl, one row per
prompt; progress.txt is the line to tail.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BENCH = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"),
                     "salome-agent", "bench")


def load_prompts(data: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(data, "t*_*.jsonl"))):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


# --- scoring ---------------------------------------------------------------

def _centred(mesh):
    m = mesh.copy()
    m.apply_translation(-(m.bounds[0] + m.bounds[1]) / 2.0)
    return m


def _nearest(a, b):
    from scipy.spatial import cKDTree
    return cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]


def _cd_f1(pts_gen, pts_ref, tau: float) -> tuple[float, float]:
    d_gen, d_ref = _nearest(pts_gen, pts_ref)
    cd = float((d_gen ** 2).mean() + (d_ref ** 2).mean())
    p, r = float((d_gen < tau).mean()), float((d_ref < tau).mean())
    return cd, (0.0 if p + r == 0 else 2 * p * r / (p + r))


def _iou_paper(gen, ref, pitch: float) -> float:
    a = set(map(tuple, gen.voxelized(pitch=pitch).sparse_indices))
    b = set(map(tuple, ref.voxelized(pitch=pitch).sparse_indices))
    return len(a & b) / len(a | b) if a | b else 0.0


def _iou_filled(gen, ref, pitch: float) -> float:
    import numpy as np
    cells = lambda m: set(map(tuple, np.round(m.voxelized(pitch=pitch).fill().points / pitch)
                              .astype(int)))
    a, b = cells(gen), cells(ref)
    return len(a & b) / len(a | b) if a | b else 0.0


def score(gen_stl: str, ref_stl: str, seed: int = 0, n_points: int = 10000,
          tau: float = 1.0) -> dict:
    import numpy as np
    import trimesh
    from trimesh.registration import icp
    np.random.seed(seed)
    gen_raw, ref_raw = trimesh.load(gen_stl), trimesh.load(ref_stl)
    cd_placed, f1_placed = _cd_f1(trimesh.sample.sample_surface(gen_raw, n_points)[0],
                                  trimesh.sample.sample_surface(ref_raw, n_points)[0], tau)
    gen, ref = _centred(gen_raw), _centred(ref_raw)
    try:
        matrix, _, _ = icp(trimesh.sample.sample_surface(gen, 5000)[0],
                           trimesh.sample.sample_surface(ref, 5000)[0], max_iterations=100)
        gen.apply_transform(matrix)
    except Exception:
        pass
    cd, f1 = _cd_f1(trimesh.sample.sample_surface(gen, n_points)[0],
                    trimesh.sample.sample_surface(ref, n_points)[0], tau)
    pitch = max(1.0, max(gen.extents.max(), ref.extents.max()) / 100.0)
    return {"cd": cd, "f1": f1, "iou_paper": _iou_paper(gen, ref, pitch),
            "iou": _iou_filled(gen, ref, pitch), "cd_placed": cd_placed,
            "f1_placed": f1_placed, "ref_volume": float(ref_raw.volume)}


# --- running ---------------------------------------------------------------

def _salome_env() -> dict:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":") if ".venv" not in p)
    return env


def export(salome_root: str, result_py: str, out_stl: str, out_json: str,
           log_path: str) -> dict:
    with open(log_path, "w") as log:
        subprocess.run([os.path.join(salome_root, "salome"), "shell", "--", "python3",
                        os.path.join(HERE, "cadsmith_export.py"), result_py, out_stl, out_json],
                       stdout=log, stderr=subprocess.STDOUT, env=_salome_env(), timeout=600)
    if not os.path.exists(out_json):
        return {"ok": False, "error": f"export wrote nothing, see {log_path}"}
    with open(out_json) as f:
        return json.load(f)


def run_agent(prompt: str, config: str, log_path: str, timeout: float) -> dict:
    cmd = [os.path.join(REPO, ".venv", "bin", "salome-agent"), prompt,
           "--batch", "-vv", "--config", config]
    t0 = time.time()
    with open(log_path, "w") as log:
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    with open(log_path) as f:
        m = re.search(r"output dir: (\S+)", f.read())
    return {"rc": rc, "seconds": round(time.time() - t0, 1),
            "run_dir": m.group(1) if m else None}


def _tokens(run_dir: str) -> dict:
    path = os.path.join(run_dir, "report.md")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        text = f.read()
    grab = lambda k: (int(m.group(1).replace(",", ""))
                      if (m := re.search(rf"- {k}: ([\d,]+)", text)) else None)
    mods = re.findall(r"stage \d+/\d+ \((\w+)\): ([a-z]+)\b", text)
    return {"llm_calls": grab("calls"), "input_tokens": grab("input tokens"),
            "output_tokens": grab("output tokens"),
            "modules": [m for m, _ in mods], "stages_ok": all(s == "ok" for _, s in mods)}


def cmd_run(args) -> int:
    with open(args.config) as f:
        salome_root = json.load(f)["salome"]["root"]
    data = os.path.abspath(args.data)
    out = os.path.join(BENCH, args.name)
    for sub in ("logs", "stl"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    results = os.path.join(out, "results.jsonl")
    done = set()
    if os.path.exists(results):
        with open(results) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    prompts = load_prompts(data)
    if args.ids:
        wanted = args.ids.split(",")
        prompts = [p for p in prompts if p["id"] in wanted]
    if args.tiers:
        prompts = [p for p in prompts if p["tier"] in args.tiers.split(",")]
    progress = open(os.path.join(out, "progress.txt"), "a", buffering=1)
    say = lambda s: (print(s, flush=True), progress.write(s + "\n"))
    say(f"== {args.name}: {len(prompts)} prompts, config {args.config}")
    for p in prompts:
        pid = p["id"]
        ref = os.path.join(data, "reference_stls", f"{pid}.stl")
        if pid in done:
            continue
        if not os.path.exists(ref):
            say(f"{pid}: skipped, no reference STL")
            continue
        row = {"id": pid, "tier": p["tier"], "prompt": p["prompt"]}
        say(f"{pid}: running")
        row.update(run_agent(p["prompt"], args.config,
                             os.path.join(out, "logs", f"{pid}.agent.log"), args.timeout))
        result_py = row["run_dir"] and os.path.join(row["run_dir"], "result.py")
        if row["run_dir"]:
            say(f"{pid}: run {os.path.basename(row['run_dir'])}  "
                f"tail -f {row['run_dir']}/llm_live.txt")
            row.update(_tokens(row["run_dir"]))
        if not result_py or not os.path.exists(result_py):
            row["status"] = "agent_failed"
        else:
            stl = os.path.join(out, "stl", f"{pid}.stl")
            ex = export(salome_root, result_py, stl, os.path.join(out, "stl", f"{pid}.json"),
                        os.path.join(out, "logs", f"{pid}.export.log"))
            row["export"] = ex
            if not ex.get("ok"):
                row["status"] = "export_failed"
            else:
                row["status"] = "scored"
                row["scores"] = score(stl, ref)
                rv = row["scores"]["ref_volume"]
                row["volume_err_pct"] = abs(ex["volume"] - rv) / rv * 100 if rv else None
        s = row.get("scores") or {}
        say(f"{pid}: {row['status']} {row['seconds']}s"
            + (f" cd={s['cd']:.3f} f1={s['f1']:.3f} iou={s['iou']:.3f} "
               f"vol_err={row['volume_err_pct']:.2f}%" if s else ""))
        with open(results, "a") as f:
            f.write(json.dumps(row) + "\n")
    say(f"== {args.name}: done")
    return cmd_summary(argparse.Namespace(name=args.name))


def _stat(vals, f):
    return f"{f(vals):.4f}" if vals else "-"


def cmd_summary(args) -> int:
    import statistics as st
    with open(os.path.join(BENCH, args.name, "results.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"{'tier':<5} {'n':>3} {'ok':>4} {'CD med':>8} {'CD mean':>8} {'F1 med':>7} "
          f"{'IoU med':>7} {'IoUp med':>8} {'CDplc med':>9} {'vol%med':>7}")
    for tier in sorted({r["tier"] for r in rows}) + ["all"]:
        rs = [r for r in rows if tier in ("all", r["tier"])]
        sc = [r for r in rs if r["status"] == "scored"]
        g = lambda k: [r["scores"][k] for r in sc]
        vol = [r["volume_err_pct"] for r in sc if r.get("volume_err_pct") is not None]
        print(f"{tier:<5} {len(rs):>3} {len(sc):>4} {_stat(g('cd'), st.median):>8} "
              f"{_stat(g('cd'), st.mean):>8} {_stat(g('f1'), st.median):>7} "
              f"{_stat(g('iou'), st.median):>7} {_stat(g('iou_paper'), st.median):>8} "
              f"{_stat(g('cd_placed'), st.median):>9} {_stat(vol, st.median):>7}")
    for r in rows:
        if r["status"] != "scored":
            print(f"  {r['id']}: {r['status']} (run {os.path.basename(r.get('run_dir') or '-')})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--ids")
    r.add_argument("--tiers")
    r.add_argument("--name", default=time.strftime("%Y%m%d-%H%M%S"))
    r.add_argument("--timeout", type=float, default=1800)
    s = sub.add_parser("summary")
    s.add_argument("name")
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_summary(args)


if __name__ == "__main__":
    sys.exit(main())
