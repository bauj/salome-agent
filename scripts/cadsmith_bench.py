"""Run salome-agent on the CADSmith benchmark prompts and score each result
against the benchmark's reference STL.

CADSmith: Barkley, Loghmani, Barati Farimani, "CADSmith: Multi-Agent CAD
Generation with Programmatic Geometric Validation", arXiv 2603.26512. The
dataset is not part of this repo: pass the folder holding its t*_*.jsonl
files and reference_stls/ with --data.

usage:
  cadsmith_bench.py run --data DIR --config CFG [--ids T2_001,...] [--tiers T2,T3]
                        [--name NAME] [--timeout S]
  cadsmith_bench.py rescore NAME --data DIR
  cadsmith_bench.py reexport NAME --data DIR --config CFG
  cadsmith_bench.py summary NAME
  cadsmith_bench.py view NAME [NAME ...] --data DIR [--no-open]

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

Point to point, two samples of the same surface sit about half a point
spacing apart, so a big part scores a worse CD and F1 than a small one even
when perfect. The *_surf scores measure each point's distance to the other
mesh's surface instead (cd, f1 at 1 mm, d99 = 99th percentile, dmax), after
an ICP alignment onto the reference surface (the paper's point-to-point ICP
leaves ~0.2 mm of noise on identical parts) and, as *_surf_placed, before it. rescore recomputes every
score from the saved STLs; reexport first rebuilds every scored run's
result.py and exports its STL again (after a change to cadsmith_export.py).

Results: ~/.local/share/salome-agent/bench/<name>/results.jsonl, one row per
prompt; progress.txt is the line to tail.

view writes <first name>/compare.html, a 3D viewer of every part over its
reference (cadsmith_compare.html filled with the STLs; three.js from a CDN),
and opens it. A part found in several runs shows the last one named. Each
part gets a verdict from its scores; <name>/notes.json, {"T1_013":
{"verdict": "miss", "note": "..."}}, overrides it with a reviewed one.
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


def _surface(gen, ref, n_points: int, tau: float) -> dict:
    """Each sampled point's distance to the other mesh's surface, both ways:
    unlike point to point, a perfect part scores ~0 at any size."""
    import numpy as np
    import trimesh
    d_gen = trimesh.proximity.closest_point(ref, trimesh.sample.sample_surface(gen, n_points)[0])[1]
    d_ref = trimesh.proximity.closest_point(gen, trimesh.sample.sample_surface(ref, n_points)[0])[1]
    p, r = float((d_gen < tau).mean()), float((d_ref < tau).mean())
    return {"cd": float((d_gen ** 2).mean() + (d_ref ** 2).mean()),
            "f1": 0.0 if p + r == 0 else 2 * p * r / (p + r),
            "d99": float(np.percentile(np.concatenate([d_gen, d_ref]), 99)),
            "dmax": float(max(d_gen.max(), d_ref.max()))}


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
    placed = _surface(gen_raw, ref_raw, n_points, tau)
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
    out = {"cd": cd, "f1": f1, "iou_paper": _iou_paper(gen, ref, pitch),
           "iou": _iou_filled(gen, ref, pitch), "cd_placed": cd_placed,
           "f1_placed": f1_placed, "ref_volume": float(ref_raw.volume)}
    gen_s = _centred(gen_raw)
    try:
        matrix, _, _ = icp(trimesh.sample.sample_surface(gen_s, 5000)[0], ref,
                           max_iterations=100)
        gen_s.apply_transform(matrix)
    except Exception:
        pass
    out.update({f"{k}_surf": v for k, v in _surface(gen_s, ref, n_points, tau).items()})
    out.update({f"{k}_surf_placed": v for k, v in placed.items()})
    return out


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


def cmd_rescore(args) -> int:
    data = os.path.abspath(args.data)
    out = os.path.join(BENCH, args.name)
    path = os.path.join(out, "results.jsonl")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        if r["status"] != "scored":
            continue
        r["scores"] = score(os.path.join(out, "stl", f"{r['id']}.stl"),
                            os.path.join(data, "reference_stls", f"{r['id']}.stl"))
        rv = r["scores"]["ref_volume"]
        r["volume_err_pct"] = abs(r["export"]["volume"] - rv) / rv * 100 if rv else None
        print(f"{r['id']}: rescored", flush=True)
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    return cmd_summary(args)


def _verdict(row: dict) -> str:
    if row["status"] != "scored":
        return "failed"
    s = row["scores"]
    if s["dmax_surf_placed"] <= 0.2:
        return "match"
    return "moved" if s["d99_surf"] <= 0.2 else "differs"


def cmd_view(args) -> int:
    import base64
    data = os.path.abspath(args.data)
    prompts = {p["id"]: p for p in load_prompts(data)}
    rows, notes = {}, {}
    for name in args.names:
        out = os.path.join(BENCH, name)
        with open(os.path.join(out, "results.jsonl")) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["id"]] = (out, r)
        if os.path.exists(os.path.join(out, "notes.json")):
            with open(os.path.join(out, "notes.json")) as f:
                notes.update(json.load(f))
    b64 = lambda path: base64.b64encode(open(path, "rb").read()).decode()
    parts = []
    for pid in sorted(rows):
        out, r = rows[pid]
        s, note = r.get("scores") or {}, notes.get(pid, {})
        stl = os.path.join(out, "stl", f"{pid}.stl")
        parts.append({
            "id": pid, "tier": r["tier"], "prompt": prompts[pid]["prompt"],
            "status": r["status"], "verdict": note.get("verdict") or _verdict(r),
            "note": note.get("note", ""), "run_dir": r.get("run_dir"),
            "seconds": r.get("seconds"), "vol": r.get("volume_err_pct"),
            "gap": s.get("dmax_surf_placed"), "d99": s.get("d99_surf"),
            "f1s": s.get("f1_surf"), "f1p": s.get("f1"),
            "ours": b64(stl) if r["status"] == "scored" else None,
            "ref": b64(os.path.join(data, "reference_stls", f"{pid}.stl"))})
    with open(os.path.join(HERE, "cadsmith_compare.html")) as f:
        page = f.read()
    title = " + ".join(args.names)
    page = (page.replace("__TITLE__", title)
            .replace("__PARTS__", json.dumps(parts).replace("</", "<\\/")))
    path = os.path.join(BENCH, args.names[0], "compare.html")
    with open(path, "w") as f:
        f.write(page)
    print(path)
    if not args.no_open:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def cmd_reexport(args) -> int:
    with open(args.config) as f:
        salome_root = json.load(f)["salome"]["root"]
    out = os.path.join(BENCH, args.name)
    with open(os.path.join(out, "results.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        if r["status"] != "scored":
            continue
        pid = r["id"]
        r["export"] = export(salome_root, os.path.join(r["run_dir"], "result.py"),
                             os.path.join(out, "stl", f"{pid}.stl"),
                             os.path.join(out, "stl", f"{pid}.json"),
                             os.path.join(out, "logs", f"{pid}.export.log"))
        if not r["export"].get("ok"):
            r["status"] = "export_failed"
        print(f"{pid}: exported {'ok' if r['export'].get('ok') else 'FAILED'}", flush=True)
    with open(os.path.join(out, "results.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    return cmd_rescore(args)


def _stat(vals, f):
    return f"{f(vals):.4f}" if vals else "-"


def cmd_summary(args) -> int:
    import statistics as st
    with open(os.path.join(BENCH, args.name, "results.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"{'tier':<5} {'n':>3} {'ok':>4} {'CD med':>8} {'CD mean':>8} {'F1 med':>7} "
          f"{'IoU med':>7} {'IoUp med':>8} {'CDs med':>8} {'F1s med':>7} {'d99s med':>8} "
          f"{'d99sp med':>9} {'vol%med':>7}")
    for tier in sorted({r["tier"] for r in rows}) + ["all"]:
        rs = [r for r in rows if tier in ("all", r["tier"])]
        sc = [r for r in rs if r["status"] == "scored"]
        g = lambda k: [r["scores"][k] for r in sc]
        vol = [r["volume_err_pct"] for r in sc if r.get("volume_err_pct") is not None]
        print(f"{tier:<5} {len(rs):>3} {len(sc):>4} {_stat(g('cd'), st.median):>8} "
              f"{_stat(g('cd'), st.mean):>8} {_stat(g('f1'), st.median):>7} "
              f"{_stat(g('iou'), st.median):>7} {_stat(g('iou_paper'), st.median):>8} "
              f"{_stat(g('cd_surf'), st.median):>8} {_stat(g('f1_surf'), st.median):>7} "
              f"{_stat(g('d99_surf'), st.median):>8} {_stat(g('d99_surf_placed'), st.median):>9} "
              f"{_stat(vol, st.median):>7}")
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
    rs = sub.add_parser("rescore")
    rs.add_argument("name")
    rs.add_argument("--data", required=True)
    re_ = sub.add_parser("reexport")
    re_.add_argument("name")
    re_.add_argument("--data", required=True)
    re_.add_argument("--config", required=True)
    s = sub.add_parser("summary")
    s.add_argument("name")
    v = sub.add_parser("view")
    v.add_argument("names", nargs="+")
    v.add_argument("--data", required=True)
    v.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)
    return {"run": cmd_run, "rescore": cmd_rescore, "reexport": cmd_reexport,
            "summary": cmd_summary,
            "view": cmd_view}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
