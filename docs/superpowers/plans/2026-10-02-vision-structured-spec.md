# Vision Structured Spec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The vision agent returns a per-part feature spec, validated mechanically (with a fix loop) before planning; the planner refers to parts by id and each geom stage receives its parts' exact JSON.

**Architecture:** New pure-Python module `salome_agent/nodes/vision_spec.py` owns the spec: validation, conflict attachment and the text renderings for planner, geom, amend and report. `nodes/vision.py` keeps the prompt and the call + fix loop. `graph.py` stores the spec as state `image_spec` and slices it per geom stage from the stage's `parts` list. The prose path (`format_description`, `image_block`, state `image_description`) is removed.

**Tech Stack:** Python 3, LangGraph state graph, pytest (`.venv/bin/python -m pytest`), OpenAI-compatible chat API via `salome_agent.llm.LLMClient`.

**Spec:** `docs/superpowers/specs/2026-10-02-vision-structured-spec-design.md`

## Global Constraints

- Branch `feat/vision-spec` (already created); never commit to `main` directly.
- Commit messages: subject line only, no body, no trailers. Stage named files only (never `git add -A`).
- Never push; the user pushes.
- Validator is pure Python: no SALOME import, no executor call.
- Point format everywhere in the spec: `"x,y,z"` strings, each coordinate a number or `"~N"`. No point names.
- Estimates: a number given as a string `"~N"`; plain numbers are read or derived.
- Arc-radius tolerance and plane tolerance: `TOL = 0.01` (mm).
- Config knob: `llm.vision_fix_retries`, default `2`.
- Provider-agnostic: nothing provider-specific in source.
- No distro or install paths in code, comments or docs.
- Minimal comments, matching the surrounding code.
- Unit tests: `.venv/bin/python -m pytest tests/unit -q`. Live tests (`tests/salome`) refuse to run while the headless server is up.

## Review Focus

1. The vision model answers a non-object (a list, or `{}`) — expect validation errors fed back, then a spec with top-level `conflicts` and no parts; the planner and report must not crash. Pinned in Task 2 (`attach_conflicts` on a non-dict) and Task 4 (`compact_spec` of an empty-parts spec).
2. The planner ignores `parts` entirely (old habits, weak model) — expect every geom stage to receive the whole spec rather than nothing (no regression vs today). Pinned in Task 2 (`stage_part_ids` fallback) and Task 5 (stage injection test).
3. The planner returns `parts` as a bare string `"P1"` instead of a list — expect it treated as `["P1"]`. Pinned in Task 4.
4. Coordinates with `~` estimates inside points (`"~40,0,0"`) — expect them accepted by the validator and rendered unchanged to geom. Pinned in Task 1.
5. A follow-up chat turn with new images — expect the new spec to replace the old one and amend to receive the rendering. Pinned in Task 5 (amend test).

---

## File Structure

- Create `salome_agent/nodes/vision_spec.py` — parsing helpers, `validate_spec`, `format_errors`, `attach_conflicts`, `all_conflicts`, `compact_spec`, `parts_block`, `amend_block`, `stage_part_ids`, `unclaimed_parts`.
- Create `tests/unit/test_vision_spec.py`.
- Modify `salome_agent/nodes/vision.py` — new prompt, fix loop; remove `format_description` and `_as_list`.
- Modify `tests/unit/test_vision.py` — rewrite.
- Modify `salome_agent/config.py`, `README.md` — `vision_fix_retries`.
- Modify `salome_agent/nodes/planner.py` — prompt paragraph, `image_spec` parameter, parts check; regenerate `tests/fixtures/golden/planner_prompt.txt`.
- Modify `salome_agent/state.py`, `salome_agent/runner.py`, `salome_agent/graph.py`, `salome_agent/specialists/geom.py` — wiring.
- Modify `salome_agent/nodes/report.py` — "Reference image" section.
- Modify tests: `tests/unit/test_planner.py`, `tests/unit/test_graph.py`, `tests/unit/test_graph_amend.py`, `tests/unit/test_specialist_geom.py`, `tests/unit/test_report.py`, `tests/unit/test_config.py`.

Spec deviations decided while planning (fold into the spec doc in Task 7):
- The planner's unknown-part feedback is its own one-shot re-ask inside `plan()` (the LLM client's JSON-retry loop only handles unparseable JSON); ids still unknown after it are dropped with a warning.
- The "assumed 2D" note goes in `report.md` (section "Reference image") and the log, not the chat turn summary.
- If no stage of a plan lists any `parts`, every geom stage gets the whole spec (fallback, no regression).

---

### Task 1: Spec validator

**Files:**
- Create: `salome_agent/nodes/vision_spec.py`
- Test: `tests/unit/test_vision_spec.py`

**Interfaces:**
- Produces:
  - `TOL: float = 0.01`
  - `parse_num(v) -> float | None` — number, or `"~N"` string; `None` otherwise (bool is not a number).
  - `parse_point(v) -> tuple[float, float, float] | None` — `"x,y,z"`, each coordinate via `parse_num`.
  - `validate_spec(spec) -> list[tuple[str | None, str]]` — `(part_id or None, message)`; empty list = valid.
  - `outline_extent(outline: dict, plane: str) -> list[tuple[float, float]] | None` — in-plane points bounding the outline (arcs as their full circle), `None` if unparseable. Used by Task 2.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vision_spec.py`:

```python
import copy

from salome_agent.nodes.vision_spec import parse_num, parse_point, validate_spec


def p1_spec():
    """The p1.png plate as a correct spec (clean benchmark: area 19380, 8 wires)."""
    return {
        "dimensionality": {"value": "2d", "assumed": False, "reason": "flat plate, request says 2D"},
        "summary": "flat plate: round top lobe over a narrower body with a round bottom end",
        "views": [{"image": 1, "view": "front", "shows": "full outline"}],
        "dimensions_read": [{"text": "R96", "means": "top lobe radius, centre O"}],
        "datum": "O = centre of top lobe; +X right, +Y up",
        "parts": [{
            "id": "P1", "name": "plate",
            "base": {"kind": "profile", "plane": "XY", "extrude": None, "revolve": None,
                     "profile": {
                         "start": "96,0,0",
                         "segments": [
                             {"to": "-96,0,0", "through": "0,96,0"},
                             {"to": "-54,0,0", "fillet": 20},
                             {"to": "-54,-80,0"},
                             {"to": "54,-80,0", "through": "0,-134,0"},
                             {"to": "54,0,0", "fillet": 20},
                             {"to": "96,0,0"}],
                         "holes": [
                             {"center": "0,0,0", "radius": 25},
                             {"center": "55.426,32,0", "radius": 15},
                             {"center": "0,64,0", "radius": 15},
                             {"center": "-55.426,32,0", "radius": 15},
                             {"center": "0,-80,0", "radius": 20},
                             {"slot": {"center": "0,-80,0", "radius": 35, "width": 18,
                                       "from_angle": -60, "to_angle": 60}},
                             {"slot": {"center": "0,-80,0", "radius": 35, "width": 18,
                                       "from_angle": 120, "to_angle": 240}}]}},
            "placement": {"origin": "0,0,0", "normal": "0,0,1"},
            "features": [],
            "relation": None}],
        "uncertainties": []}


def box_spec():
    return {
        "dimensionality": {"value": "3d", "assumed": False, "reason": "photo"},
        "summary": "a block with a hole", "views": [], "dimensions_read": [], "datum": "corner",
        "parts": [{"id": "P1", "name": "block",
                   "base": {"kind": "primitive", "shape": "box",
                            "dims": {"dx": "~40", "dy": "~30", "dz": 10}},
                   "placement": {"origin": "~0,0,0", "normal": "0,0,1"},
                   "features": [{"type": "hole", "center": "20,15,10", "diameter": 8,
                                 "depth": "through"},
                                {"type": "pattern", "of": 1, "kind": "linear", "count": 3}]},
                  {"id": "P2", "name": "lid",
                   "base": {"kind": "approx", "description": "rounded cap", "dims": {"d": "~30"}},
                   "placement": {"origin": "20,15,10", "normal": "0,0,1"},
                   "ops": [{"op": "fuse", "with": "P1"}]}],
        "uncertainties": []}


def msgs(errors):
    return [m for _, m in errors]


def test_parse_num_takes_numbers_and_estimates():
    assert parse_num(3) == 3.0
    assert parse_num("~40") == 40.0
    assert parse_num("~ -2.5") == -2.5
    assert parse_num("forty") is None
    assert parse_num(True) is None


def test_parse_point_takes_estimated_coordinates():
    assert parse_point("1,2,3") == (1.0, 2.0, 3.0)
    assert parse_point("~40, 0, 0") == (40.0, 0.0, 0.0)
    assert parse_point("1,2") is None
    assert parse_point("A") is None


def test_correct_specs_pass():
    assert validate_spec(p1_spec()) == []
    assert validate_spec(box_spec()) == []


def test_non_object_spec_is_one_error():
    assert validate_spec([1, 2]) == [(None, "the spec must be a JSON object")]


def test_missing_parts_and_bad_dimensionality():
    errs = validate_spec({"dimensionality": {"value": "flat"}, "parts": []})
    assert (None, 'dimensionality.value must be "2d" or "3d"') in errs
    assert (None, "parts must be a non-empty list") in errs


def test_duplicate_ids():
    spec = box_spec()
    spec["parts"][1]["id"] = "P1"
    assert ("P1", "duplicate part id") in validate_spec(spec)


def test_arc_end_off_its_radius():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["segments"][3] = {"to": "54,-78,0", "center": "0,-80,0"}
    (pid, msg), = validate_spec(spec)
    assert pid == "P1"
    assert msg.startswith("base.profile segment 4: arc start (-54, -80, 0) is 54 from centre (0, -80, 0)")
    assert "both ends must lie on one radius" in msg


def test_collinear_through_point():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["segments"][2] = {"to": "-54,-80,0", "through": "-54,-40,0"}
    (_, msg), = validate_spec(spec)
    assert "segment 3: through point (-54, -40, 0) is on the line" in msg


def test_degenerate_outline():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"] = {"start": "0,0,0",
                                           "segments": [{"to": "10,0,0"}, {"to": "20,0,0"}]}
    (_, msg), = validate_spec(spec)
    assert "degenerate" in msg


def test_point_off_the_profile_plane():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["segments"][2] = {"to": "-54,-80,5"}
    assert any("is not in the XY plane" in m for m in msgs(validate_spec(spec)))


def test_hole_outside_the_outline():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["holes"].append({"center": "200,0,0", "radius": 5})
    (_, msg), = validate_spec(spec)
    assert msg.startswith("base.profile.holes[8]: centre (200, 0) lies outside the outline")


def test_bad_estimate_string():
    spec = box_spec()
    spec["parts"][0]["base"]["dims"]["dx"] = "about 40"
    assert ("P1", 'base.dims.dx must be a number or "~N"') in validate_spec(spec)


def test_pattern_must_point_at_another_feature():
    spec = box_spec()
    spec["parts"][0]["features"][1]["of"] = 2
    assert ("P1", "features[2].of must be the 1-based index of another feature (1-2)") \
        in validate_spec(spec)


def test_ops_must_name_another_part():
    spec = box_spec()
    spec["parts"][1]["ops"] = [{"op": "fuse", "with": "P9"}]
    assert ("P2", "ops[1].with must name another part (P1, P2)") in validate_spec(spec)


def test_2d_target_has_no_extrusion_and_no_primitives():
    spec = p1_spec()
    spec["parts"][0]["base"]["extrude"] = {"height": 5}
    assert ("P1", "base: a 2d target has no extrude or revolve") in validate_spec(spec)
    spec = box_spec()
    spec["dimensionality"]["value"] = "2d"
    assert ("P1", "base: a 2d target is built from profile parts only") in validate_spec(spec)


def test_extrude_and_revolve_together():
    spec = p1_spec()
    spec["dimensionality"]["value"] = "3d"
    base = spec["parts"][0]["base"]
    base["extrude"] = {"height": 5}
    base["revolve"] = {"point": "0,0,0", "direction": "0,1,0", "angle": 360}
    assert ("P1", "base: give extrude or revolve, not both") in validate_spec(spec)


def test_spec_hole_cannot_name_a_wire():
    spec = p1_spec()
    spec["parts"][0]["base"]["profile"]["holes"] = [{"wire": "W1"}]
    assert ("P1", "base.profile.holes[1]: a spec hole cannot name a wire; give its outline") \
        in validate_spec(spec)


def test_validator_does_not_mutate_the_spec():
    spec = p1_spec()
    before = copy.deepcopy(spec)
    validate_spec(spec)
    assert spec == before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_vision_spec.py -q`
Expected: FAIL / collection error `ModuleNotFoundError: No module named 'salome_agent.nodes.vision_spec'`

- [ ] **Step 3: Write the implementation**

Create `salome_agent/nodes/vision_spec.py`:

```python
"""The vision agent's structured spec (task 286): mechanical validation, and the
renderings the planner, geom stages, amend and the report get."""
import math
import re

TOL = 0.01
PLANE_AXES = {"XY": (0, 1), "YZ": (1, 2), "XZ": (0, 2)}
PLANE_FIXED = {"XY": 2, "YZ": 0, "XZ": 1}
PRIMITIVES = ("box", "cone", "cylinder", "sphere", "torus")
FEATURES = ("hole", "fillet", "chamfer", "pattern")

_NUM = re.compile(r"^~?\s*-?\d+(\.\d+)?$")


def parse_num(v):
    """A number, or an estimate given as "~N"; None when it is neither."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and _NUM.match(v.strip()):
        return float(v.strip().lstrip("~").strip())
    return None


def parse_point(v):
    """"x,y,z" (any coordinate may be "~N") -> (x, y, z); None otherwise."""
    if not isinstance(v, str):
        return None
    bits = v.split(",")
    if len(bits) != 3:
        return None
    vals = [parse_num(b.strip()) for b in bits]
    return None if None in vals else tuple(vals)


def _fmt(p) -> str:
    return "(" + ", ".join(f"{c:g}" for c in p) + ")"


def _flat(p, plane):
    a, b = PLANE_AXES[plane]
    return (p[a], p[b])


def _off_line(p, q, r) -> float:
    """Distance from q to the line p-r (inf when p and r coincide)."""
    d = math.dist(p, r)
    if d < TOL:
        return math.inf
    return abs((r[0] - p[0]) * (q[1] - p[1]) - (r[1] - p[1]) * (q[0] - p[0])) / d


def _circumcircle(p, q, r):
    ax, ay = p
    bx, by = q
    cx, cy = r
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return (ux, uy), math.dist((ux, uy), p)


def _check_outline(outline, plane, where, err):
    """Check one {start, segments} outline; return the in-plane points that bound
    it (arcs as their full circle), or None when it cannot be traced."""
    start = parse_point(outline.get("start"))
    segs = outline.get("segments")
    if start is None:
        err(f'{where}.start must be an "x,y,z" point')
        return None
    if not isinstance(segs, list) or not segs:
        err(f"{where}.segments must be a non-empty list")
        return None
    fixed = PLANE_FIXED[plane]

    def on_plane(p, label):
        if abs(p[fixed] - start[fixed]) > TOL:
            err(f"{where} {label}: {_fmt(p)} is not in the {plane} plane of start {_fmt(start)}")

    extent = [_flat(start, plane)]
    corners = [_flat(start, plane)]
    straight_only = True
    cur = start
    for k, seg in enumerate(segs, 1):
        label = f"segment {k}"
        to = parse_point(seg.get("to")) if isinstance(seg, dict) else None
        if to is None:
            err(f'{where} {label}: "to" must be an "x,y,z" point')
            return None
        on_plane(to, label)
        forms = [key for key in ("center", "through", "via") if seg.get(key) is not None]
        if len(forms) > 1:
            err(f"{where} {label}: give one of center, through or via, not {' and '.join(forms)}")
        elif forms == ["center"]:
            c = parse_point(seg["center"])
            if c is None:
                err(f'{where} {label}: center must be an "x,y,z" point')
            else:
                straight_only = False
                r0, r1 = math.dist(cur, c), math.dist(to, c)
                if abs(r0 - r1) > TOL:
                    err(f"{where} {label}: arc start {_fmt(cur)} is {r0:g} from centre {_fmt(c)} "
                        f"but its end {_fmt(to)} is {r1:g}; both ends must lie on one radius")
                cx, cy = _flat(c, plane)
                r = max(r0, r1)
                extent += [(cx - r, cy - r), (cx + r, cy + r)]
        elif forms == ["through"]:
            t = parse_point(seg["through"])
            if t is None:
                err(f'{where} {label}: through must be an "x,y,z" point')
            else:
                straight_only = False
                on_plane(t, label)
                p, q, r = _flat(cur, plane), _flat(t, plane), _flat(to, plane)
                circle = _circumcircle(p, q, r)
                if _off_line(p, q, r) <= TOL or circle is None:
                    err(f"{where} {label}: through point {_fmt(t)} is on the line from "
                        f"{_fmt(cur)} to {_fmt(to)}, so it defines no arc")
                else:
                    (cx, cy), rad = circle
                    extent += [(cx - rad, cy - rad), (cx + rad, cy + rad)]
        elif forms == ["via"]:
            via = ([parse_point(v) for v in seg["via"]]
                   if isinstance(seg["via"], list) and seg["via"] else [None])
            if None in via:
                err(f'{where} {label}: via must be a list of "x,y,z" points')
            else:
                straight_only = False
                extent += [_flat(v, plane) for v in via]
        extent.append(_flat(to, plane))
        corners.append(_flat(to, plane))
        cur = to
    distinct = []
    for c in corners:
        if all(math.dist(c, d) > TOL for d in distinct):
            distinct.append(c)
    flat_line = (len(distinct) < 3 or
                 all(_off_line(distinct[0], q, distinct[1]) <= TOL for q in distinct[2:]))
    if len(distinct) < 2 or (straight_only and flat_line):
        err(f"{where}: the outline is degenerate (its corners lie on one line)")
    return extent


def outline_extent(outline, plane):
    if not isinstance(outline, dict) or plane not in PLANE_AXES:
        return None
    return _check_outline(outline, plane, "", lambda m: None)


def _box(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _check_hole(hole, plane, box, where, err):
    x0, y0, x1, y1 = box

    def inside(q, label):
        if not (x0 - TOL <= q[0] <= x1 + TOL and y0 - TOL <= q[1] <= y1 + TOL):
            err(f"{where}: {label} {_fmt(q)} lies outside the outline "
                f"({x0:g}..{x1:g} by {y0:g}..{y1:g} in the {plane} plane)")
            return False
        return True

    if not isinstance(hole, dict):
        err(f"{where} must be an object")
    elif "segments" in hole:
        pts = _check_outline(hole, plane, where, err)
        for q in pts or []:
            if not inside(q, "point"):
                break
    elif "slot" in hole:
        s = hole["slot"]
        if not isinstance(s, dict) or (parse_num(s.get("width")) or 0) <= 0:
            err(f"{where}.slot.width must be a positive number")
        elif "from" in s or "to" in s:
            ends = [parse_point(s.get("from")), parse_point(s.get("to"))]
            if None in ends:
                err(f'{where}.slot: "from" and "to" must be "x,y,z" points')
            else:
                for e in ends:
                    if not inside(_flat(e, plane), "slot end"):
                        break
        elif "center" in s:
            c = parse_point(s.get("center"))
            angles = [parse_num(s.get("from_angle")), parse_num(s.get("to_angle"))]
            if c is None or (parse_num(s.get("radius")) or 0) <= 0 or None in angles:
                err(f'{where}.slot: an arc slot needs "center" as "x,y,z", a positive '
                    f'"radius" and "from_angle"/"to_angle" in degrees')
            else:
                inside(_flat(c, plane), "slot centre")
        else:
            err(f"{where}.slot: give {{from, to, width}} or "
                f"{{center, radius, width, from_angle, to_angle}}")
    elif "center" in hole:
        c = parse_point(hole.get("center"))
        r = parse_num(hole.get("radius"))
        if c is None or r is None or r <= 0:
            err(f'{where}: a round hole needs "center" as "x,y,z" and a positive "radius"')
        else:
            inside(_flat(c, plane), "centre")
    elif "wire" in hole:
        err(f"{where}: a spec hole cannot name a wire; give its outline")
    else:
        err(f"{where}: give an outline {{start, segments}}, a circle {{center, radius}} or a {{slot}}")


def _check_dims(dims, where, err):
    if not isinstance(dims, dict):
        err(f"{where} must be an object of named sizes")
        return
    for k, v in dims.items():
        if parse_num(v) is None:
            err(f'{where}.{k} must be a number or "~N"')


def _check_profile_base(base, dim, err):
    plane = base.get("plane")
    if plane not in PLANE_AXES:
        err('base.plane must be "XY", "YZ" or "XZ"')
        return
    ext, rev = base.get("extrude"), base.get("revolve")
    if ext and rev:
        err("base: give extrude or revolve, not both")
    if dim == "2d" and (ext or rev):
        err("base: a 2d target has no extrude or revolve")
    if ext:
        if not isinstance(ext, dict) or not parse_num(ext.get("height")):
            err("base.extrude.height must be a non-zero number")
        elif ext.get("direction") is not None and parse_point(ext["direction"]) is None:
            err('base.extrude.direction must be an "x,y,z" vector')
    if rev:
        if (not isinstance(rev, dict) or parse_point(rev.get("point")) is None
                or parse_point(rev.get("direction")) is None):
            err('base.revolve needs "point" and "direction" as "x,y,z"')
        elif not parse_num(rev.get("angle")):
            err("base.revolve.angle must be a non-zero number of degrees")
    prof = base.get("profile")
    if not isinstance(prof, dict):
        err("base.profile must be an object {start, segments, holes}")
        return
    pts = _check_outline(prof, plane, "base.profile", err)
    if not pts:
        return
    box = _box(pts)
    for h, hole in enumerate(prof.get("holes") or [], 1):
        _check_hole(hole, plane, box, f"base.profile.holes[{h}]", err)


def _check_feature(f, j, n, err):
    w = f"features[{j}]"
    if not isinstance(f, dict) or f.get("type") not in FEATURES:
        err(f"{w}.type must be one of {', '.join(FEATURES)}")
        return
    t = f["type"]
    if t == "hole":
        if parse_point(f.get("center")) is None:
            err(f'{w}.center must be an "x,y,z" point')
        if (parse_num(f.get("diameter")) or 0) <= 0:
            err(f"{w}.diameter must be a positive number")
        depth = f.get("depth")
        if depth != "through" and (parse_num(depth) or 0) <= 0:
            err(f'{w}.depth must be a positive number or "through"')
        if f.get("axis") is not None and parse_point(f["axis"]) is None:
            err(f'{w}.axis must be an "x,y,z" vector')
    elif t in ("fillet", "chamfer"):
        key = "radius" if t == "fillet" else "size"
        if (parse_num(f.get(key)) or 0) <= 0:
            err(f"{w}.{key} must be a positive number")
        if not isinstance(f.get("where"), str) or not f["where"].strip():
            err(f"{w}.where must say which edges")
    else:
        of = f.get("of")
        if isinstance(of, bool) or not isinstance(of, int) or not 1 <= of <= n or of == j:
            err(f"{w}.of must be the 1-based index of another feature (1-{n})")
        if f.get("kind") not in ("circular", "linear"):
            err(f'{w}.kind must be "circular" or "linear"')
        c = f.get("count")
        if isinstance(c, bool) or not isinstance(c, int) or c < 2:
            err(f"{w}.count must be an integer >= 2")


def _check_part(part, dim, ids, err):
    base = part.get("base")
    kind = base.get("kind") if isinstance(base, dict) else None
    if kind == "profile":
        _check_profile_base(base, dim, err)
    elif kind in ("primitive", "approx"):
        if dim == "2d":
            err("base: a 2d target is built from profile parts only")
        if kind == "primitive" and base.get("shape") not in PRIMITIVES:
            err(f"base.shape must be one of {', '.join(PRIMITIVES)}")
        if kind == "approx" and (not isinstance(base.get("description"), str)
                                 or not base["description"].strip()):
            err("base.description must say what the shape is")
        _check_dims(base.get("dims"), "base.dims", err)
    else:
        err('base.kind must be "profile", "primitive" or "approx"')
    pl = part.get("placement")
    if not isinstance(pl, dict):
        err("placement must be an object {origin, normal}")
    else:
        for key in ("origin", "normal"):
            if parse_point(pl.get(key)) is None:
                err(f'placement.{key} must be an "x,y,z" point')
        if pl.get("x_dir") is not None and parse_point(pl["x_dir"]) is None:
            err('placement.x_dir must be an "x,y,z" vector')
        n = parse_point(pl.get("normal"))
        if n is not None and math.hypot(*n) < TOL:
            err("placement.normal must not be zero")
    feats = part.get("features") or []
    if not isinstance(feats, list):
        err("features must be a list")
        feats = []
    for j, f in enumerate(feats, 1):
        _check_feature(f, j, len(feats), err)
    for k, op in enumerate(part.get("ops") or [], 1):
        if not isinstance(op, dict) or op.get("op") not in ("cut", "fuse"):
            err(f'ops[{k}].op must be "cut" or "fuse"')
        elif op.get("with") not in ids or op.get("with") == part.get("id"):
            err(f"ops[{k}].with must name another part ({', '.join(sorted(ids))})")


def validate_spec(spec) -> list[tuple[str | None, str]]:
    """Mechanical checks of a vision spec; [] when it is sound. Each error is
    (part id or None, message)."""
    if not isinstance(spec, dict):
        return [(None, "the spec must be a JSON object")]
    errs: list[tuple[str | None, str]] = []
    dim_block = spec.get("dimensionality")
    dim = dim_block.get("value") if isinstance(dim_block, dict) else None
    if dim not in ("2d", "3d"):
        errs.append((None, 'dimensionality.value must be "2d" or "3d"'))
    parts = spec.get("parts")
    if not isinstance(parts, list) or not parts:
        errs.append((None, "parts must be a non-empty list"))
        return errs
    ids = {p["id"] for p in parts if isinstance(p, dict) and isinstance(p.get("id"), str)}
    seen = set()
    for i, part in enumerate(parts, 1):
        if not isinstance(part, dict):
            errs.append((None, f"parts[{i}] must be an object"))
            continue
        pid = part.get("id")
        if not isinstance(pid, str) or not pid:
            errs.append((None, f"parts[{i}] needs a string id"))
            continue
        if pid in seen:
            errs.append((pid, "duplicate part id"))
            continue
        seen.add(pid)
        _check_part(part, dim, ids, lambda m, pid=pid: errs.append((pid, m)))
    return errs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_vision_spec.py -q`
Expected: all PASS. If `test_correct_specs_pass` fails, print `validate_spec(p1_spec())` and fix the check, not the fixture (the fixture is the clean benchmark part).

- [ ] **Step 5: Commit**

```bash
git add salome_agent/nodes/vision_spec.py tests/unit/test_vision_spec.py
git commit -m "feat: vision spec validator (closure, arc radii, holes inside, references)"
```

---

### Task 2: Spec renderings and conflict handling

**Files:**
- Modify: `salome_agent/nodes/vision_spec.py` (append)
- Test: `tests/unit/test_vision_spec.py` (append)

**Interfaces:**
- Consumes: `validate_spec`, `outline_extent`, `parse_num` (Task 1).
- Produces:
  - `format_errors(errors: list[tuple[str | None, str]]) -> str` — one `- P1: msg` / `- msg` line each.
  - `attach_conflicts(spec, errors) -> dict` — deep copy with part errors in `parts[i]["conflicts"]`, others in top-level `"conflicts"`; a non-dict spec becomes `{"summary": <json of it>, "parts": [], "conflicts": [...]}`.
  - `all_conflicts(spec: dict) -> list[str]` — `"P1: msg"` / `"msg"` strings.
  - `compact_spec(spec: dict) -> str` — planner block, starts with `"Reference image spec"`.
  - `parts_block(spec: dict, ids: list[str]) -> str` — geom block with those parts' exact JSON.
  - `amend_block(spec: dict) -> str` — compact rendering + all parts' JSON.
  - `stage_part_ids(plan: list[dict], idx: int, spec: dict) -> list[str]`
  - `unclaimed_parts(plan: list[dict], spec: dict) -> list[str]`
  - `part_ids(stage: dict) -> list[str]` — a stage's `parts`, a bare string read as a one-item list.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_vision_spec.py`:

```python
import json

from salome_agent.nodes.vision_spec import (all_conflicts, amend_block, attach_conflicts,
                                            compact_spec, format_errors, part_ids, parts_block,
                                            stage_part_ids, unclaimed_parts)


def test_format_errors_lists_each_with_its_part():
    assert format_errors([("P1", "bad arc"), (None, "no parts")]) == "- P1: bad arc\n- no parts"


def test_attach_conflicts_puts_errors_on_their_part():
    spec = box_spec()
    out = attach_conflicts(spec, [("P2", "lid too big"), (None, "datum unclear")])
    assert out["parts"][1]["conflicts"] == ["lid too big"]
    assert out["conflicts"] == ["datum unclear"]
    assert "conflicts" not in spec["parts"][1]
    assert all_conflicts(out) == ["datum unclear", "P2: lid too big"]


def test_attach_conflicts_wraps_a_non_object_answer():
    out = attach_conflicts(["a box"], [(None, "the spec must be a JSON object")])
    assert out == {"summary": '["a box"]', "parts": [],
                   "conflicts": ["the spec must be a JSON object"]}
    assert compact_spec(out).startswith("Reference image spec")


def test_compact_spec_summarises_each_part_without_its_numbers():
    text = compact_spec(p1_spec())
    assert text.startswith("Reference image spec")
    assert "Target: 2D in the XY plane" in text
    assert "- P1 plate: profile in XY, flat face, 192 x 230, 7 holes" in text
    assert "55.426" not in text


def test_compact_spec_flags_an_assumed_2d_target():
    spec = p1_spec()
    spec["dimensionality"] = {"value": "2d", "assumed": True, "reason": "single outline"}
    assert "Target: 2D in the XY plane (assumed: single outline)" in compact_spec(spec)


def test_compact_spec_names_primitive_approx_features_and_conflicts():
    spec = attach_conflicts(box_spec(), [("P1", "hole off the face")])
    text = compact_spec(spec)
    assert '- P1 block: box {"dx": "~40", "dy": "~30", "dz": 10}; features: 1 hole, 1 pattern; 1 conflict' in text
    assert '- P2 lid: approx. rounded cap {"d": "~30"}' in text


def test_parts_block_carries_exact_json_of_the_listed_parts_only():
    text = parts_block(box_spec(), ["P2"])
    assert '"id": "P2"' in text and '"id": "P1"' not in text
    assert "Target: 3D" in text
    data = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert data == [box_spec()["parts"][1]]


def test_parts_block_keeps_estimated_coordinates_verbatim():
    assert '"origin": "~0,0,0"' in parts_block(box_spec(), ["P1"])


def test_amend_block_has_summary_and_all_parts():
    text = amend_block(box_spec())
    assert "a block with a hole" in text
    assert '"id": "P1"' in text and '"id": "P2"' in text


def test_part_ids_reads_a_bare_string():
    assert part_ids({"parts": "P1"}) == ["P1"]
    assert part_ids({"parts": ["P1", "P2"]}) == ["P1", "P2"]
    assert part_ids({}) == []


def test_stage_part_ids_uses_the_stage_list():
    plan = [{"module": "geom", "parts": ["P1"]}, {"module": "geom", "parts": ["P2"]},
            {"module": "geom"}]
    assert stage_part_ids(plan, 1, box_spec()) == ["P2"]
    assert stage_part_ids(plan, 2, box_spec()) == []


def test_stage_part_ids_falls_back_to_every_part_when_no_stage_lists_any():
    plan = [{"module": "geom"}, {"module": "geom"}]
    assert stage_part_ids(plan, 1, box_spec()) == ["P1", "P2"]


def test_unclaimed_parts():
    assert unclaimed_parts([{"module": "geom", "parts": ["P1"]}], box_spec()) == ["P2"]
    assert unclaimed_parts([{"module": "geom"}], box_spec()) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_vision_spec.py -q`
Expected: FAIL with `ImportError: cannot import name 'all_conflicts'`

- [ ] **Step 3: Write the implementation**

Append to `salome_agent/nodes/vision_spec.py` (add `import copy` and `import json` to the imports at the top):

```python
def format_errors(errors) -> str:
    return "\n".join(f"- {pid}: {msg}" if pid else f"- {msg}" for pid, msg in errors)


def attach_conflicts(spec, errors) -> dict:
    """The spec with each remaining validator error stored where it applies."""
    if not isinstance(spec, dict):
        return {"summary": json.dumps(spec, ensure_ascii=False), "parts": [],
                "conflicts": [m for _, m in errors]}
    out = copy.deepcopy(spec)
    by_id = {p.get("id"): p for p in out.get("parts") or [] if isinstance(p, dict)}
    for pid, msg in errors:
        target = by_id.get(pid, out)
        target.setdefault("conflicts", []).append(msg)
    return out


def _parts(spec) -> list[dict]:
    return [p for p in spec.get("parts") or [] if isinstance(p, dict)]


def all_conflicts(spec: dict) -> list[str]:
    out = list(spec.get("conflicts") or [])
    for p in _parts(spec):
        out += [f"{p.get('id')}: {c}" for c in p.get("conflicts") or []]
    return out


def _target_line(spec) -> str | None:
    dim = spec.get("dimensionality")
    if not isinstance(dim, dict) or dim.get("value") not in ("2d", "3d"):
        return None
    line = f"Target: {dim['value'].upper()}"
    if dim["value"] == "2d":
        planes = {(p.get("base") or {}).get("plane") for p in _parts(spec)} - {None}
        line += f" in the {'/'.join(sorted(planes)) or 'XY'} plane"
    if dim.get("assumed"):
        line += f" (assumed: {dim.get('reason') or 'the image gave no depth'})"
    return line


def _json(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def _part_line(p) -> str:
    base = p.get("base") if isinstance(p.get("base"), dict) else {}
    kind = base.get("kind")
    head = f"{p.get('id')} {p.get('name') or ''}".rstrip()
    if kind == "profile":
        desc = f"profile in {base.get('plane')}"
        if base.get("extrude"):
            desc += f", extruded {(base['extrude'] or {}).get('height')}"
        elif base.get("revolve"):
            desc += f", revolved {(base['revolve'] or {}).get('angle')} deg"
        else:
            desc += ", flat face"
        prof = base.get("profile") if isinstance(base.get("profile"), dict) else {}
        pts = outline_extent(prof, base.get("plane"))
        if pts:
            x0, y0, x1, y1 = _box(pts)
            desc += f", {x1 - x0:g} x {y1 - y0:g}"
        holes = len(prof.get("holes") or [])
        if holes:
            desc += f", {holes} hole{'s' if holes > 1 else ''}"
    elif kind == "primitive":
        desc = f"{base.get('shape')} {_json(base.get('dims') or {})}"
    elif kind == "approx":
        desc = f"approx. {base.get('description')} {_json(base.get('dims') or {})}"
    else:
        desc = "unknown base"
    feats = [f.get("type") for f in p.get("features") or [] if isinstance(f, dict)]
    if feats:
        counts = {}
        for t in feats:
            counts[t] = counts.get(t, 0) + 1
        desc += "; features: " + ", ".join(f"{n} {t}" for t, n in counts.items())
    if p.get("relation"):
        desc += f"; {p['relation']}"
    if p.get("conflicts"):
        n = len(p["conflicts"])
        desc += f"; {n} conflict{'s' if n > 1 else ''}"
    return f"- {head}: {desc}"


def _context_lines(spec) -> list[str]:
    lines = []
    target = _target_line(spec)
    if target:
        lines.append(target)
    if spec.get("summary"):
        lines.append(f"Summary: {spec['summary']}")
    if spec.get("datum"):
        lines.append(f"Datum: {spec['datum']}")
    if spec.get("uncertainties"):
        lines.append("Uncertainties: " + "; ".join(str(u) for u in spec["uncertainties"]))
    if spec.get("conflicts"):
        lines.append("Conflicts: " + "; ".join(str(c) for c in spec["conflicts"]))
    return lines


def compact_spec(spec: dict) -> str:
    """The planner's view: parts by id and kind, no numbers to restate."""
    lines = ["Reference image spec (read from the user's image(s); each geom stage is "
             "handed the exact data of the parts it lists):"]
    lines += _context_lines(spec)
    parts = _parts(spec)
    if parts:
        lines.append("Parts:")
        lines += [_part_line(p) for p in parts]
    return "\n".join(lines)


def parts_block(spec: dict, ids: list[str]) -> str:
    """A geom stage's view: its parts' exact JSON."""
    chosen = [p for p in _parts(spec) if p.get("id") in ids]
    lines = ["Reference image spec for this stage's part(s), read from the user's "
             "image(s) by an image-analysis agent (you do not see the images):"]
    lines += _context_lines(spec)
    lines.append("```json\n" + json.dumps(chosen, ensure_ascii=False, indent=1) + "\n```")
    return "\n".join(lines)


def amend_block(spec: dict) -> str:
    return (compact_spec(spec) + "\nParts data:\n```json\n"
            + json.dumps(_parts(spec), ensure_ascii=False, indent=1) + "\n```")


def part_ids(stage: dict) -> list[str]:
    ids = stage.get("parts") or []
    return [ids] if isinstance(ids, str) else [i for i in ids if isinstance(i, str)]


def stage_part_ids(plan: list[dict], idx: int, spec: dict) -> list[str]:
    """The parts stage idx builds; every part when no stage of the plan lists any
    (a planner that ignored `parts` must not leave geom without the spec)."""
    if any(part_ids(s) for s in plan):
        return part_ids(plan[idx])
    return [p.get("id") for p in _parts(spec)]


def unclaimed_parts(plan: list[dict], spec: dict) -> list[str]:
    claimed = {i for s in plan for i in part_ids(s)}
    if not claimed:
        return []
    return [p.get("id") for p in _parts(spec) if p.get("id") not in claimed]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_vision_spec.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add salome_agent/nodes/vision_spec.py tests/unit/test_vision_spec.py
git commit -m "feat: vision spec renderings for planner, geom stages, amend; conflicts on their part"
```

---

### Task 3: Vision prompt and fix loop, `llm.vision_fix_retries`

**Files:**
- Modify: `salome_agent/nodes/vision.py` (whole file)
- Modify: `salome_agent/config.py:11-14`
- Modify: `README.md` (config example block near line 79 and the `llm` table near line 116)
- Test: `tests/unit/test_vision.py` (rewrite), `tests/unit/test_config.py` (append)

**Interfaces:**
- Consumes: `validate_spec`, `format_errors`, `attach_conflicts` (Tasks 1-2).
- Produces: `describe_images(request: str, images: list[str], llm, fix_retries: int = 2) -> dict` — always a dict (the spec, with `conflicts` when it could not be fixed). `format_description` is removed.

- [ ] **Step 1: Write the failing tests**

Replace `tests/unit/test_vision.py` with:

```python
from unittest.mock import MagicMock

from salome_agent.nodes.vision import VISION_SYSTEM_PROMPT, describe_images
from tests.unit.test_vision_spec import box_spec


def test_describe_images_sends_images_to_the_vision_module():
    llm = MagicMock()
    llm.chat.return_value = box_spec()
    out = describe_images("make this", ["data:image/png;base64,AAAA"], llm)
    assert out == box_spec()
    assert llm.chat.call_count == 1
    kwargs = llm.chat.call_args.kwargs
    assert kwargs["module"] == "vision"
    content = kwargs["messages"][1]["content"]
    assert content[0]["type"] == "text" and "make this" in content[0]["text"]
    assert content[1] == {"type": "image_url",
                          "image_url": {"url": "data:image/png;base64,AAAA"}}


def test_a_broken_spec_is_sent_back_with_its_errors():
    bad = box_spec()
    bad["parts"][1]["ops"] = [{"op": "fuse", "with": "P9"}]
    llm = MagicMock()
    llm.chat.side_effect = [bad, box_spec()]
    out = describe_images("make this", ["data:image/png;base64,AAAA"], llm)
    assert out == box_spec()
    messages = llm.chat.call_args_list[1].kwargs["messages"]
    assert messages[2]["role"] == "assistant" and '"P9"' in messages[2]["content"]
    assert messages[3]["role"] == "user"
    assert "- P2: ops[1].with must name another part (P1, P2)" in messages[3]["content"]
    assert "whole corrected JSON" in messages[3]["content"]


def test_errors_left_after_the_retries_become_conflicts():
    bad = box_spec()
    bad["parts"][1]["ops"] = [{"op": "fuse", "with": "P9"}]
    llm = MagicMock()
    llm.chat.return_value = bad
    out = describe_images("make this", ["x"], llm, fix_retries=1)
    assert llm.chat.call_count == 2
    assert out["parts"][1]["conflicts"] == ["ops[1].with must name another part (P1, P2)"]


def test_zero_retries_never_re_asks():
    llm = MagicMock()
    llm.chat.return_value = {"summary": "a box"}
    out = describe_images("make this", ["x"], llm, fix_retries=0)
    assert llm.chat.call_count == 1
    assert "parts must be a non-empty list" in out["conflicts"]


def test_prompt_asks_for_dimensions_read_before_parts_and_profiles_as_drawn():
    p = VISION_SYSTEM_PROMPT
    assert p.index('"dimensions_read"') < p.index('"parts"')
    assert '"kind": "profile"' in p and '"kind": "approx"' in p
    assert "the tool finds the tangent points" in p
    assert '"assumed": true' in p
```

Append to `tests/unit/test_config.py`:

```python
def test_vision_fix_retries_is_a_known_llm_key():
    from salome_agent.config import LLM_KNOWN_KEYS
    assert "vision_fix_retries" in LLM_KNOWN_KEYS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_vision.py tests/unit/test_config.py -q`
Expected: FAIL (`describe_images` has no `fix_retries`; prompt lacks `"kind": "profile"`; key missing)

- [ ] **Step 3: Write the implementation**

Replace `salome_agent/nodes/vision.py` with:

```python
import json

from salome_agent.llm import with_images
from salome_agent.logs import log
from salome_agent.nodes.vision_spec import attach_conflicts, format_errors, validate_spec

VISION_SYSTEM_PROMPT = """You are the image-analysis agent of a CAD assistant. You
receive one or more reference images (photos, sketches, technical drawings, plans,
renders) plus the user's text request. Your job is to turn what is visible into a
structured spec that a planner and the SALOME GEOM tools rebuild. You do NOT plan
or write CAD code.

First decide whether the TARGET is 2D or 3D -- every image is flat, so this is
about the object to build, not the picture:
- "2d": a planar shape with no thickness to build -- an outline, a profile, a
  cross-section, a floor plan, a flat sketch of contours. It is rebuilt as a
  planar face.
- "3d": a solid or an assembly with depth -- a photo, a render, an isometric or
  perspective view, shading, several orthographic views (front/top/side) of the
  same part, or any thickness/depth/height dimension. A technical drawing of a
  3D part is "3d" even though the drawing itself is flat.
The text request wins over the image: "extrude", "plate 5 mm thick", "solid" make
it 3d; "profile", "outline", "face", "2D", "sketch only" make it 2d. When the
image(s) alone cannot tell (a single outline with no depth cue and a request
that does not say), choose "2d", set "assumed": true and list the missing depth
under uncertainties.

Several images are usually different views of the SAME object (front, top, side,
isometric...): read them together and reconcile them into one spec instead of
describing each picture as a separate object.

Respond as JSON with exactly these keys, in this order, and output only the JSON
object:
{
  "dimensionality": {"value": "2d" or "3d", "assumed": true only when nothing
                     told you, else false, "reason": the cue that decided it},
  "summary": one or two sentences: what the object is overall,
  "views": [{"image": 1-based image number, "view": "front|top|side|isometric|
             section|plan|detail|photo|unknown", "shows": what it contributes}],
  "dimensions_read": [{"text": a dimension, annotation or scale exactly as
                       written, with its unit, "means": what it measures -- what
                       each extension line sits on, or the feature it labels}],
  "datum": the origin and axes every coordinate below uses, e.g. "O = centre of
           the big bore; +X right, +Y up, +Z towards the viewer",
  "parts": [one entry per part, see below],
  "uncertainties": ["anything ambiguous, hidden or not inferable from the image(s)"]
}

Each part:
{
  "id": "P1", "P2", ... (unique),
  "name": short label,
  "base": exactly one of
    {"kind": "profile", "plane": "XY" | "YZ" | "XZ",
     "profile": {"start": "x,y,z", "segments": [...], "holes": [...]},
     "extrude": {"height": h, "direction": "x,y,z" (optional)} or null,
     "revolve": {"point": "x,y,z", "direction": "x,y,z", "angle": degrees} or null}
      -- an outline the image gives exactly (a dimensioned drawing or sketch); at
      most one of extrude / revolve, neither for a "2d" target;
    {"kind": "primitive", "shape": "box" | "cylinder" | "sphere" | "cone" | "torus",
     "dims": {"<name>": number, ...}};
    {"kind": "approx", "description": what the shape is,
     "dims": {"<name>": number, ...}}
      -- a shape you can only estimate (photos, renders): its overall sizes,
  "placement": {"origin": "x,y,z", "normal": "x,y,z", "x_dir": "x,y,z" (optional)}
      -- where the part sits in the datum frame and which way its base faces,
  "features": [features cut into or added to this part, in order:
    {"type": "hole", "center": "x,y,z", "diameter": d, "depth": d or "through",
     "axis": "x,y,z" (optional)},
    {"type": "fillet", "radius": r, "where": which edges, in words},
    {"type": "chamfer", "size": s, "where": which edges, in words},
    {"type": "pattern", "of": 1-based index of the feature it repeats,
     "kind": "circular" | "linear", "count": n, plus its axis or step as drawn}],
  "relation": how it sits against the other parts, in words, or null,
  "ops": [{"op": "cut" | "fuse", "with": another part's id}] (optional)
}

A profile is "start" then "segments", each {"to": "x,y,z"} -- a straight line --
plus at most one of "center": "x,y,z" (a circular arc about that centre, the
shorter way round; add "reverse": true for the longer), "through": "x,y,z" (a
point on the arc; use it for a half circle) or "via": ["x,y,z", ...] (a smooth
curve), and optionally "fillet": r, which rounds the corner at that segment's
"to": give the sharp corner as drawn, the tool finds the tangent points. If the
last "to" is not "start", a straight line closes the outline. "holes" are
outlines of the same form, {"center": "x,y,z", "radius": r} circles,
{"slot": {"from": "x,y,z", "to": "x,y,z", "width": w}} straight slots (the centre
line's ends) or {"slot": {"center": "x,y,z", "radius": r, "width": w,
"from_angle": a, "to_angle": b}} slots along an arc (degrees from +X). All points
are "x,y,z" in the datum frame and lie in the profile's plane.

Rules:
- Write "dimensions_read" first. Every number in "parts" is one of those values,
  derived from them, or an estimate written as a string with "~", e.g. "~40"
  (also inside points: "~40,0,0").
- Millimetres throughout; convert a drawing in another unit and say so in
  dimensions_read.
- For "2d", never invent a thickness or height; for "3d", give the depth of every
  part, estimated with "~" if no view shows it.
- Never invent detail that is not visible; list it under uncertainties instead.
- Reading technical drawings:
  - A linear dimension measures between its two extension lines: say what each one
    sits on (a centre line, an edge, a tangent point). Ending on a centre line means
    the distance is to that centre, not to an edge.
  - A radius or diameter arrowed to a chain-dotted arc or circle is a pitch/centre-
    line radius (bolt circle, slot centre line), not an edge.
  - A curved slot with R ends is as wide as twice that R, centred on its pitch arc.
  - "nX" counts identical features; "TYP" applies the value to every like feature.
  - Give the centre of every arc and circle as coordinates from the datum.
- Before answering, check that the outline closes: consecutive arcs and lines must
  meet or be tangent at the centres and radii you give. If they do not, re-read the
  dimensions; if a conflict remains, name the conflicting values in uncertainties.
- Keep it factual and compact: no CAD tool names, no step-by-step build order."""


def describe_images(request: str, images: list[str], llm, fix_retries: int = 2) -> dict:
    """Run the vision agent: one multimodal call that turns the reference image(s)
    into a structured spec, re-asked with the validator's errors up to
    fix_retries times; errors still left are attached as conflicts."""
    text = ("User request (context only -- describe the image(s), do not plan):\n"
            + request)
    messages = [{"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": with_images(text, images)}]
    spec = llm.chat(messages=messages, module="vision")
    for attempt in range(fix_retries + 1):
        errors = validate_spec(spec)
        if not errors:
            return spec
        if attempt == fix_retries:
            log.warning("vision spec still has %d problem(s); passing them on as conflicts",
                        len(errors))
            return attach_conflicts(spec, errors)
        log.info("vision spec has %d problem(s), asking for a fix (%d/%d):\n%s",
                 len(errors), attempt + 1, fix_retries, format_errors(errors))
        messages = messages + [
            {"role": "assistant", "content": json.dumps(spec, ensure_ascii=False)},
            {"role": "user", "content":
                "The spec has these problems:\n" + format_errors(errors)
                + "\nRe-read the image(s) where needed, fix them, and answer with the "
                  "whole corrected JSON spec only."}]
        spec = llm.chat(messages=messages, module="vision")
    return spec
```

In `salome_agent/config.py`, change `LLM_KNOWN_KEYS` to:

```python
LLM_KNOWN_KEYS = {
    "base_url", "api_key", "ssl_cert_file", "model", "temperature", "timeout",
    "json_retries", "max_retries", "extra_headers", "extra_body", "module_extra_body",
    "vision_fix_retries",
}
```

In `README.md`, in the full config example add `"vision_fix_retries": 2,` on the line after `"json_retries": 2,`, and add this row to the `llm` table right after the `json_retries` row:

```markdown
| `vision_fix_retries` | `2` | When the image-analysis agent's spec fails the mechanical checks (an outline that does not close, an arc end off its radius, a hole outside the outline, a dangling reference), the errors are fed back and it is re-asked this many times; errors still left go to the planner and geom as `conflicts`. |
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_vision.py tests/unit/test_config.py tests/unit/test_vision_spec.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add salome_agent/nodes/vision.py salome_agent/config.py README.md tests/unit/test_vision.py tests/unit/test_config.py
git commit -m "feat: vision agent returns a per-part feature spec, re-asked with validator errors (llm.vision_fix_retries)"
```

Note: `salome_agent/graph.py` still imports `format_description` after this commit; `tests/unit/test_graph.py` describe-node tests fail until Task 5. Do not run the full suite as a gate here.

---

### Task 4: Planner takes the spec and lists parts per stage

**Files:**
- Modify: `salome_agent/nodes/planner.py`
- Modify: `salome_agent/state.py:6-15` (StagePlan)
- Modify: `tests/fixtures/golden/planner_prompt.txt` (regenerated)
- Test: `tests/unit/test_planner.py`

**Interfaces:**
- Consumes: `compact_spec`, `part_ids` (Task 2).
- Produces: `plan(request: str, llm, detail: str = "balanced", image_spec: dict | None = None) -> list[dict]`; stages may carry `parts: list[str]` (only known ids). `StagePlan.parts: NotRequired[list[str]]`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_planner.py`, replace `test_plan_appends_image_description_to_the_request` (line 83 onward, the whole function) with:

```python
def test_plan_appends_the_compact_spec_to_the_request():
    from tests.unit.test_vision_spec import p1_spec
    llm = MagicMock()
    llm.chat.return_value = {"stages": [{"module": "geom", "intent": "plate", "params": {},
                                         "inputs": [], "parts": ["P1"]}]}
    stages = plan("make this plate", llm, image_spec=p1_spec())
    user = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert user.startswith("make this plate\n\nReference image spec")
    assert "- P1 plate: profile in XY" in user
    assert stages[0]["parts"] == ["P1"]


def test_unknown_part_ids_are_sent_back_once_then_dropped():
    from tests.unit.test_vision_spec import p1_spec
    llm = MagicMock()
    bad = {"stages": [{"module": "geom", "intent": "plate", "params": {}, "inputs": [],
                       "parts": ["P1", "P7"]}]}
    llm.chat.side_effect = [bad, bad]
    stages = plan("make this plate", llm, image_spec=p1_spec())
    assert llm.chat.call_count == 2
    retry = llm.chat.call_args_list[1].kwargs["messages"]
    assert retry[-1]["role"] == "user"
    assert 'Unknown part id(s) P7 in "parts"; the parts are P1.' in retry[-1]["content"]
    assert stages[0]["parts"] == ["P1"]


def test_a_bare_string_parts_value_is_read_as_a_list():
    from tests.unit.test_vision_spec import p1_spec
    llm = MagicMock()
    llm.chat.return_value = {"stages": [{"module": "geom", "intent": "plate", "params": {},
                                         "inputs": [], "parts": "P1"}]}
    stages = plan("make this plate", llm, image_spec=p1_spec())
    assert llm.chat.call_count == 1
    assert stages[0]["parts"] == ["P1"]


def test_prompt_tells_the_planner_to_list_part_ids_not_numbers():
    assert '"parts"' in PLANNER_SYSTEM_PROMPT
    assert "Reference image spec" in PLANNER_SYSTEM_PROMPT
    assert "never restate" in PLANNER_SYSTEM_PROMPT.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_planner.py -q`
Expected: FAIL (`plan()` got an unexpected keyword argument `image_spec`; prompt lacks the text)

- [ ] **Step 3: Write the implementation**

In `salome_agent/nodes/planner.py`:

Add imports at the top:

```python
import json

from salome_agent.logs import log
from salome_agent.modules import spec_for
from salome_agent.nodes.vision_spec import compact_spec, part_ids
```

Replace the last paragraph of `_PLANNER_BASE` (from `If the request comes with an "Image description" block,` to the closing `"""`) with:

```python
If the request comes with a "Reference image spec" block, an image-analysis
agent read it from the user's reference image(s): one entry per part, each with
an id (P1, P2, ...). Shape your stages around those parts and give every stage
that builds one of them a "parts" list of their ids, e.g. "parts": ["P1"]; that
stage is handed the part's exact data. Never restate a part's numbers in intent
or params -- refer to it by id. Several views described together are ONE object,
not separate parts. Explicit numeric dimensions in the text request still win:
put those in params. Mention listed uncertainties or conflicts in a stage's
params rather than inventing detail."""
```

Replace `plan()` with:

```python
def plan(request: str, llm, detail: str = "balanced",
         image_spec: dict | None = None) -> list[dict]:
    system = PLANNER_SYSTEM_PROMPT + _DETAIL_HINT.get(detail, "")
    if image_spec:
        request = f"{request}\n\n{compact_spec(image_spec)}"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": request}]
    stages = llm.chat(messages=messages, module="planner")["stages"]
    if image_spec:
        stages = _check_parts(stages, image_spec, llm, messages)
    return stages


def _check_parts(stages: list[dict], image_spec: dict, llm, messages: list[dict]) -> list[dict]:
    """Stages may only list spec part ids: an unknown one is sent back once, then
    dropped."""
    known = [p.get("id") for p in image_spec.get("parts") or [] if isinstance(p, dict)]
    unknown = sorted({i for s in stages for i in part_ids(s)} - set(known))
    if unknown:
        log.warning("planner listed unknown part id(s) %s; asking again", ", ".join(unknown))
        messages = messages + [
            {"role": "assistant", "content": json.dumps({"stages": stages}, ensure_ascii=False)},
            {"role": "user", "content":
                f'Unknown part id(s) {", ".join(unknown)} in "parts"; the parts are '
                f'{", ".join(known)}. Answer with the whole corrected JSON plan.'}]
        stages = llm.chat(messages=messages, module="planner")["stages"]
    for s in stages:
        if "parts" in s:
            s["parts"] = [i for i in part_ids(s) if i in known]
    return stages
```

In `salome_agent/state.py`, add to `StagePlan` after the `tools` field:

```python
    # Optional: the vision spec part ids this stage builds (task 286); the stage
    # is handed those parts' exact data.
    parts: NotRequired[list[str]]
```

Regenerate the golden planner prompt:

```bash
.venv/bin/python -c "from salome_agent.nodes.planner import PLANNER_SYSTEM_PROMPT as p; open('tests/fixtures/golden/planner_prompt.txt','w').write(p)"
git diff --stat tests/fixtures/golden/planner_prompt.txt
```

Expected: only the image paragraph changed (check `git diff tests/fixtures/golden/planner_prompt.txt` shows just that paragraph).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_planner.py tests/unit/test_golden_surfaces.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add salome_agent/nodes/planner.py salome_agent/state.py tests/unit/test_planner.py tests/fixtures/golden/planner_prompt.txt
git commit -m "feat: planner gets the compact image spec and lists part ids per stage, never their numbers"
```

---

### Task 5: Wire the spec through graph, geom stages and amend

**Files:**
- Modify: `salome_agent/state.py` (`RunState.image_description` -> `image_spec`)
- Modify: `salome_agent/runner.py:119` (initial state key)
- Modify: `salome_agent/graph.py` (`_amend_node` ~line 105, `_describe_image_node` ~207-235, `_plan_node` ~255, `_run_stage_node` ~318-322)
- Modify: `salome_agent/specialists/geom.py` (prompt paragraph ~78-84, `run_geom_specialist` ~182-202, remove `image_block` ~205-211)
- Test: `tests/unit/test_graph.py`, `tests/unit/test_graph_amend.py`, `tests/unit/test_specialist_geom.py`

**Interfaces:**
- Consumes: `describe_images(..., fix_retries)` (Task 3); `plan(..., image_spec=)` (Task 4); `parts_block`, `stage_part_ids`, `amend_block` (Task 2).
- Produces: state key `image_spec: Optional[dict]`; `run_geom_specialist(..., reference_block: str | None = None)`.

- [ ] **Step 1: Update and write the failing tests**

In `tests/unit/test_graph.py`, replace the three describe-node tests and the plan-node test shown at lines ~389-441 (`test_describe_image_node_stores_the_description_for_the_planner`, `test_describe_image_node_hands_a_follow_up_turn_to_amend`, `test_plan_node_hands_the_description_not_the_images_to_the_planner`) with:

```python
def test_describe_image_node_stores_the_spec_for_the_planner():
    from salome_agent import graph
    from tests.unit.test_vision_spec import box_spec

    llm = MagicMock()
    llm.chat.return_value = box_spec()
    state = {"user_request": "x", "reference_images": ["data:image/png;base64,AAAA"]}
    cmd = graph._describe_image_node(state, {"configurable": {"llm_client": llm}})
    assert cmd.goto == "plan"
    assert cmd.update["image_spec"] == box_spec()
    assert cmd.update["reference_images"] is None


def test_describe_image_node_reads_vision_fix_retries_from_config():
    from salome_agent import graph

    llm = MagicMock()
    llm.chat.return_value = {"summary": "a box"}
    state = {"user_request": "x", "reference_images": ["x"],
             "config": {"llm": {"vision_fix_retries": 0}}}
    cmd = graph._describe_image_node(state, {"configurable": {"llm_client": llm}})
    assert llm.chat.call_count == 1
    assert "parts must be a non-empty list" in cmd.update["image_spec"]["conflicts"]


def test_describe_image_node_hands_a_follow_up_turn_to_amend():
    from salome_agent import graph
    from tests.unit.test_vision_spec import box_spec

    llm = MagicMock()
    llm.chat.return_value = box_spec()
    state = {"user_request": "x", "turn_idx": 1,
             "reference_images": ["data:image/png;base64,AAAA"]}
    cmd = graph._describe_image_node(state, {"configurable": {"llm_client": llm}})
    assert cmd.goto == "amend" and cmd.update["image_spec"]["summary"] == "a block with a hole"
    cmd = graph._describe_image_node({"user_request": "x", "turn_idx": 1},
                                     {"configurable": {"llm_client": llm}})
    assert cmd.goto == "amend"


def test_plan_node_hands_the_spec_not_the_images_to_the_planner(monkeypatch):
    from salome_agent import graph
    from salome_agent.nodes import planner

    seen = {}

    def fake_plan(request, llm, detail="balanced", image_spec=None):
        seen.update(image_spec=image_spec)
        return [{"module": "geom", "intent": "box", "params": {}, "inputs": []}]

    monkeypatch.setattr(planner, "plan", fake_plan)
    state = {"user_request": "x", "config": {"policy": {}},
             "reference_images": ["data:image/png;base64,AAAA"],
             "image_spec": {"summary": "a box", "parts": []}}
    graph._plan_node(state, {"configurable": {"llm_client": MagicMock()}})
    assert seen == {"image_spec": {"summary": "a box", "parts": []}}
```

Append to `tests/unit/test_graph.py` (stage-injection tests; they patch `ToolSpecialist.run` to capture blocks):

```python
def _capture_stage_blocks(monkeypatch, plan, idx, image_spec):
    from salome_agent import graph
    from salome_agent.specialists import tool_specialist

    seen = {}

    def fake_run(self, stage_plan, llm, client, cap, blocks=(), **kw):
        seen["blocks"] = blocks
        return {"status": "ok", "call_log": [], "calls": [], "result_names": []}

    monkeypatch.setattr(tool_specialist.ToolSpecialist, "run", fake_run)
    monkeypatch.setattr(graph, "retry_context", lambda state, idx, client: None)
    state = {"plan": plan, "stage_idx": idx, "stages": {i: {"module": "geom", "status": "pending"}
                                                         for i in range(len(plan))},
             "config": {"policy": {"tool_iteration_cap": 5}}, "image_spec": image_spec,
             "build_graph": None}
    try:
        graph._run_stage_node(state, {"configurable": {"llm_client": MagicMock(),
                                                       "executor_client": None}})
    except Exception:  # noqa: BLE001 — only the blocks handed to the specialist matter
        pass
    return "\n".join(seen.get("blocks") or ())


def test_geom_stage_gets_only_its_parts(monkeypatch):
    from tests.unit.test_vision_spec import box_spec
    plan = [{"module": "geom", "intent": "a", "params": {}, "inputs": [], "parts": ["P1"]},
            {"module": "geom", "intent": "b", "params": {}, "inputs": [], "parts": ["P2"]}]
    text = _capture_stage_blocks(monkeypatch, plan, 1, box_spec())
    assert '"id": "P2"' in text and '"id": "P1"' not in text


def test_geom_stage_gets_every_part_when_the_planner_listed_none(monkeypatch):
    from tests.unit.test_vision_spec import box_spec
    plan = [{"module": "geom", "intent": "a", "params": {}, "inputs": []}]
    text = _capture_stage_blocks(monkeypatch, plan, 0, box_spec())
    assert '"id": "P1"' in text and '"id": "P2"' in text


def test_geom_stage_without_parts_gets_no_spec_when_others_list_theirs(monkeypatch):
    from tests.unit.test_vision_spec import box_spec
    plan = [{"module": "geom", "intent": "a", "params": {}, "inputs": [], "parts": ["P1", "P2"]},
            {"module": "geom", "intent": "colour it", "params": {}, "inputs": []}]
    text = _capture_stage_blocks(monkeypatch, plan, 1, box_spec())
    assert "Reference image spec" not in text
```

In `tests/unit/test_graph_amend.py`, in `test_follow_up_with_images_is_described_then_amended`, change the monkeypatch and the assert to:

```python
    monkeypatch.setattr(vision_mod, "describe_images",
                        lambda request, images, llm, fix_retries=2: {"summary": "a D-shaped handle",
                                                                      "parts": []})
```

```python
    assert "a D-shaped handle" in seen["image_description"]
    assert seen["image_description"].startswith("Reference image spec")
```

In `tests/unit/test_specialist_geom.py`, replace `test_image_description_is_injected_as_text_never_as_images` with:

```python
def test_reference_block_is_injected_as_text_never_as_images():
    llm = MagicMock()
    llm.chat.side_effect = [_resp(("finish_stage", {"result_names": []}))]
    client = MagicMock()
    client.list_tools.return_value = {"tools": []}
    run_geom_specialist({"module": "geom", "intent": "box", "params": {}, "inputs": []},
                        llm, client, iteration_cap=3,
                        reference_block="Reference image spec for this stage's part(s): P1")
    user = llm.chat.call_args.kwargs["messages"][1]["content"]
    assert isinstance(user, str)
    assert "Reference image spec for this stage's part(s): P1" in user


def test_geom_prompt_explains_the_reference_image_spec():
    from salome_agent.specialists.geom import _GEOM_BASE
    assert "Reference image spec" in _GEOM_BASE
    assert "make_profile_face unchanged" in _GEOM_BASE
    assert "conflicts" in _GEOM_BASE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_graph.py tests/unit/test_graph_amend.py tests/unit/test_specialist_geom.py -q`
Expected: FAIL (`ImportError: cannot import name 'format_description'`, missing `image_spec`, `reference_block`)

- [ ] **Step 3: Write the implementation**

`salome_agent/state.py` — in `RunState`, replace `image_description: Optional[str]` with:

```python
    image_spec: Optional[dict]
```

`salome_agent/runner.py` — in `initial_state`, replace `"image_description": None,` with `"image_spec": None,`.

`salome_agent/graph.py`:

In `_amend_node`, replace

```python
    description = state.get("image_description")
    extra = {"image_description": description} if description else {}
```

with

```python
    extra = {}
    if state.get("image_spec"):
        from salome_agent.nodes.vision_spec import amend_block
        extra = {"image_description": amend_block(state["image_spec"])}
```

Replace `_describe_image_node` with:

```python
def _describe_image_node(state: dict, config: "RunnableConfig"):
    from langgraph.types import Command
    from salome_agent.nodes.vision import describe_images

    images = state.get("reference_images")
    after = "amend" if (state.get("turn_idx") or 0) > 0 else "plan"
    if not images:
        return Command(goto=after)
    llm = config["configurable"]["llm_client"]
    retries = ((state.get("config") or {}).get("llm") or {}).get("vision_fix_retries", 2)
    log.info("describing %d reference image(s)...", len(images))
    snap0 = _usage_snap(llm)
    t0 = time.monotonic()
    try:
        spec = describe_images(state["user_request"], images, llm, fix_retries=retries)
    except Exception as e:  # noqa: BLE001
        msg = _llm_endpoint_fatal(e)
        if msg is None:
            raise
        log.error("%s", msg)
        return Command(update={"fatal": msg}, goto="assemble")
    log.info("image spec:\n%s", json.dumps(spec, ensure_ascii=False, indent=1))
    # only this node reads the images; the checkpointer would otherwise carry
    # the base64 through every later step
    update = {"image_spec": spec, "reference_images": None}
    if snap0 is not None:
        vt = _timing_delta(snap0, _usage_snap(llm), time.monotonic() - t0)
        log.info("  vision timing: wall=%.1fs llm=%.1fs", vt["wall"], vt["llm_s"])
    return Command(update=update, goto=after)
```

(Add `import json` at the top of `graph.py` if it is not already imported: check with `grep -n "^import json" salome_agent/graph.py`.)

In `_plan_node`, replace

```python
        description = state.get("image_description")
        extra = {"image_description": description} if description else {}
        stage_plans = plan(request, llm, detail, **extra)
```

with

```python
        image_spec = state.get("image_spec")
        extra = {"image_spec": image_spec} if image_spec else {}
        stage_plans = plan(request, llm, detail, **extra)
```

In `_run_stage_node`, replace

```python
        # smesh/paravis never see the image description: it only feeds CAD construction
        if state.get("image_description"):
            from salome_agent.specialists.geom import image_block
            blocks.append(image_block(state["image_description"]))
```

with

```python
        # smesh/paravis never see the image spec: it only feeds CAD construction
        image_spec = state.get("image_spec")
        if image_spec:
            from salome_agent.nodes.vision_spec import parts_block, stage_part_ids
            ids = stage_part_ids(state["plan"], idx, image_spec)
            if ids:
                blocks.append(parts_block(image_spec, ids))
```

`salome_agent/specialists/geom.py`:

Replace the last paragraph of `_GEOM_BASE` (from `If your first message includes a "Description of the user's reference image(s)"` to the closing `"""`) with:

```python
If your first message includes a "Reference image spec" block, an image-analysis
agent read it from the user's reference image(s) (you do not see the images): the
exact data of the part(s) this stage builds, as JSON. Build from it as given: a
"profile" base goes into make_profile_face unchanged, then make_prism for
"extrude" or make_revolution for "revolve"; "placement" gives where the part sits
and which way it faces. Values written "~N" are estimates; plain numbers were read
from the image. "conflicts" name values that disagree: build anyway and let the
tool errors tell you which value fits. Explicit dimensions in the stage's params
win over the spec."""
```

Replace `run_geom_specialist` and delete `image_block`:

```python
def run_geom_specialist(stage_plan: dict, llm, client, iteration_cap: int,
                        upstream: dict | None = None,
                        tool_selection: str = "hints",
                        verify_cadence: str = "final",
                        retry: dict | None = None,
                        seed_call_log: list[dict] | None = None,
                        seed_calls: list[dict] | None = None,
                        reference_block: str | None = None) -> dict:
    """Kept as the geom entry point for the existing tests and callers; the
    loop itself is generic now."""
    blocks = []
    if upstream:
        blocks.append(_upstream_block(upstream))
    if reference_block:
        blocks.append(reference_block)
    return ToolSpecialist(GEOM_SPEC).run(
        stage_plan, llm, client, iteration_cap, blocks=tuple(blocks),
        tool_selection=tool_selection, verify_cadence=verify_cadence,
        retry=retry, seed_call_log=seed_call_log, seed_calls=seed_calls,
        upstream=tuple(upstream or ()))
```

Then confirm nothing still refers to the old names:

```bash
grep -rn "image_description\|image_block\|format_description" salome_agent tests | grep -v "specialists/amend.py\|test_amend.py\|test_graph_amend.py"
```

Expected: no output (amend keeps its `image_description` string parameter on purpose).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all PASS (full unit suite: this task closes the gap left by Task 3).

If `_capture_stage_blocks` sees no blocks because `_run_stage_node` fails before calling the specialist, read the traceback (drop the `try/except` temporarily) and add the missing state key to the test's `state` dict — do not change `_run_stage_node` for the test.

- [ ] **Step 5: Commit**

```bash
git add salome_agent/state.py salome_agent/runner.py salome_agent/graph.py salome_agent/specialists/geom.py tests/unit/test_graph.py tests/unit/test_graph_amend.py tests/unit/test_specialist_geom.py
git commit -m "feat: image spec in run state; each geom stage gets its parts' exact JSON, amend the whole spec"
```

---

### Task 6: Report section "Reference image"

**Files:**
- Modify: `salome_agent/nodes/report.py`
- Test: `tests/unit/test_report.py`

**Interfaces:**
- Consumes: `unclaimed_parts`, `all_conflicts` (Task 2); state keys `image_spec`, `plan`.
- Produces: `write_report` output gains a `## Reference image` section when there is something to say.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_report.py`:

```python
def test_report_says_an_assumed_2d_target_was_built_flat():
    spec = {"dimensionality": {"value": "2d", "assumed": True}, "parts": [{"id": "P1"}]}
    out = write_report(_state(image_spec=spec, plan=[]))
    assert "## Reference image" in out
    assert 'Built as a flat face: the image gave no thickness -- say "extrude N mm" to make it solid.' in out


def test_report_lists_unclaimed_parts_and_conflicts():
    spec = {"dimensionality": {"value": "3d"},
            "parts": [{"id": "P1", "conflicts": ["arc off radius"]}, {"id": "P2"}],
            "conflicts": ["datum unclear"]}
    out = write_report(_state(image_spec=spec, plan=[{"module": "geom", "parts": ["P1"]}]))
    assert "Parts in the image spec no stage built: P2." in out
    assert "- datum unclear" in out and "- P1: arc off radius" in out


def test_report_has_no_image_section_when_nothing_to_say():
    spec = {"dimensionality": {"value": "3d"}, "parts": [{"id": "P1"}]}
    assert "## Reference image" not in write_report(_state(image_spec=spec, plan=[]))
    assert "## Reference image" not in write_report(_state())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_report.py -q`
Expected: FAIL (no "## Reference image" section)

- [ ] **Step 3: Write the implementation**

In `salome_agent/nodes/report.py`, add above `write_report`:

```python
def _image_lines(state: dict) -> list[str]:
    from salome_agent.nodes.vision_spec import all_conflicts, unclaimed_parts

    spec = state.get("image_spec")
    if not isinstance(spec, dict):
        return []
    lines = []
    dim = spec.get("dimensionality")
    if isinstance(dim, dict) and dim.get("value") == "2d" and dim.get("assumed"):
        lines.append('Built as a flat face: the image gave no thickness -- say "extrude N mm" '
                     "to make it solid.")
    unclaimed = unclaimed_parts(state.get("plan") or [], spec)
    if unclaimed:
        lines.append(f"Parts in the image spec no stage built: {', '.join(unclaimed)}.")
    conflicts = all_conflicts(spec)
    if conflicts:
        lines += ["Conflicts the vision check could not resolve:"] + [f"- {c}" for c in conflicts]
    return ["## Reference image", "", *lines, ""] if lines else []
```

In `write_report`, insert right after `lines += _timing_lines(state)`:

```python
    lines += _image_lines(state)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add salome_agent/nodes/report.py tests/unit/test_report.py
git commit -m "feat: report notes an assumed flat target, unbuilt spec parts and unresolved vision conflicts"
```

---

### Task 7: Live verification and spec doc update

**Files:**
- Modify: `docs/superpowers/specs/2026-10-02-vision-structured-spec-design.md` (record the three planning deviations listed at the top of this plan)

This task needs the headless server running on the branch code and real LLM access (the user's `config.json`). Restart the server after Task 6 so it runs current code (the server caches modules): stop the running one by killing its process group, then start it with `setsid scripts/start-headless-server.sh` (it reads the SALOME root from `config.json` `salome.root`).

- [ ] **Step 1: Record the planning deviations in the spec**

Append to the spec doc a section:

```markdown
## Decisions made while planning

- Unknown part ids from the planner: one re-ask inside `plan()` with the list of
  valid ids, then still-unknown ids are dropped with a warning (the LLM client's
  JSON-retry loop only covers unparseable JSON).
- The "assumed 2D" note goes in `report.md` (section "Reference image") and the
  log, not in the chat turn summary.
- If no stage of a plan lists any `parts`, every geom stage gets the whole spec,
  so a planner that ignores `parts` is no worse off than before.
```

Commit:

```bash
git add docs/superpowers/specs/2026-10-02-vision-structured-spec-design.md
git commit -m "docs: vision spec design records the decisions made while planning"
```

- [ ] **Step 2: p1.png, three runs**

For each run (single run budget is a few minutes; run in the foreground with a 900 s timeout):

```bash
.venv/bin/salome-agent "Rebuild this plate as a flat 2D face in the XY plane." --images samples/p1.png --batch -vv
```

(Check the image flag name with `.venv/bin/salome-agent --help` first.) For each run record from the run directory under `~/.local/share/salome-agent/runs/<run_id>/`:
- the vision spec (`-vv` log `image spec:` block): did validation pass first time, how many fix rounds, any `conflicts`;
- the first geom turn time (last planner call to first geom call in `llm_trace.jsonl`);
- the result: check_shape valid, wire count 8, area within 1% of 19380 mm2 (measure with the executor's `basic_properties` / `check_shape` on the result object, as in `tests/salome/test_geom_tools_live.py::test_fillet_p1_plate`).

Expected: 3/3 valid faces, no 600 s cutoff. If not, stop and report the spec and the failure; do not tune prompts without the user.

- [ ] **Step 3: p2.png, one run**

```bash
.venv/bin/salome-agent "Rebuild the part in this image." --images samples/p2.png --batch -vv
```

Expected: completes, no worse than before (compare with the latest earlier p2 run directory if one exists).

- [ ] **Step 4: one photo or render**

Ask the user for a photo or render to use (do not fetch images from the web). Run it the same way. Expected: the spec uses `approx` / `primitive` bases, the run completes.

- [ ] **Step 5: Report**

Summarise per run: vision time, fix rounds, conflicts, first geom turn time, result validity and area. Then hand over to superpowers:finishing-a-development-branch.
