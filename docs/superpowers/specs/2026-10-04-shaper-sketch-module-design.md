# SHAPER sketch module — design

Task 301 (SHAPER constrained sketch vs GEOM make_profile_face), with 300 in view.
Session 72, 2026-10-04.

## Problem

p2.png (a construction drawing: Ø100 disc, bars 10 wide on centre lines at ±24°,
ring 8.5, gap 2.5) is dimensioned by centre lines and stroke widths; the voids are
whatever is left over. geom's `make_profile_face` needs every outline traced
explicitly, so the model has to derive the void outlines itself, and vision runs
traced them wrong (runs eeef9d8f2db2, eb9e1c976599).

SHAPER's sketcher has a constraint solver: the model states lines, widths, angles
and relations, the solver computes the geometry, and closed regions fall out of
overlapping strokes. Whether an LLM can drive it well is unknown. A one-shot,
no-tools test would only measure API hallucination, which is what the agent's
tools exist to prevent, so the question has to be answered through the real agent
loop. That needs a real SHAPER module.

## Goals

- A `shaper` module in the engine, built like geom (tool registry, generic
  `ToolSpecialist`), meant to grow into the product, not an experiment harness.
- Tools for a constrained sketch plus simple geometry from it: face from sketch
  regions, extrusion, extrusion cut / fuse, revolution, measure.
- Every sketch call returns what the solver says (DOF, rejected constraints, loose
  ends, regions, moved entities), so the model fixes from measured data.
- result.py rebuilds the sketch with its constraints, so the output stays
  parametric.
- The planner routes to shaper only on explicit cues; geom stays the default.
- Clean-text p2 runs through both modules and can be compared.

## Non-goals

- Planner routing refactor (capability table, two-phase plan). Only a scope line
  and an explicit-name override are added here.
- Sketches on faces of solids, fillet/chamfer, patterns, general booleans between
  solids.
- Changes to salome-agent-gui beyond checking that SHAPER objects appear after a
  live GUI run.
- Solving `PAL_NOT_DONE_ERROR` (see Hand-off to geom); parked.
- Vision changes (task 300).

## Probe findings (headless `salome shell`, SSL mode)

- `salome.shaper.model` loads and runs in a headless session.
- The solver reports DOF (`model.dof(sketch)`) and a conflict message
  (`sketch.solverError()`); removing the offending constraint feature clears the
  error.
- A sketch result lists its closed regions (`ResultConstruction.facesNum()` /
  `face(i)`). Rectangle 40×20 with a Ø10 circle: regions 721.46 and 78.54.
- `model.addFace` with a (result, face) selection gives the face (area 721.46);
  `model.addExtrusion` of it by 10 gives volume 7214.602.
- `model.publishToShaperStudy()` puts the result in the study as a GEOM-typed
  object (`IsShape()` true, type SOLID). geompy operations on it fail with
  `PAL_NOT_DONE_ERROR`; `geompy.RestoreShape(obj.GetShapeStream())` gives a
  working geom copy (volume 7214.602, a cut on it works).
- Not checked yet: whether the solver names the conflicting constraint (the design
  does not depend on it), and live GUI mode.

## Architecture

### Server side: `executor/shaper/`

A sibling of `executor/geom/` with its own `ToolRegistry`:

- `state.py` — the run's SHAPER Part, and per sketch the table from the model's
  ids (`L1`, `C3`) to SHAPER entities and constraint features.
- `sketch.py` — `create_sketch`, `sketch_add`, `sketch_constrain`,
  `sketch_remove`, `sketch_status`.
- `features.py` — `make_face`, `extrude`, `extrude_cut`, `extrude_fuse`,
  `revolve`.
- `queries.py` — `measure`.
- `__init__.py` — `list_tools()` / `call_tool()` with the same contract as geom's.

### Server routing

`list_tools` and `call_tool` requests gain an optional `module` field;
`agent_server/core.py` dispatches to geom or shaper and defaults to geom, so
existing clients and geom's golden `list_tools` baseline do not change.
`refresh`, `forget` and `clear_study` get a shaper branch. The namespace imports
`salome.shaper.model` lazily on the first shaper call, so geom-only runs pay
nothing.

### Engine side

`salome_agent/specialists/shaper.py` registers `ModuleSpec("shaper")` on the
generic `ToolSpecialist`: its prompt, `finish_stage` terminator, `execute` (calls
the client with `module="shaper"`), `render_result`, `tools_for`, `audit` (the
finish gate), `forgets=True`, `catalogue` (so the planner can route to it), and a
SHAPER preamble for codegen. `executor_client.list_tools` / `call_tool` take the
optional `module`.

## Sketch tools

### `create_sketch(plane)`

`plane` is `"XOY"`, `"XOZ"`, `"YOZ"` or `{base, offset}`. Returns the sketch's
published name. Every sketch automatically gets three references: `ORIGIN`,
`OX`, `OY` (projections of the origin and axes), so geometry can be anchored and
DOF can reach 0.

### `sketch_add(sketch, entities)`

Each entity carries an id chosen by the model, unique within its sketch:

- `{id, type: "line", start: [x, y], end: [x, y]}`
- `{id, type: "circle", center, radius}`
- `{id, type: "arc", center, start, end, reversed?}`
- `{id, type: "point", at}`

Any entity may carry `construction: true`: it takes part in constraints but forms
no region and is never a loose end (centre lines, helper lines). Sub-points are
addressed as `L1.start`, `L1.end`, `A2.center`, `C3.center`.

### `sketch_constrain(sketch, constraints)`

Each constraint is `{id?, type, refs: [...], value?}`. Types: coincident,
horizontal, vertical, parallel, perpendicular, tangent, equal, collinear,
midpoint, fixed, length, distance, horizontal_distance, vertical_distance, radius,
angle. Ids default to `K1`, `K2`, ...

Constraints are applied one at a time with a solve after each. A constraint that
makes the solver fail is removed at once and reported in `rejected` with the
solver's message; the others stay. The sketch is never left in a failed state,
and the culprit is named without relying on SHAPER to name it.

### `sketch_remove(sketch, ids)`

Removes entities or constraints. Removing an entity removes the constraints that
reference it; their ids are listed in the result.

### `sketch_status(sketch)`

The full state on demand.

### What every sketch call returns

- `dof`
- `rejected` — constraint id, type, solver message
- `loose_ends` — end points used by only one non-construction curve, with
  coordinates
- `regions` — closed regions: index, area, inside point
- `moved` — entities the solver moved by more than a tolerance, with their new
  coordinates (after solving, the drawn coordinates are not the real ones)

### Validation

Unknown ids, a ref of the wrong kind (a radius on a line, a sub-point that does
not exist) or a duplicate id fail the whole call before anything is added. The
error names the problem and lists the valid ids.

## Feature tools

- `make_face(sketch, regions)` — `regions` is a list of indices from the latest
  status, or `"all"`; one face per region. Indices change with edits, so the
  result.py record selects each region by its inside point, not its index.
- `extrude(base, size | {to, from}, direction?, reversed?)` — `base` is a face or
  a whole sketch (all its regions); `direction` defaults to the sketch normal.
- `extrude_cut(base, target, size | through_all)` and
  `extrude_fuse(base, target, size)` — SHAPER's pocket and boss on an existing
  solid.
- `revolve(base, axis, angle)` — `axis` is a sketch line (`sketch_1:L4`) or `OX`
  / `OY` / `OZ`.
- `measure(names)` — area, volume, face and edge counts, bbox, validity per
  object, with the same keys as geom's.

Every feature call returns the new object's name and its measure block, plus a
`delta` note when something is numerically wrong (a cut that removed nothing, a
fuse that added nothing), as geom does.

## Naming, finish, publishing

- Results get readable names (`sketch_1`, `face_1`, `extrude_1`) under the same
  per-run prefix rule as geom in headless mode, so leftovers and forget behave the
  same. Sketch entity ids stay local to their sketch.
- `finish_stage(result_names)` refuses while any sketch in or under the results
  has loose ends. Conflicts cannot exist (rejected on entry). DOF is only reported
  in the stage summary.
- Objects the results do not reach are removed, as in geom.
- Final results are published through SHAPERSTUDY.

## Hand-off to geom

When a later geom stage names a SHAPERSTUDY object, geom's name resolver restores
a geom copy from its BREP stream. The copy is one-way: editing the sketch does not
update it. Direct use in live GUI mode is checked in the plan's first task; if it
works there, the copy is the headless fallback only. Why direct use fails headless
is parked.

## result.py

Each tool call logs records in SHAPER's Python form (`model.addSketch`,
`Sketch.addLine`, `setCoincident`, `model.addFace` by inside point,
`model.addExtrusion`, ...). Codegen emits them between a SHAPER preamble
(`model.begin()`, part creation) and `model.end()` + `publishToShaperStudy()`.
SHAPER's own dump-to-Python is weighed in the plan as a cross-check.

## Prompt

A short SHAPER system prompt in geom's style: build the sketch in chunks (outline,
construction lines, strokes); constrain with the given values; read `rejected` and
`moved` before the next step; draw what is dimensioned but not built as
construction geometry; reference results by published name.

## Planner routing

- A shaper scope line in the planner prompt's "Module scope" block: shaper =
  constrained / parametric sketches and features built from them.
- geom stays the default for all CAD.
- A module the request names ("in shaper", "using GEOM") always wins.
- The planner prompt's golden baseline is updated once for this; geom's own
  baselines do not change.

## Testing

- Unit (no SALOME): ModuleSpec wiring; argument validation; `render_result`;
  codegen output parses; finish gate blocks on loose ends; server routing on
  `module` with geom as default and geom's golden `list_tools` unchanged; planner
  golden updated.
- SALOME integration (headless, like `tests/salome`): the probe case (regions
  721.46 / 78.54, extrude volume 7214.602); a batch with one conflicting
  constraint rejects and names it and keeps the rest; entity removal cascades;
  `moved` reported after solving; region selection by inside point survives an
  edit; result.py in a fresh session rebuilds the same volume; SHAPERSTUDY publish
  then geom restore-copy works.
- Live agent smoke: "in shaper, plate 40×20 with a Ø10 hole, extruded 10" checked
  by volume, face count and validity.

## The 301 experiment

Once the module works: clean-text p2, 3 runs with "in shaper" and 3 with geom,
same fixed model. Measure runs that end exact (one valid face of the reference
area), wall time, tool turns, reasoning tokens, and final DOF for shaper. The
reference area is computed once by hand.

## Order of work

1. Server routing and the SHAPER state layer (plus the live GUI interop check).
2. Sketch tools and their status feedback.
3. Feature tools and measure.
4. ModuleSpec, prompt, finish gate.
5. Codegen.
6. Planner scope line and override; smoke run.
7. The experiment.
