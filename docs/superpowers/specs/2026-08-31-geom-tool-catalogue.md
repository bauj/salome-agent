# GEOM tool catalogue (task 176 + GH issue #3)

**Date:** 2026-08-31
**Branch:** `feat/176-geom-tool-catalogue`
**Status:** design approved, implementation pending (haiku agent)

## 1. Goal

Extend the geom specialist's tool registry from the current 4 tools
(`make_box`, `fillet_edges`, `check_shape`, `list_study_objects`) to a
**curated core** of ~33 tools covering the geometry a real SALOME
module-dev prompt actually hits: primitives, basic construction
objects, complex (swept) objects, booleans, transformations, local
edge operations, and display (colour / transparency / display mode).

The display tools require a small, self-contained extension to
`codegen.py` / `assembler.py` — this closes **GH issue #3** (codegen
can only emit `geompy.<fn>(...)` calls today).

### Non-goals (deferred — add later, same pattern, when a stage needs them)

Sketcher (2D and 3D), parametric / interpolation curves, polylines,
build-by-blocks, groups, import / export, shape repair (glue,
remove-extra-edges, close contour), field operations, advanced objects
(pipe-tee, di-pipe, smoothing surface).

**Subshape selection & query** (explode, get-shapes-on-plane/cylinder/
shape, sorted centres, measurement) is deferred to a follow-up — see
§10. This catalogue's only selection surface is the `edges` arg on
`fillet_edges` / `chamfer_edges` (`"all"` or a raw integer-ID list).

## 2. Background: the pattern to follow

Every tool is added in the **same six places**, exactly as
`tool_make_box` and `tool_fillet_edges` already are. Read
`executor/geom_tools.py` end to end before starting — it is the
template.

| # | Where | What to add |
|---|-------|-------------|
| 1 | `TOOLS` dict | `"<name>": tool_<name>` |
| 2 | `tool_<name>(executor, args)` fn | resolve refs, call geompy, `CheckShape`, `_publish`, return `{ok, records, sanity}` |
| 3 | `ARG_TYPES` dict | `{"<numeric arg>": float, ...}` for every number the tool takes |
| 4 | `TOOL_SCHEMAS` dict | JSON Schema (`type`/`properties`/`required`) |
| 5 | `CallRecord` in the returned `records` | `fn` = exact geompy method, `args` in exact positional order (minus `theName`), `out_vars`, `publish` |
| 6 | unit test in `tests/unit/test_geom_tools.py` | mocked-geompy test asserting the `CallRecord` and the schema shape |

### Helpers already in `geom_tools.py` — reuse, do not reinvent

- `_next_name(executor, prefix)` → `"<run_prefix><prefix>_<n>"`, monotonic per prefix.
- `_resolve(executor, arg_ref)` → turns an `ArgRef` back into a live object (`literal` / `ref` / `list` / `study`).
- `_publish(executor, shape, name)` → `geompy.addToStudy` + records the binding in `executor.geom_bindings`.
- `executor.geom_bindings[name]` → the live `GEOM_Object` for a published name.
- `executor.namespace["geompy"]` → the `geomBuilder.New()` instance.
- `executor.namespace["find_object"]` → whole-study lookup by name (used by the `study` ArgRef kind).

### Spatial feedback — attached centrally, not per handler

`call_tool` calls `_attach_shape_info()` on every successful result: the
last published shape's `{bbox, size, centre}` (mm, bbox-midpoint centre)
goes into `result["shape_info"]` + `result["result_shape"]`. Handlers do
**not** add this themselves, and it is **not** part of any `CallRecord` —
it is feedback for the specialist to check placement (task 188). Fails
soft (a `BoundingBox` error just omits the key).

### Sanity check — do it in every shape-producing tool

```python
is_valid, msg = geompy.CheckShape(shape, True, 1)   # theReturnStatus=1 -> (bool, str); default 0 -> bare bool
if not is_valid:
    return {"ok": False, "error": f"<name>: CheckShape failed ({msg})"}
```

Call `CheckShape` **before** `_publish` — an unpublished failed shape
leaves no orphan to clean up (decision 16 invariant holds by
construction).

## 3. Conventions

### 3.1 Argument kinds (`ArgRef.kind` in `CallRecord.args`)

| kind | schema type | `_fmt_arg` emits | use for |
|------|-------------|------------------|---------|
| `literal` | `number` / `integer` / `string` / `boolean` | `repr(value)` | dimensions, counts, flags |
| `ref` | `string` (a published name) | the bare name | any shape / point / vector / plane the LLM names |
| `list` | `array` | `[a, b, ...]` | edge-id lists, shape lists for `partition` |
| `study` | `string` | `find_object("name")` | cross-stage upstream names (rare in geom) |
| `salomeds_color` | see §5.2 | `SALOMEDS.Color(r, g, b)` | **new** — `set_color` only |

### 3.2 `axis` shorthand

Tools that need a rotation / revolution / mirror axis (a `GEOM_Object`
vector) accept **either**:

- a string `"OX"` / `"OY"` / `"OZ"` — the handler builds it with
  `geompy.MakeVectorDXDYDZ` (1,0,0 / 0,1,0 / 0,0,1) and emits that as
  an extra `CallRecord` (prefix `vec`), then references it;
- a `ref` name of a vector the LLM created with `make_vector`.

Same pattern for a `plane` arg (`"XOY"`/`"YOZ"`/`"ZOX"` →
`geompy.MakePlane` at origin with the matching normal, trim 2000) and
a `point` arg (`"O"` → `geompy.MakeVertex(0,0,0)`).

Implement this once as `_resolve_axis(executor, arg, records)` /
`_resolve_plane` / `_resolve_point` helpers that append to the
`records` list and return the ref name. **Task 1 below adds these
helpers before any tool that needs them.**

### 3.3 Angles

geompy rotation / revolution angles are **radians**. The tools accept
**degrees** (`angle_deg`) because that is what an LLM reasons in. The
handler does `rad = math.radians(args["angle_deg"])` and puts the
**float radians** into the `CallRecord` as a `literal`. `result.py`
then shows the literal radians value — acceptable, no `import math`
needed in the generated script.

### 3.4 Naming prefixes (arg to `_next_name`)

`box` `cyl` `sph` `cone` `torus` `face` `disk` `pt` `vec` `line`
`plane` `circ` `prism` `rev` `pipe` `thick` `fill` `fuse` `cut`
`common` `part` `tr` `rot` `mir` `scale` `mtr` `mrot` `off` `fillet`
`chamfer`

Display tools (`set_color` / `set_transparency` / `set_display_mode`)
publish nothing and allocate no name.

## 4. Part A — shape tools

All signatures below are **verified against the live build**
(`geomBuilder.geomBuilder`, 2026-08-31). `theName` is always the last
positional arg and is **never** part of the `CallRecord` (the tool
calls `_publish` itself).

Legend: **R** = required. Args are listed in geompy positional order.

### 4.1 Primitives

#### `make_box_two_pnt`
- geompy: `MakeBoxTwoPnt(thePnt1, thePnt2)`
- args: `pnt1` (ref, R), `pnt2` (ref, R)
- ARG_TYPES: —
- prefix: `box`
- notes: opposite corners. The origin-anchored `make_box` (dx/dy/dz)
  already exists — keep it.

#### `make_cylinder`
- geompy: `MakeCylinderRH(theR, theH)`
- args: `radius` (number, R), `height` (number, R)
- ARG_TYPES: `{radius: float, height: float}`
- prefix: `cyl`
- notes: axis = +Z at origin. Placement elsewhere = `translate` / `rotate` after.

#### `make_sphere`
- geompy: `MakeSphereR(theR)` when only `radius` given;
  `MakeSpherePntR(thePnt, theR)` when `center` (ref) given.
- args: `radius` (number, R), `center` (ref, optional)
- ARG_TYPES: `{radius: float}`
- prefix: `sph`
- notes: two `fn` values depending on `center` presence — pick in the handler.

#### `make_cone`
- geompy: `MakeConeR1R2H(theR1, theR2, theH)`
- args: `radius1` (number, R), `radius2` (number, R), `height` (number, R)
- ARG_TYPES: `{radius1: float, radius2: float, height: float}`
- prefix: `cone`
- notes: `radius2 = 0` → sharp cone. Axis +Z at origin.

#### `make_torus`
- geompy: `MakeTorusRR(theRMajor, theRMinor)`
- args: `radius_major` (number, R), `radius_minor` (number, R)
- ARG_TYPES: `{radius_major: float, radius_minor: float}`
- prefix: `torus`

#### `make_face`
- geompy: `MakeFaceHW(theH, theW, theOrientation)`
- args: `height` (number, R), `width` (number, R), `orientation`
  (integer 1/2/3 = XOY/YOZ/ZOX, default 1)
- ARG_TYPES: `{height: float, width: float, orientation: int}`
- prefix: `face`
- notes: planar rectangular face centred at origin.

#### `make_disk`
- geompy: `MakeDiskR(theR, theOrientation)`
- args: `radius` (number, R), `orientation` (integer 1/2/3, default 1)
- ARG_TYPES: `{radius: float, orientation: int}`
- prefix: `disk`

### 4.2 Basic objects

#### `make_point`
- geompy: `MakeVertex(theX, theY, theZ)`
- args: `x` (number, R), `y` (number, R), `z` (number, R)
- ARG_TYPES: `{x: float, y: float, z: float}`
- prefix: `pt`
- sanity: skip `CheckShape` for a vertex (always valid) — return `{"ok": True, ...}`.

#### `make_vector`
- geompy: `MakeVectorDXDYDZ(theDX, theDY, theDZ)`
- args: `dx` (number, R), `dy` (number, R), `dz` (number, R)
- ARG_TYPES: `{dx: float, dy: float, dz: float}`
- prefix: `vec`
- sanity: skip.

#### `make_line`
- geompy: `MakeLineTwoPnt(thePnt1, thePnt2)`
- args: `pnt1` (ref, R), `pnt2` (ref, R)
- prefix: `line`

#### `make_plane`
- geompy: `MakePlane(thePnt, theVec, theTrimSize)`
- args: `point` (ref, R), `normal` (ref, R), `trim_size` (number, default 2000)
- ARG_TYPES: `{trim_size: float}`
- prefix: `plane`

#### `make_circle`
- geompy: `MakeCircle(thePnt, theVec, theR)`
- args: `center` (ref, R), `normal` (ref, R), `radius` (number, R)
- ARG_TYPES: `{radius: float}`
- prefix: `circ`

### 4.3 Complex (swept / derived) objects

#### `make_prism`  (linear extrusion)
- geompy: `MakePrismDXDYDZ(theBase, theDX, theDY, theDZ)`
- args: `base` (ref, R — a face/wire/edge), `dx` (number, R), `dy` (number, R), `dz` (number, R)
- ARG_TYPES: `{dx: float, dy: float, dz: float}`
- prefix: `prism`

#### `make_revolution`
- geompy: `MakeRevolution(theBase, theAxis, theAngle)`
- args: `base` (ref, R), `axis` (axis-shorthand or ref, R), `angle_deg` (number, R)
- ARG_TYPES: `{angle_deg: float}` (converted to radians literal — §3.3)
- prefix: `rev`

#### `make_pipe`
- geompy: `MakePipe(theBase, thePath)`  (`IsGenerateGroups` left default `False`, omit from CallRecord)
- args: `base` (ref, R), `path` (ref, R — a wire/edge)
- prefix: `pipe`

#### `make_thick_solid`
- geompy: `MakeThickSolid(theShape, theThickness, theFacesIDs, theInside=False)`
- args: `shape` (ref, R), `thickness` (number, R), `faces` (list of ints, R
  — face subshape IDs to remove/open), `inside` (boolean, default False)
- ARG_TYPES: `{thickness: float}`
- prefix: `thick`
- CallRecord args: `[ref shape, literal thickness, literal faces, literal inside]`
- **`faces` is required**: verified live — `MakeThickSolid` on a *closed*
  solid with an empty list always raises `RuntimeError: Offset driver
  failed`. At least one face must be removed to open the solid. Nicer
  face selection comes with task 176b; for now the caller passes explicit
  IDs.

#### `make_filling`
- geompy: `MakeFilling(theContours)`  (all tolerance/degree params left default, omit)
- args: `contours` (list of refs, R — edges/wires)
- prefix: `fill`

### 4.4 Boolean operations

Note the real signatures carry trailing optional params
(`checkSelfInte`, `theName`, `theFuzzyParam`) — the `CallRecord` stops
at the last **required** positional (the two shapes / the lists).

#### `fuse`
- geompy: `MakeFuse(theShape1, theShape2)`
- args: `shape1` (ref, R), `shape2` (ref, R)
- prefix: `fuse`
- notes: for >2 inputs use `MakeFuseList(theShapesList)` — expose as
  `fuse` accepting `shapes` (list, len ≥ 2): handler picks
  `MakeFuse` for exactly 2, `MakeFuseList` for more, and the
  `CallRecord.fn` / `args` reflect which was used.

#### `cut`
- geompy: `MakeCut(theShape1, theShape2)`
- args: `main` (ref, R), `tool` (ref, R)
- prefix: `cut`
- notes: multi-tool → `MakeCutList(theMainShape, theShapesList)`; same
  handler-branch pattern as `fuse`.

#### `common`
- geompy: `MakeCommon(theShape1, theShape2)`  (multi → `MakeCommonList(theShapesList)`)
- args: `shapes` (list of refs, len ≥ 2, R)
- prefix: `common`

#### `partition`
- geompy: `MakePartition(ListShapes, ListTools=[])`
- args: `shapes` (list of refs, R), `tools` (list of refs, default `[]`)
- prefix: `part`
- CallRecord args: `[list shapes, list tools]`.

### 4.5 Transformations — **copy semantics only**

Every one produces a **new** published shape; the input is untouched.

#### `translate`
- geompy: `MakeTranslation(theObject, theDX, theDY, theDZ)`
- args: `shape` (ref, R), `dx` (number, R), `dy` (number, R), `dz` (number, R)
- ARG_TYPES: `{dx: float, dy: float, dz: float}`
- prefix: `tr`

#### `rotate`
- geompy: `MakeRotation(theObject, theAxis, theAngle)`
- args: `shape` (ref, R), `axis` (axis-shorthand or ref, R), `angle_deg` (number, R)
- ARG_TYPES: `{angle_deg: float}` (→ radians literal)
- prefix: `rot`

#### `mirror`
- geompy: `MakeMirrorByPlane(theObject, thePlane)` — default;
  `MakeMirrorByAxis(theObject, theAxis)` if `by` == `"axis"`;
  `MakeMirrorByPoint(theObject, thePoint)` if `by` == `"point"`.
- args: `shape` (ref, R), `by` (enum `plane`/`axis`/`point`, default `plane`),
  `entity` (plane-shorthand / axis-shorthand / `"O"` / ref, R)
- prefix: `mir`

#### `scale`
- geompy: `MakeScaleTransform(theObject, thePoint, theFactor)` — uniform;
  `MakeScaleAlongAxes(theObject, thePoint, theFactorX, theFactorY, theFactorZ)` if per-axis factors given.
- args: `shape` (ref, R), `factor` (number) **or** `factor_x`/`factor_y`/`factor_z` (numbers),
  `center` (`"O"` or ref, default `"O"`)
- ARG_TYPES: `{factor: float, factor_x: float, factor_y: float, factor_z: float}`
- prefix: `scale`

#### `multi_translate`
- geompy: `MakeMultiTranslation1D(theObject, theVector, theStep, theNbTimes)` — 1D;
  `MakeMultiTranslation2D(theObject, theVector1, theStep1, theNbTimes1, theVector2, theStep2, theNbTimes2)` — 2D.
- args (1D): `shape` (ref, R), `vector` (axis-shorthand or ref, R), `step` (number, R), `count` (integer, R)
- args (2D): + `vector2`, `step2`, `count2`
- ARG_TYPES: `{step: float, count: int, step2: float, count2: int}`
- prefix: `mtr`

#### `multi_rotate`
- geompy: `MultiRotate1DNbTimes(theObject, theAxis, theNbTimes)` — full circle, N copies;
  `MultiRotate1DByStep(theObject, theAxis, theAngleStep, theNbTimes)` if `angle_deg` given.
- args: `shape` (ref, R), `axis` (axis-shorthand or ref, R), `count` (integer, R), `angle_deg` (number, optional)
- ARG_TYPES: `{count: int, angle_deg: float}` (→ radians literal for the ByStep form)
- prefix: `mrot`

#### `offset_shape`
- geompy: `MakeOffset(theObject, theOffset)`
- args: `shape` (ref, R), `offset` (number, R — signed distance)
- ARG_TYPES: `{offset: float}`
- prefix: `off`

#### `position`  (Modify Location)
- geompy: `MakeMarker(*origin, *x_dir, *y_dir)` then
  `MakePosition(theObject, None, marker)` (start LCS `None` = global frame)
- args: `shape` (ref, R), `origin` (list[3], default `[0,0,0]`),
  `x_dir` (list[3], default `[1,0,0]`), `y_dir` (list[3], default `[0,1,0]`)
- coercion: element-wise `float()` in the handler (list args, not ARG_TYPES)
- two CallRecords: `MakeMarker` (prefix `lcs`, 9 literal floats) then
  `MakePosition` (prefix `pos`, args `[ref shape, literal None, ref marker]`)
- copy semantics verified; place a shape's local frame onto a target frame
  — for assembling parts. `MakePosition` accepts a `None` start LCS so only
  the target marker is built.

### 4.6 Local edge operations

#### `fillet_edges`  — **FIX the existing tool**

`geompy.MakeFilletEdges` **does not exist on this build** (verified —
`geomBuilder.py` has `MakeFillet`, `MakeFilletAll`, `MakeFillet1D/2D`,
no `MakeFilletEdges`). The current `tool_fillet_edges`
(`geom_tools.py:65`) would `AttributeError` at runtime.

Replace the geompy call with:

```python
edge_type = geompy.ShapeType["EDGE"]
filleted = geompy.MakeFillet(shape, radius, edge_type, edge_ids)
```

- geompy: `MakeFillet(theShape, theR, theShapeType, theListShapes)`
- args: `shape` (ref, R), `radius` (number, R), `edges` (`"all"` or list of ints, R)
- CallRecord: `fn="MakeFillet"`, args `[ref shape, literal radius, literal <edge_type int>, literal edge_ids]`
  — `geompy.ShapeType["EDGE"]` evaluates to a small int; emit that int as a literal so `result.py` needs no `geompy.ShapeType` lookup. **Verify the int value in the live test** and hard-code it with a comment.
- ARG_TYPES: `{radius: float}` (unchanged)
- prefix: `fillet` (unchanged)
- keep the existing `"all"` → `SubShapeAllIDs(shape, EDGE)` id-list expansion and the `CheckShape` guard.

#### `chamfer_edges`
- geompy: `MakeChamferEdges(theShape, theD1, theD2, theEdges)`
- args: `shape` (ref, R), `d1` (number, R), `d2` (number, optional — defaults to `d1`), `edges` (`"all"` or list of ints, R)
- ARG_TYPES: `{d1: float, d2: float}`
- prefix: `chamfer`
- `"all"` expansion + `CheckShape` guard, same as `fillet_edges`.

## 5. Part B — display tools + codegen extension (GH issue #3)

### 5.1 `CallRecord` gains an optional `emit` field

```python
class CallRecord(TypedDict):
    fn: str
    args: list[ArgRef]
    out_vars: list[str]
    publish: list[str] | None
    emit: NotRequired[str]        # "geompy_call" (default) | "method_call" | "gui_call"
    recv: NotRequired[str]        # ref name of the receiver, for method_call / gui_call
```

`codegen.generate()` per record:

| `emit` | line emitted | assign? | `addToStudy`? |
|--------|--------------|---------|---------------|
| `"geompy_call"` / absent | `{out} = geompy.{fn}({args})` | yes | if `publish` |
| `"method_call"` | `{recv}.{fn}({args})` | no | no |
| `"gui_call"` | `try:\n    gg.{fn}(salome.ObjectToID({recv}), {args})\nexcept Exception:\n    pass` | no | no |

`_reachable()` currently keys off `out_vars`. Display records have
none — include a display record **iff its `recv` is in the reachable
set**. Add that as a second condition in the reachability sweep and in
the emit filter.

### 5.2 `_fmt_arg` gains `salomeds_color`

```python
if kind == "salomeds_color":
    r, g, b = arg["value"]                     # ints 0-255
    return f"SALOMEDS.Color({r/255:.4f}, {g/255:.4f}, {b/255:.4f})"
```

### 5.3 `assembler.py` conditional preambles

Existing assembler already adds init blocks conditionally (session 8
rewrite). Add:

- if any emitted line contains `SALOMEDS.Color(` → prepend
  `from salome.kernel import SALOMEDS` (verified import path on this build).
- if any emitted line contains `gg.` → prepend
  `gg = salome.ImportComponentGUI("GEOM")`
  (headless: prints a warning, returns a safe no-op stub — verified;
  the `try/except` around the call is belt-and-braces).

### 5.4 The three display tools

All three: `emit` != `geompy_call`, no `out_vars`, no `publish`, no
name allocation, `sanity` `{"ok": True}`. The handler still executes
the call live (so `--live-gui` shows it immediately).

#### `set_color`
- live call: `shape.SetColor(SALOMEDS.Color(r/255, g/255, b/255))`
  (`from salome.kernel import SALOMEDS` in the executor namespace) — **works headless, persists in the study.**
- args: `shape` (ref, R), `red` (integer 0-255, R), `green` (integer 0-255, R), `blue` (integer 0-255, R)
- ARG_TYPES: `{red: int, green: int, blue: int}`
- CallRecord: `emit="method_call"`, `recv=<shape name>`, `fn="SetColor"`,
  `args=[{"kind": "salomeds_color", "value": [r, g, b]}]`

#### `set_transparency`
- live call: `gg = salome.ImportComponentGUI("GEOM"); gg.setTransparency(salome.ObjectToID(shape), t)`
- args: `shape` (ref, R), `transparency` (number 0.0-1.0, R)
- ARG_TYPES: `{transparency: float}`
- CallRecord: `emit="gui_call"`, `recv=<shape name>`, `fn="setTransparency"`,
  `args=[{"kind": "literal", "value": t}]`

#### `set_display_mode`
- live call: `gg.setDisplayMode(salome.ObjectToID(shape), mode)`
- args: `shape` (ref, R), `mode` (integer enum: 0 wireframe / 1 shading / 2 shading+edges, R)
- ARG_TYPES: `{mode: int}`
- CallRecord: `emit="gui_call"`, `recv=<shape name>`, `fn="setDisplayMode"`,
  `args=[{"kind": "literal", "value": mode}]`

### 5.5 Executor namespace additions

`executor` must expose `SALOMEDS` and (in live/GUI mode) a cached
`ImportComponentGUI("GEOM")` handle for the handlers. Add `SALOMEDS`
to `build_namespace()` unconditionally; the GUI handle lazily on first
display-tool call (mirrors the `_ensure_pvsimple` deferral).

## 6. System prompt update

`GEOM_SYSTEM_PROMPT` (`salome_agent/specialists/geom.py`) — add one
paragraph listing the tool families now available so the model knows
to reach for them:

> You can create primitives (`make_box`, `make_cylinder`,
> `make_sphere`, `make_cone`, `make_torus`, `make_face`, `make_disk`),
> basic objects (`make_point`, `make_vector`, `make_line`,
> `make_plane`, `make_circle`), swept shapes (`make_prism`,
> `make_revolution`, `make_pipe`, `make_thick_solid`, `make_filling`),
> combine shapes (`fuse`, `cut`, `common`, `partition`), transform
> them — always producing a copy — (`translate`, `rotate`, `mirror`,
> `scale`, `multi_translate`, `multi_rotate`, `offset_shape`), round
> or bevel edges (`fillet_edges`, `chamfer_edges`), and set display
> properties (`set_color`, `set_transparency`, `set_display_mode`).
> Refer to any shape, point, vector or plane by the name it was
> published under. For an axis you may pass `"OX"`, `"OY"` or `"OZ"`
> instead of building a vector. Angles are in degrees.

## 7. Testing

### 7.1 Unit (`tests/unit/test_geom_tools.py`, mocked geompy)

Per tool: assert `list_tools()` includes it with a non-empty schema;
call the handler with a `MagicMock` geompy; assert the returned
`CallRecord` `fn` + `args` order/kinds; assert `ARG_TYPES` coerces a
string-typed number. One test per tool minimum.

`tests/unit/test_codegen.py` — new cases: `method_call` emits
`recv.fn(...)` with no assign/publish; `gui_call` emits the
`try/except` block; `salomeds_color` arg formats to
`SALOMEDS.Color(...)`; a display record is dropped when its `recv` is
unreachable and kept when reachable.

`tests/unit/test_assembler.py` — `SALOMEDS` import appears iff a color
line is present; `gg = ImportComponentGUI` appears iff a `gg.` line is
present.

### 7.2 Live (`tests/salome/test_geom_tools_live.py`, skips without `config.json` + `SALOME_ROOT`)

- Every primitive / basic / complex tool: build it, `CheckShape` passes.
- One boolean, one transform, `fillet_edges` (the fix), `chamfer_edges`: end to end.
- `set_color`: build box, `set_color`, assert `box.GetColor()` matches (÷255, tolerance 1e-3).
- **Verify `geompy.ShapeType["EDGE"]` int value** and confirm it is what the `fillet_edges` / `chamfer_edges` CallRecords hard-code.
- Assemble a `result.py` that uses color + a boolean, run it standalone under `salome shell`, assert it executes clean (headless no-op for `gg.` lines).

### 7.3 Live LLM smoke (one run, per the session-7 lesson)

`salome-agent "make two 10mm cubes 30mm apart, fuse them, fillet all
edges 2mm, make it red"` against the real endpoint — exercises
primitives + transform + boolean + local op + display in one plan.

## 8. Implementation task order (for the haiku agent)

Each task: implement, add unit tests, run `pytest tests/unit`, commit
with a bare `feat:` / `fix:` subject line (repo convention — no body,
no trailers).

1. **Resolver helpers** — `_resolve_axis` / `_resolve_plane` /
   `_resolve_point` + `math` import + the `salomeds_color` /
   `emit` / `recv` additions to `CallRecord`. No tools yet. Unit-test
   the helpers with mocked geompy.
2. **Primitives** — `make_box_two_pnt`, `make_cylinder`, `make_sphere`,
   `make_cone`, `make_torus`, `make_face`, `make_disk`.
3. **Basic objects** — `make_point`, `make_vector`, `make_line`,
   `make_plane`, `make_circle`.
4. **Complex** — `make_prism`, `make_revolution`, `make_pipe`,
   `make_thick_solid`, `make_filling`.
5. **Booleans** — `fuse`, `cut`, `common`, `partition` (incl. the
   2-vs-list branch).
6. **Transformations** — `translate`, `rotate`, `mirror`, `scale`,
   `multi_translate`, `multi_rotate`, `offset_shape`.
7. **Local ops** — fix `fillet_edges` (`MakeFillet`), add `chamfer_edges`.
8. **Codegen extension** — `codegen.py` `emit` handling +
   `salomeds_color` in `_fmt_arg` + reachability for display records;
   `assembler.py` conditional preambles. Full `test_codegen.py` /
   `test_assembler.py` coverage.
9. **Display tools** — `set_color`, `set_transparency`,
   `set_display_mode` + executor namespace (`SALOMEDS`, lazy GUI handle).
10. **System prompt** paragraph.
11. **Live tests** — `tests/salome/test_geom_tools_live.py`; run the
    §7.3 LLM smoke; record results and any gotchas in the project
    memory. Merge to `main` only after §7.2 + §7.3 pass live
    (`feedback-work-by-branch`).

## 9. Open questions / risks

- **`fillet_edges` "all" on a non-solid** — `SubShapeAllIDs(shape,
  EDGE)` on a face/wire still returns ids; `MakeFillet` may reject a
  0-radius or a wire input. Keep the `CheckShape` guard; let it
  surface as `ok: False`.
- **`mirror` / `scale` multi-form dispatch** — the handler chooses the
  geompy method from the args. Keep the choice logic tiny and
  covered by a unit test per branch.
- **`gui_call` in a headless `result.py`** — verified the
  `ImportComponentGUI` stub no-ops without raising for
  `setColor`/`setTransparency`/`setDisplayMode`; the `try/except` is
  still there in case a future SALOME build's stub raises.
- **`ShapeType["EDGE"]` literal** — hard-coding the int couples
  `result.py` to a build constant. Verified on this build:
  `VERTEX=7 EDGE=6 WIRE=5 FACE=4 SHELL=3 SOLID=2 COMPOUND=0`. The live
  test must still assert `EDGE == 6`.

## 10. Deferred follow-up — subshape selection & query family (Task 176b)

**Not in this catalogue. Its own spec + plan.** This is the answer to
"fillet only the top four edges" — today the LLM would have to guess
raw integer IDs with no way to discover them.

### Why it is separate

- One new architectural idea: **query tools that return data to the
  LLM, not a shape.** A query tool produces **no `CallRecord`** — it
  runs live, puts an ID list / coordinates / measurements into its
  tool result, the model reads that and passes the resolved IDs as
  **literals** to a downstream `fillet_edges` / `chamfer_edges` /
  `get_subshape` call. Those literals are all that reaches `result.py`
  — reproducible, but the script carries bare magic numbers (emit an
  explanatory comment from the originating query if cheap).
- `explode` / `get_subshape` *do* publish and *do* emit `CallRecord`s
  — but `SubShapeAllSortedCentres` returns a **list**, so codegen
  needs multi-`out_var` / multi-`publish` handling it does not have
  yet.

### Proposed tools (all signatures verified on this build, 2026-08-31)

#### `explode` — publish every subshape as its own named object
- geompy: `SubShapeAllSortedCentres(aShape, aType)` (sorted → deterministic order)
- args: `shape` (ref, R), `sub_type` (enum `vertex`/`edge`/`wire`/`face`/`shell`/`solid`, R)
- publishes `<shape>_<sub_type>_<i>` for each; normal `ref` args work afterwards
- prefix: derived per subshape
- codegen: needs list-return support (multiple `out_vars` + `publish` from one call)

#### `get_subshapes` — query, returns an ID list (the `select_subshapes` idea)
- **no `CallRecord`** — result payload `{"ids": [...], "centres": [[x,y,z], ...]}`
- args: `shape` (ref, R), `sub_type` (enum, R), `where` (one of):
  - `"all"` → `SubShapeAllSortedCentresIDs(shape, type)`
  - `{"on_plane": {"point": [x,y,z], "normal": [x,y,z]}, "state": "on"|"in"|"out"|"onin"|"onout"}`
    → `GetShapesOnPlaneWithLocationIDs(shape, type, normalVec, pnt, GEOM.ST_*)`
  - `{"on_cylinder": {"axis": ..., "radius": r}, "state": ...}` → `GetShapesOnCylinderIDs`
  - `{"on_sphere": {"center": [x,y,z], "radius": r}, "state": ...}` → `GetShapesOnSphereIDs`
  - `{"on_shape": "<ref name>", "state": ...}` → `GetShapesOnShapeIDs`
  - `{"sorted_by": "x"|"y"|"z", "take": "max"|"min", "count": N}`
    → sort `SubShapeAllSortedCentres` by that centre coord, slice N
  - `{"by_size": "longest"|"shortest"|"largest"|"smallest", "count": N}`
    → `BasicProperties` per subshape (length for edges, area for faces), slice N
- `GEOM.ST_*` enum reached via `geomBuilder.GEOM.ST_ON` / `ST_IN` / `ST_OUT` / `ST_ONIN` / `ST_ONOUT` (verified).

#### `get_subshape` — build one object from an ID list
- geompy: `GetSubShape(aShape, ListOfID)` (single) or `SubShapes(aShape, anIDs)` (compound)
- args: `shape` (ref, R), `ids` (list of ints, R)
- publishes, emits a `CallRecord`; prefix `sub`

#### measurement helpers — query, no `CallRecord`, feed the model numbers
- `bounding_box(shape)` → `BoundingBox(theShape)` → `[xmin,xmax,ymin,ymax,zmin,zmax]`
- `min_distance(shape1, shape2)` → `MinDistance(s1, s2)` → float
- `basic_properties(shape)` → `BasicProperties(theShape)` → `[length, area, volume]`
- `point_coordinates(point_ref)` → `PointCoordinates(pt)` → `[x,y,z]`

### Status

Implemented on branch `feat/185-subshape-selection`
(plan: `docs/superpowers/plans/2026-09-03-geom-subshape-selection.md`).
Shipped: `explode`, `get_subshape`, `get_subshapes` (predicates `all` /
`on_plane` / `sorted_by` / `by_size`), `min_distance`, `basic_properties`,
codegen `list_call` emit, the query-then-act prompt paragraph. Deferred:
`get_subshapes` predicates `on_cylinder` / `on_sphere` / `on_shape`; the
`bounding_box` / `point_coordinates` helpers (the existing `measure` tool
already covers them); checklist item 8 (inline `where` sugar on
`fillet_edges` / `chamfer_edges`).

### Task 176b checklist (when picked up)

1. Codegen: multi-`out_var` / multi-`publish` list-return support + a
   test.
2. Specialist loop: confirm a tool result with an `ids` payload and no
   `records` round-trips cleanly (no `CallRecord` appended, model sees
   the payload). This mostly works already — verify.
3. `explode` + `get_subshape` (shape-producing).
4. `get_subshapes` with the `"all"` / `on_plane` / `sorted_by` /
   `by_size` predicates (start with those four; add `on_cylinder` /
   `on_sphere` / `on_shape` when needed).
5. Measurement helpers.
6. `GEOM_SYSTEM_PROMPT`: a paragraph on the query-then-act pattern
   ("call `get_subshapes` to find edge IDs, then pass them to
   `fillet_edges`").
7. Live LLM smoke: `"fillet the top 4 edges of a 20mm cube, radius 3mm"`.
8. Optional: have `fillet_edges` / `chamfer_edges` accept a `where`
   object directly (inline `get_subshapes`) as sugar, once the query
   layer is proven.
