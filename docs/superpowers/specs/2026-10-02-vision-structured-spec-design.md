# Vision structured spec — design

Tasks 286 (vision follow-ups) — the INPUT problem. Session 63, 2026-10-02.

## Problem

Image runs fail on input, not construction. On p1.png every image run hit the
600 s geom first-turn cutoff with no tool call, while the same part as clean text
builds exactly (area 19380, 8 wires). Causes seen in sessions 57-61:

- the vision agent misreads or hedges, and its JSON is flattened to prose by
  `format_description()` (`salome_agent/nodes/vision.py`);
- `position` is free text, so placement is never numeric;
- the planner restates the prose in stage instructions, rounding (55.426 -> 55.4)
  or inventing precision (session 60);
- geom gets the description twice (planner wording + raw `image_block`), possibly
  conflicting;
- nothing checks the description before planning.

Text2CAD (minimal structured JSON stops hallucination) and ExeCAD (structured input
beats natural language for every model) point the same way: hand over a structured
spec, not prose.

## Goals

- The vision agent returns a per-part feature spec, validated mechanically before
  planning.
- Numbers reach geom as exact JSON, never through the planner's wording.
- Works for sketches, technical drawings, photos, renders, multi-view drawings and
  assemblies (not only 2D profiles).
- p1.png builds end to end.

## Non-goals

- SVG input (task 287); the profile format is chosen so an SVG importer can fill
  it later.
- Geom asking the user a question.
- Per-module model override, a two-call (read, then condense) vision variant, a
  classifier gate.
- ExeCAD / Text2CAD benchmark (task 296).
- smesh / paravis: they never receive image input.

## Data flow

```
image(s) + request
   |
   v
describe_image  (one multimodal call, module="vision")
   |  spec JSON
   v
validate_spec  (pure Python, no SALOME)
   |  errors -> fed back to the same vision conversation,
   |            up to llm.vision_fix_retries (default 2)
   |  still errors -> attached to their part as "conflicts", continue
   v
plan  (sees the compact rendering of the spec)
   |  stages built from the image list "parts": ["P1", ...]
   v
geom stage  (code injects the exact JSON of its listed parts
             + top-level dimensionality / datum / uncertainties)
```

- State key `image_spec` (dict) replaces `image_description` (str). It stays in the
  checkpointer, so chat follow-up turns and amend keep it.
- `format_description()` and the prose `image_block()` are removed; replaced by
  `compact_spec()` (planner) and `parts_block()` (geom).
- Planner stage schema gains optional `parts: [id, ...]`. Unknown ids are fed back
  through the planner's existing JSON-retry path ("unknown part P7; parts are
  P1-P4").
- A spec part no stage claims: a note in the report, not an error (the planner may
  merge parts or the user may have said to ignore one).
- A stage with no `parts` gets no image slice.

## Spec schema

The vision model writes `dimensions_read` before `parts`. Number convention: a
plain number is read from the image or derived from read values; a string `"~40"`
is an estimate.

```json
{
  "dimensionality": {"value": "2d", "assumed": true,
                     "reason": "single outline, no depth cue"},
  "summary": "flat plate: round top lobe over a narrower body with a round bottom end",
  "views": [{"image": 1, "view": "front", "shows": "full outline, all dimensions"}],
  "dimensions_read": [
    {"text": "R96", "means": "radius of top lobe arc, centre O"},
    {"text": "80", "means": "O to centre of bottom arc (both extension lines on centre lines)"}
  ],
  "datum": "O = centre of top lobe; +X right, +Y up",
  "parts": [{
    "id": "P1",
    "name": "plate",
    "base": {"kind": "profile", "plane": "XY",
             "profile": {"start": "96,0,0", "segments": [], "holes": []},
             "extrude": null},
    "placement": {"origin": "0,0,0", "normal": "0,0,1"},
    "features": [],
    "relation": null
  }],
  "uncertainties": ["R36 ring: boss edge or groove?"]
}
```

### `dimensionality`

`value` is `"2d"` or `"3d"`. When one outline gives no depth cue and the request
does not say, the value is `"2d"` with `assumed: true`; the run summary then states
it ("built as a flat face; the image gave no thickness -- say 'extrude N mm' to
make it solid"). A 2D target has a `plane` on its profile and no
`extrude` / `revolve`.

### `parts[].base`

One of:

- `{"kind": "profile", "plane": "XY|YZ|XZ", "profile": {...},
  "extrude": {"height": h, "direction": "x,y,z"?} | null,
  "revolve": {"point": "x,y,z", "direction": "x,y,z", "angle": deg} | null}` —
  `profile` is exactly `make_profile_face`'s argument shape
  (`{start, segments[{to, center|through|via, reverse, fillet}], holes[...]}`, holes
  as outlines, `{center, radius}` circles or `{slot}`), points as `"x,y,z"` strings,
  so geom passes it unchanged. At most one of `extrude` / `revolve`.
- `{"kind": "primitive", "shape": "box|cylinder|sphere|cone|torus", "dims": {...}}`
- `{"kind": "approx", "description": "...", "dims": {...}}` — photos and renders;
  dims mostly `~`.

Exact profiles only where a drawing gives a real outline.

### `parts[].placement`

Numeric: `origin` + `normal` (+ optional `x_dir`), `~` allowed. Free text goes only
in `relation` (`"sits centred on top of P2"`), which is extra context and never a
substitute for the numbers.

### `parts[].features`

- `{"type": "hole", "center": "x,y,z", "diameter": d, "depth": d | "through", "axis": "x,y,z"?}`
- `{"type": "fillet", "radius": r, "where": "text"}`
- `{"type": "chamfer", "size": s, "where": "text"}`
- `{"type": "pattern", "of": <feature index>, "kind": "circular|linear", "count": n, ...}`

`where` stays text: geom's `where` subshape selection reads it.

### Assemblies

Several parts plus `relation`. Optional `parts[].ops: [{"op": "cut|fuse", "with": "P2"}]`.

### Conflicts

After the fix loop, remaining validator errors are stored on their part as
`"conflicts": ["segment 4: arc end (54,-78) is 2.0 from centre (0,-80), radius 54"]`
(top-level errors go to a top-level `conflicts`).

## Validator (`validate_spec`)

Pure Python in `salome_agent/nodes/vision_spec.py`, no SALOME. Returns a list of
error strings, each prefixed with its location (`P1.profile segment 4: ...`).

Checks:

- JSON shape: required keys, `dimensionality.value` in {2d, 3d}, `base.kind` known,
  part ids unique, at most one of extrude / revolve, a 2D target has no
  extrude / revolve.
- Number fields are numbers or parseable `"~N"` strings; points parse as `"x,y,z"`
  (a point name is an error here: the spec has no named points).
- Profile, per outline (outer and outline holes):
  - each centre arc's start and end lie on one radius (tolerance 0.01 mm);
  - a `through` point is not collinear with the arc's ends;
  - segments carry only `to`, so an outline is continuous by format; if the last
    `to` is not `start`, the tool closes it with a straight line, so that is
    allowed too -- the check is only that the outline has at least 2 distinct
    points and is not degenerate (all points on one line);
  - points lie in the profile's `plane`.
- Holes (circles, slots, outlines) lie inside the outer outline's bounding box.
- `pattern.of` indexes an existing feature; `ops.with` names an existing part.
- Fillets are not checked (the tool finds tangent points).

## Fix loop

`describe_images` becomes a loop:

1. Vision call -> JSON (bad JSON: existing `llm.json_retries` path).
2. `validate_spec`. No errors -> done.
3. Errors and retries left: append the assistant answer + a user message listing
   the errors ("Fix these in the spec and answer with the whole corrected JSON") to
   the same conversation, go to 1.
4. Retries spent: attach the errors as `conflicts`, done.

New config knob `llm.vision_fix_retries` (default 2), added to the known `llm.*`
keys and the README config reference.

## Prompts

- Vision: the session-61 drawing-reading rules stay. The JSON section becomes the
  schema above. New rules: "write `dimensions_read` first; every number in `parts`
  comes from it, is derived from it, or carries `~`"; "give a profile as drawn,
  the tool finds tangent points"; points as `"x,y,z"` in the datum frame.
- Planner: the "Image description" paragraph becomes: parts by id; list the ids a
  stage builds in `parts`; never restate their numbers in stage text.
- Geom: the image paragraph becomes: your stage's parts as exact JSON; a `profile`
  goes into `make_profile_face` unchanged; `~` values are estimates; `conflicts`
  name values that disagree -- resolve them with tool errors.
- Golden prompt / tool fixtures regenerated.

## Error handling

- Vision call fails outright: as today, continue without image input, log a warning.
- Bad JSON after `json_retries`: as above.
- Validator errors after `vision_fix_retries`: `conflicts`, planning continues.
- Planner names an unknown part: JSON-retry feedback.

## Testing

Unit:

- `validate_spec`: a correct p1 spec passes; an arc end off its radius, a
  degenerate outline, collinear `through`, a hole outside the outline, a bad `pattern.of`,
  a bad `~` string, a duplicate id each give one clear error.
- Fix loop with a fake LLM: errors are sent back; after retries, `conflicts` are
  attached.
- Planner: `parts` checked against spec ids.
- Geom: a stage receives only its listed parts' JSON.
- `compact_spec()` rendering.

Live (success criteria):

- p1.png, 3 runs (vision is noisy): valid face (check_shape ok, 8 wires), area
  within 1% of 19380 mm2, first geom turn well under the 600 s cutoff.
- p2.png: no worse than today.
- One photo or render: spec uses `approx` / `primitive`, run completes.

## Files touched

- `salome_agent/nodes/vision.py` — prompt, fix loop.
- `salome_agent/nodes/vision_spec.py` (new) — `validate_spec`, `compact_spec`,
  `parts_block`.
- `salome_agent/graph.py` — `image_spec` state, describe node, stage slice injection.
- `salome_agent/nodes/planner.py` — prompt, `parts` validation.
- `salome_agent/specialists/geom.py` — prompt, `parts_block` instead of `image_block`.
- `salome_agent/config.py`, `README.md` — `llm.vision_fix_retries`.
- `salome_agent/nodes/report.py` — assumed-2D note, unclaimed-part note.
- tests + golden fixtures.
