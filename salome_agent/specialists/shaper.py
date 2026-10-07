from executor.shaper.state import POSTAMBLE, PREAMBLE, PUBLISH_LINE
from salome_agent.calc import CALC_PROMPT
from salome_agent.logs import log
from salome_agent.codegen import _reachable, roots_for
from salome_agent.modules import ModuleSpec, Terminator, register
from salome_agent.specialists.geom import FINISH_STAGE_DESCRIPTION, FINISH_STAGE_SCHEMA

SHAPER_SYSTEM_PROMPT = """You build parametric geometry in SHAPER by calling
tools. Lengths are millimetres, unconverted; angles are in
degrees. Reference sketches and features by the name they were published under.

A sketch is entities (lines, circles, arcs, points), optionally tied together
by constraints that the solver keeps. Build it in chunks: create_sketch, then
for one logical piece at a time sketch_add its entities. When the given
dimensions fix the coordinates, draw the entities there: ends that meet close
the contour and the sizes are right without any constraint. Add
sketch_constrain only for what the coordinates do not settle, such as a
tangency you would otherwise have to compute. Entities that are not
constrained to each other do not follow when the solver moves one of them;
coincident ends keep a contour closed when a constraint moves part of it. A
sketch does not need to be fully constrained.
One sketch is edited at a time: when it is done, call close_sketch on it.
Nothing else -- a new sketch, a face, a solid -- can be built while a sketch is
still being edited.

Read every sketch reply before the next step:
- `rejected`: constraints that conflicted with ones already there. Change or
  remove one of the pair; never resend a rejected constraint unchanged.
- `moved`: entities the solver moved, with their new coordinates — these are the
  real coordinates now.
- `loose_ends`: ends of drawn curves that touch nothing — an open contour.
- `regions`: the closed areas, each with an index, its area and a point inside.
  Check the areas against what you expect.
- `dof`: degrees of freedom left. Lower means more fully defined; a sketch with
  dof > 0 is still valid geometry.

Lines that are dimensioned but are not material — centre lines, axes of
symmetry, construction circles — are entities with construction: true;
constrain the drawn curves against them.

A part described as pieces -- bars of a given width along centre lines, rings,
overlapping outlines -- can be built two ways; neither needs the combined outline
computed by hand:
- in one sketch: draw each piece whole, overlapping the others freely. Curves
  that cross split the sketch into regions; the reply lists each with its area
  and a point inside, and make_face on the regions that are material gives one
  face (touching regions are merged). Holes are the regions left out.
- piece by piece: each piece its own face or solid, then fuse, cut and common
  combine them. Every reply carries the result's area, volume and face count.

Solids come from sketches or faces: extrude, extrude_cut (a pocket in a solid),
extrude_fuse (a boss on a solid), revolve; or from primitives (box, cylinder,
sphere, cone, torus). Every feature reply carries its measure (volume, area,
face count, valid).
{verify}
When done, call finish_stage(result_names) with the published names that
matter (sketches, faces or solids). It is refused while a sketch it uses has
loose ends. If you abandon an attempt, call discard on what you built for it.

Tool calls are cheap: each one answers in a second or two, anything you build
can be removed (sketch_remove, discard), and nothing is final until
finish_stage. So do not work out the whole part before your first call: as soon
as you have enough for one call, make it. When you are unsure how something
will come out -- where curves cross, which regions form, whether one piece
reaches another -- build it and read the reply instead of working it out in
your reasoning.

The rhythm, on a different part (a flange: disc of diameter 60, centre hole of
diameter 20, six holes of diameter 6 on a pitch circle of diameter 44, the
first at 30 degrees). Each turn is one or two sentences, then calls; the trig
is never done in the reasoning:
1. "The bolt-hole centres need trig; everything else is given. Start the
   sketch and get the centres in the same turn."
   create_sketch(plane="XOY") -> sketch_1, regions []
   calc("[(round(22*cos(radians(a)), 4), round(22*sin(radians(a)), 4))
   for a in range(30, 360, 60)]") -> [(19.0526, 11.0), (0.0, 22.0),
   (-19.0526, 11.0), (-19.0526, -11.0), (-0.0, -22.0), (19.0526, -11.0)]
2. "Centres in hand; draw all eight circles."
   sketch_add(sketch_1, [D: circle (0,0) r30, C: circle (0,0) r10,
   B1..B6: circle at each centre r3]) -> dof 24, loose_ends [], regions:
   0 area 2343.6281 inside (0, 15); 1 area 314.1593 inside (0, 0);
   2..7 area 28.2743 inside each bolt centre
3. "Material is region 0, the only point outside every hole. Check it."
   calc("pi*30**2 - pi*10**2 - 6*pi*3**2") -> 2343.6281
4. "Matches." close_sketch(sketch_1); make_face(sketch_1, [0]) -> sh_face_1,
   area 2343.6281, faces 1
5. finish_stage(["sh_face_1"])

Do only what the request asks. Each turn: state your reasoning in at most 2-3
sentences, then call one or more tools. Batch INDEPENDENT calls in one turn;
a call that needs a name another call is about to create waits for the next
turn.

If your first message includes a "Reference image spec" block, an image-analysis
agent read it from the user's reference image(s) (you do not see the images): the
exact data of the part(s) this stage builds, as JSON. A "profile" base is an
outline traced from `start` through `segments` (a line to `to`, or an arc with
`center` or `through`; a `fillet` is a tangent arc of that radius at the
segment's `to` corner) plus `holes`: draw it in one sketch as lines and arcs at
those coordinates (ends drawn at the same point are joined), then make_face on the material region and extrude or revolve it as the base says.
"placement" gives where the part sits and which way it faces. Values written
"~N" are estimates; plain numbers were read from the image. "conflicts" name
values that disagree: build anyway and let the tool replies tell you which value
fits. If the spec's prose and its data disagree, the data wins. Explicit
dimensions in the stage's params win over the spec.

You may be given only a subset of the SHAPER tools. If you need one that is not
in your list, call load_tools with its name or category."""

# policy.verify_cadence: how hard the specialist is pushed to check placement.
_VERIFY_HINT = {
    "none": "",
    "final": (
        "\nEvery feature reply reports the result's bbox. Glance at it each time; "
        "call measure on your result(s) once before finish_stage, and on any "
        "earlier object whose position you need but did not build yourself.\n"),
    "strict": (
        "\nEvery feature reply reports the result's bbox. After each fuse, cut, "
        "common, extrude_cut or extrude_fuse, read it (or call measure) and confirm "
        "the result sits and extends where you intended before continuing; fix a "
        "misplaced piece now, not at the end. Call measure on any earlier object "
        "whose position you need but did not build yourself.\n"),
}


def _shaper_prompt(verify_cadence: str = "final") -> str:
    return (SHAPER_SYSTEM_PROMPT.format(
        verify=_VERIFY_HINT.get(verify_cadence, _VERIFY_HINT["final"]))
            + "\n\n" + CALC_PROMPT)

# Planner-facing one-liners: what a shaper stage can do, not how to call it.
PLANNER_TOOL_LINES = {
    "create_sketch": "new sketch on XOY / XOZ / YOZ, a plane offset from one, a construction plane or a flat face of a solid",
    "sketch_add": "lines, circles, arcs, points, ellipses, splines, rectangles, polylines, projected edges (construction too)",
    "sketch_constrain": "dimensions and relations: length, radius, distance, angle, coincident, tangent, ...",
    "sketch_remove": "remove entities or constraints from a sketch",
    "sketch_status": "solved coordinates, dof, open ends and closed regions of a sketch",
    "close_sketch": "finish editing a sketch, before anything else is built",
    "make_face": "flat face from closed regions of a sketch, holes left out; a face can be the final result",
    "extrude": "solid from a face or sketch",
    "extrude_cut": "pocket: extrude and cut from a solid",
    "extrude_fuse": "boss: extrude and fuse onto a solid",
    "revolve": "solid of revolution from a face or sketch",
    "fuse": "fuse faces or solids into one",
    "cut": "cut faces or solids from faces or solids",
    "common": "keep only where faces or solids overlap",
    "measure": "area, volume, face count, validity of shaper objects",
}
CORE_TOOLS = frozenset(PLANNER_TOOL_LINES)
PLANNER_TOOL_LINES.update({
    "make_box": "box with a corner at the origin, from dx, dy, dz",
    "make_box_two_pnt": "box between two opposite corners",
    "make_cylinder": "cylinder from radius and height, placed at an origin along an axis",
    "make_sphere": "sphere from radius and centre",
    "make_cone": "cone or truncated cone from two radii and a height",
    "make_torus": "torus from ring and tube radii",
    "make_point": "construction point at x, y, z",
    "make_axis": "construction axis through two points or a point and a direction",
    "make_plane": "construction plane from a point and a normal, or offset from a base plane",
    "check_shape": "is a shape valid geometry",
    "list_study_objects": "every sketch and feature built so far, used-up ones marked",
    "min_distance": "shortest distance between two shapes",
    "basic_properties": "length, area, volume of a shape",
    "set_color": "colour of a shape",
    "set_transparency": "transparency of a shape",
    "translate": "move a shape by dx, dy, dz",
    "rotate": "rotate a shape about an axis",
    "mirror": "mirror a shape across a plane, axis or point, optionally keeping the original",
    "scale": "scale a shape, uniformly or per axis",
    "multi_translate": "row or grid of copies of a shape",
    "multi_rotate": "circular pattern of copies of a shape",
    "offset_shape": "grow or shrink a shape by offsetting its faces",
    "position": "place a shape built at the origin into a target frame",
    "align": "move a shape against, on top of or centred on another",
    "get_subshapes": "ids of a shape's edges, faces, vertices or solids, filtered by position, size or plane",
    "get_subshape": "new shape copied from some faces or edges of a shape",
    "explode": "every edge, face, ... of a shape as its own shape",
    "fillet_edges": "round edges of a solid",
    "chamfer_edges": "bevel edges of a solid",
    "make_thick_solid": "hollow a solid into a wall, open at chosen faces",
    "partition": "cut shapes by each other, keeping every piece with shared faces",
    "split": "split shapes into pieces with other shapes or planes",
    "make_polyline": "3D wire of straight segments through points",
    "make_curve": "smooth 3D curve through points",
    "make_pipe": "sweep a face or wire along a path",
    "make_loft": "solid through two sections",
    "make_filling": "surface spanning boundary curves",
    "sketch_fillet": "round a corner of a sketch",
    "sketch_copy": "mirror, offset or pattern curves of a sketch",
})

_CATALOGUE = ("SHAPER tools available to a single SHAPER stage:\n"
              + "\n".join(f"  {n} -- {d}" for n, d in PLANNER_TOOL_LINES.items()))


def _finish(args: dict, call_log: list) -> dict:
    names = args.get("result_names") or []
    if isinstance(names, str):
        names = [names]
    return {"result_names": list(names), "call_log": call_log}


def _gate(args: dict, call_log: list, infos: dict) -> str | None:
    names = args.get("result_names") or []
    if isinstance(names, str):
        names = [names]
    used = _reachable(call_log, roots_for(call_log, list(names)))
    lines = []
    for name in sorted(used):
        ends = (infos.get(name) or {}).get("loose_ends")
        if ends:
            refs = ", ".join(f"{e['ref']} at {e['at']}" for e in ends)
            lines.append(f"- {name} has open ends: {refs}")
    if not lines:
        return None
    return ("Not finished: a sketch you use has open ends.\n" + "\n".join(lines)
            + "\nJoin these ends (coincident constraints, or the missing curve), or "
            "remove the stray curves, then call finish_stage again.")


def _upstream_line(name: str, info) -> str:
    if not isinstance(info, dict):
        return f"  {name}"
    if "regions" in info:
        return f"  {name} — a sketch, {len(info['regions'])} regions"
    n = info.get("faces", 0)
    line = (f"  {name} — bbox {info.get('bbox')}, area {info.get('area')}, "
            f"{n} face{'' if n == 1 else 's'}")
    if info.get("volume"):
        line += f", volume {info['volume']}"
    if info.get("valid") is False:
        line += ", NOT valid"
    return line


def _upstream(client, names: list[str]) -> str | None:
    """Without it a later SHAPER stage never learns what earlier ones built,
    and redraws them instead of fusing them."""
    if not names:
        return None
    info = {}
    try:
        m = client.call_tool("measure", {"names": list(names)}, module="shaper")
        if m.get("ok"):
            info = m.get("measurements") or {}
        else:
            log.warning("  measure(upstream) failed: %s", m.get("error"))
    except Exception as e:  # noqa: BLE001 — best-effort, must not abort the stage
        log.warning("  measure(upstream) errored: %s", e)
    # A result a later boolean used up is hidden in SHAPER: naming it fails.
    names = [n for n in names if not (info.get(n) or {}).get("concealed")]
    if not names:
        return None
    return ("Objects built by an EARLIER stage — use them by these exact names "
            "(fuse, cut, measure); do not rebuild them. Their numbers were measured "
            "by SHAPER and are exact: take them as given, do not recompute them:\n"
            + "\n".join(_upstream_line(n, info.get(n)) for n in names))


def _retry_block(retry: dict) -> str:
    lines = []
    err = retry.get("last_error")
    if err:
        lines.append(f"Your previous attempt at this stage failed: {err.get('message')}")
    if retry.get("hint"):
        lines.append(f"User hint for this retry: {retry['hint']}")
    built = retry.get("built") or []
    if built:
        info = retry.get("built_info") or {}
        lines.append("Objects your previous attempt already built and published — reuse "
                     "them if they are still correct, or rebuild only the ones the error "
                     "was actually in:")
        lines += [_upstream_line(n, info.get(n)) for n in built]
    return "\n".join(lines)


def _audit(args, calls, call_log, upstream, infos):
    from executor.shaper import registry
    from salome_agent.specialists.geom import _audit_finish
    return _audit_finish(args, calls, call_log, upstream, infos,
                         categories=registry.categories(), info_line=_upstream_line)


def _render(result: dict) -> str:
    hidden = {"records", "result_shape", "shape_info"}
    return str({k: v for k, v in result.items() if k not in hidden})


SHAPER_SPEC = register(ModuleSpec(
    name="shaper",
    prompt=_shaper_prompt,
    terminator=Terminator(name="finish_stage", schema=FINISH_STAGE_SCHEMA,
                          description=FINISH_STAGE_DESCRIPTION, handle=_finish),
    execute=lambda client, name, args: client.call_tool(name, args, module="shaper"),
    render_result=_render,
    render_upstream=_upstream,
    render_retry=_retry_block,
    audit=_audit,
    tools_for=lambda client: {t["name"]: t
                              for t in client.list_tools(module="shaper")["tools"]},
    needs_session=True,
    kickoff=("Start with the first step now. Each tool result tells you what you "
             "need for the next one."),
    no_tool_nudge=("You did not call a tool. Call at least one tool now — "
                   "finish_stage once the geometry is done."),
    on_exhausted=lambda call_log, last_error: {"call_log": call_log,
                                               "last_error": last_error},
    gate=_gate,
    core_tools=CORE_TOOLS,
    categories=("primitive", "sketch", "swept", "boolean",
                "transform", "feature", "display", "query"),
    risky_categories=frozenset({"boolean", "transform"}),
    measure_tools=frozenset({"measure"}),
    forgets=True,
    checks_results=True,
    calc=True,
    call_prefix="model.",
    preamble=PREAMBLE,
    postamble=POSTAMBLE,
    publish_line=PUBLISH_LINE,
    catalogue=_CATALOGUE,
))
