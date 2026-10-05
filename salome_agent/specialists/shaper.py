from executor.shaper.state import POSTAMBLE, PREAMBLE, PUBLISH_LINE
from salome_agent.calc import CALC_PROMPT
from salome_agent.logs import log
from salome_agent.codegen import _reachable, roots_for
from salome_agent.modules import ModuleSpec, Terminator, register
from salome_agent.specialists.geom import FINISH_STAGE_DESCRIPTION, FINISH_STAGE_SCHEMA

SHAPER_SYSTEM_PROMPT = """You build parametric geometry in SHAPER by calling
tools. Lengths are millimetres, unconverted; angles are in
degrees. Reference sketches and features by the name they were published under.

A sketch is entities (lines, circles, arcs, points) tied together by
constraints; the solver computes the exact geometry. Build it in chunks:
create_sketch, then for one logical piece at a time sketch_add its entities at
roughly the right coordinates and sketch_constrain them with the given
dimensions and relations (coincident ends, horizontal, tangent, length, radius,
distance, angle). Anchor the geometry to ORIGIN and the plane's axes.

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
extrude_fuse (a boss on a solid), revolve. Every feature reply carries its
measure (volume, area, face count, valid).

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
4. "Matches." make_face(sketch_1, [0]) -> sh_face_1, area 2343.6281, faces 1
5. finish_stage(["sh_face_1"])

Do only what the request asks. Each turn: state your reasoning in at most 2-3
sentences, then call one or more tools. Batch INDEPENDENT calls in one turn;
a call that needs a name another call is about to create waits for the next
turn."""

# Planner-facing one-liners: what a shaper stage can do, not how to call it.
PLANNER_TOOL_LINES = {
    "create_sketch": "new sketch on XOY / XOZ / YOZ or a plane offset from one",
    "sketch_add": "lines, circles, arcs, points (construction lines too)",
    "sketch_constrain": "dimensions and relations: length, radius, distance, angle, coincident, tangent, ...",
    "sketch_remove": "remove entities or constraints from a sketch",
    "sketch_status": "solved coordinates, dof, open ends and closed regions of a sketch",
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
    return ("Not finished: a sketch you use is not closed.\n" + "\n".join(lines)
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
            "(fuse, cut, measure); do not rebuild them:\n"
            + "\n".join(_upstream_line(n, info.get(n)) for n in names))


def _render(result: dict) -> str:
    hidden = {"records", "result_shape", "shape_info"}
    return str({k: v for k, v in result.items() if k not in hidden})


SHAPER_SPEC = register(ModuleSpec(
    name="shaper",
    prompt=lambda cadence: SHAPER_SYSTEM_PROMPT + "\n\n" + CALC_PROMPT,
    terminator=Terminator(name="finish_stage", schema=FINISH_STAGE_SCHEMA,
                          description=FINISH_STAGE_DESCRIPTION, handle=_finish),
    execute=lambda client, name, args: client.call_tool(name, args, module="shaper"),
    render_result=_render,
    render_upstream=_upstream,
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
    forgets=True,
    checks_results=True,
    calc=True,
    call_prefix="model.",
    preamble=PREAMBLE,
    postamble=POSTAMBLE,
    publish_line=PUBLISH_LINE,
    catalogue=_CATALOGUE,
))
