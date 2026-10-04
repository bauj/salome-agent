from executor.shaper.state import POSTAMBLE, PREAMBLE, PUBLISH_LINE
from salome_agent.codegen import _reachable, roots_for
from salome_agent.modules import ModuleSpec, Terminator, register
from salome_agent.specialists.geom import FINISH_STAGE_DESCRIPTION, FINISH_STAGE_SCHEMA

SHAPER_SYSTEM_PROMPT = """You build parametric geometry in SHAPER by calling
tools, not by writing code. Lengths are millimetres, unconverted; angles are in
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
constrain the drawn curves against them. Curves that cross split the sketch into
regions; make_face takes the region indices you want.

Solids come from sketches or faces: extrude, extrude_cut (a pocket in a solid),
extrude_fuse (a boss on a solid), revolve. Every feature reply carries its
measure (volume, area, face count, valid).

When done, call finish_stage(result_names) with the published names that
matter (sketches, faces or solids). It is refused while a sketch it uses has
loose ends. If you abandon an attempt, call discard on what you built for it.

Do only what the request asks. Each turn: state your reasoning in at most 2-3
sentences, then call one or more tools. Batch INDEPENDENT calls in one turn;
a call that needs a name another call is about to create waits for the next
turn."""

_CATALOGUE = ("- shaper: parametric sketches with constraints, and faces and solids "
              "built from them (extrude, pocket, boss, revolve).")


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


def _render(result: dict) -> str:
    hidden = {"records", "result_shape", "shape_info"}
    return str({k: v for k, v in result.items() if k not in hidden})


SHAPER_SPEC = register(ModuleSpec(
    name="shaper",
    prompt=lambda cadence: SHAPER_SYSTEM_PROMPT,
    terminator=Terminator(name="finish_stage", schema=FINISH_STAGE_SCHEMA,
                          description=FINISH_STAGE_DESCRIPTION, handle=_finish),
    execute=lambda client, name, args: client.call_tool(name, args, module="shaper"),
    render_result=_render,
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
    call_prefix="model.",
    preamble=PREAMBLE,
    postamble=POSTAMBLE,
    publish_line=PUBLISH_LINE,
    catalogue=_CATALOGUE,
))
