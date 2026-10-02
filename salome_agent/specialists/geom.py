import json

from executor.geom import TOOL_DESCRIPTIONS, registry
from salome_agent.codegen import _refs_in
from salome_agent.logs import log
from salome_agent.modules import ModuleSpec, Terminator, register
from salome_agent.specialists.tool_selection import CORE_TOOLS
from salome_agent.specialists.tool_specialist import ToolSpecialist

_GEOM_BASE = """You build geometry by calling tools, not by writing
geompy source. Each tool carries its own description — read it, especially
for the default placement and orientation of primitives. Reference any
shape, point, vector or plane by the name it was published under. For an
axis you may pass "OX", "OY" or "OZ" instead of building a vector. Lengths
are millimetres, unconverted; angles are in degrees.

When done, call finish_stage(result_names) naming which of your published
objects matter — everything they transitively depend on is kept, everything
else is removed from the study.

If you abandon an attempt — a part you are going to rebuild differently —
call discard on the shapes you built for it before you move on.

Do only what the request asks. Do not invent colours, transparency or
display modes that were not requested.

If a requested value or step order cannot work geometrically (a fillet
wider than the wall it sits on, a hole larger than its part), make the
smallest change that does work and say what you changed when you finish.
When a position or size is vague ("near one end", "roughly 10 mm") or the
given value would make features overlap, choose one clean value that fits and
build with it; do not weigh different readings of the wording.
When a tool error names a value to use, use exactly that value; do not
work out whether a larger one would also fit. Pick one fix
and try it; do not weigh a list of alternatives. Plan only your next step
or two, not the whole stage: later steps get decided with their results in
hand.

If a tool result carries a `delta` note, it flagged something numerically
wrong with that boolean or transform (a cut that removed nothing, a fuse of
parts that never touched, a rigid move that changed volume) — fix the cause
before you continue. If a call comes back saying an identical call already
failed, do not retry it unchanged: change the arguments or gather
information first.

Each turn: state your reasoning in at most 2-3 sentences, then call one or
more tools. Do not reply without calling a tool. You may batch several
INDEPENDENT calls in one turn (e.g. the four walls of a frame, or four
translates of shapes that already exist). Keep DEPENDENT steps in separate
turns -- if one call needs the name of a shape another call is about to
create, it must wait for the next turn.
{verify}
To place one part relative to another -- on top of, butted against a face,
flush with an edge, centred on it -- use `align` rather than measuring both
shapes and working out the translation yourself. `align` reads both bounding
boxes and moves the part for you.

Every transform (translate, rotate, ...) produces a NEW shape that does not
inherit the source's colour. So apply display properties (set_color,
set_transparency, set_display_mode) LAST, once, on the final shape — after
every primitive, boolean and transform.

To act on specific edges or faces (e.g. "fillet the top four edges"), first
call get_subshapes to get their integer ids -- by where = "all", a centre
axis (sorted_by / take / count), size (by_size / count), or a plane
(on_plane / state) -- then pass that id list straight to fillet_edges,
chamfer_edges or get_subshape. fillet_edges / chamfer_edges also accept the
same where predicate directly instead of an id list. get_subshapes only
reports ids and centres;
it changes no geometry. Use explode when you instead want every subshape
published as its own named object.

You are given only a subset of the geom tools, chosen for this stage. If you
need a tool that is not in your list, call load_tools with specific tool
`names` or whole `categories` (primitive, sketch, swept, boolean, transform,
feature, display, query) to add them, then call the tool you wanted.

If your first message includes a "Description of the user's reference image(s)"
block, treat it as the visual target for the geometry you build: match the
proportions, shape and relative placement it gives. You do not see the images
themselves, only that description. Use explicit numeric dimensions from the
request/params first, then dimensions the description says were read from the
image; values marked with "~" are estimates. If a stage-level dimension is
missing, estimate it from the description rather than inventing detail."""

# policy.verify_cadence — how hard the specialist is pushed to self-check
# shape placement. shape_info rides on every ok result for free; the question
# is how often to spend an explicit `measure` round-trip.
_VERIFY_HINT = {
    "none": "",
    "final": (
        "\nEvery successful tool result reports the shape's bbox, size and "
        "centre (mm) in `shape_info` — glance at it each time. Call `measure` "
        "explicitly only when a position looks wrong, and on your result "
        "shape(s) once before finish_stage. Call `measure` on any upstream "
        "object whose position you need but did not build yourself.\n"
    ),
    "strict": (
        "\nEvery successful tool result reports the shape's bbox, size and "
        "centre (mm) in `shape_info`. After each translate, rotate, fuse or "
        "cut, read that `shape_info` (or call `measure`) and confirm the "
        "shape's centre and extent match what you intended before continuing "
        "— if a part is in the wrong place, correct it now, not at the end. "
        "Call `measure` on any upstream object whose position you need but "
        "did not build yourself.\n"
    ),
}

# Categories whose output can land in the wrong place — the ones a measure
# self-check is actually about.
_PLACEMENT_RISKY = {"boolean", "transform"}
_MEASURE_TOOLS = {"measure", "check_shape"}
# strict: nudge once every N risky ops with no intervening measure.
_STRICT_NUDGE_EVERY = 3

# The result.py preamble a geom stage needs. Assembler owns when to emit it;
# the module owns what it is.
_GEOM_PREAMBLE = """from salome.geom import geomBuilder
geompy = geomBuilder.New()
"""


def _geom_prompt(verify_cadence: str = "final") -> str:
    return _GEOM_BASE.format(verify=_VERIFY_HINT.get(verify_cadence, _VERIFY_HINT["final"]))


GEOM_SYSTEM_PROMPT = _geom_prompt("final")

FINISH_STAGE_SCHEMA = {
    "type": "object",
    "properties": {"result_names": {"type": "array", "items": {"type": "string"}}},
    "required": ["result_names"],
}
FINISH_STAGE_DESCRIPTION = (
    "Finish this stage. `result_names` = the published names that matter; "
    "their dependencies are kept, everything else is dropped."
)

# Boolean ops consume their operands into the result — an operand kept in
# result_names alongside the boolean's output is a superseded intermediate.
# Transforms (translate/rotate/align/...) are copy semantics: the source
# stays independently valid, so it is NOT dropped.
_BOOLEAN_FNS = {"MakeFuse", "MakeFuseList", "MakeCut", "MakeCutList",
                "MakeCommon", "MakeCommonList", "MakeBoolean", "MakePartition"}


def _leaf_result_names(call_log: list[dict], names: list[str]) -> list[str]:
    """Drop any name that a boolean op folded into another kept result. An
    over-split stage that finish_stage's a pre-boolean operand would otherwise
    hand the superseded shape to the next stage, which then builds on the
    wrong object (GH #9)."""
    names = list(dict.fromkeys(names))
    if len(names) < 2:
        return names
    kept = set(names)
    consumed: set[str] = set()
    for call in call_log:
        if call["fn"] in _BOOLEAN_FNS and kept & set(call["out_vars"]):
            for arg in call["args"]:
                consumed |= _refs_in(arg)
    leaves = [n for n in names if n not in consumed]
    return leaves or names


def _compact_result(result: dict) -> dict:
    """A prior tool result, stripped to what the model reads back on later
    turns. The verbose `records` blob (geompy dispatch detail — only codegen
    needs it, and codegen reads it straight from `call_log`) is dropped;
    linear-growth transcript cost was ~66% of the input on a real run."""
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error")}
    keep = {"ok": True}
    for k in ("result_shape", "shape_info", "delta", "colour", "measurements", "names", "sanity",
              "ids", "centres", "distance", "properties"):
        if k in result:
            keep[k] = result[k]
    published = [n for rec in result.get("records", []) for n in (rec.get("publish") or [])]
    if published:
        keep["published"] = published
    return keep


def run_geom_specialist(stage_plan: dict, llm, client, iteration_cap: int,
                        upstream: dict | None = None,
                        tool_selection: str = "hints",
                        verify_cadence: str = "final",
                        retry: dict | None = None,
                        seed_call_log: list[dict] | None = None,
                        seed_calls: list[dict] | None = None,
                        image_description: str | None = None) -> dict:
    """Kept as the geom entry point for the existing tests and callers; the
    loop itself is generic now."""
    blocks = []
    if upstream:
        blocks.append(_upstream_block(upstream))
    if image_description:
        blocks.append(image_block(image_description))
    return ToolSpecialist(GEOM_SPEC).run(
        stage_plan, llm, client, iteration_cap, blocks=tuple(blocks),
        tool_selection=tool_selection, verify_cadence=verify_cadence,
        retry=retry, seed_call_log=seed_call_log, seed_calls=seed_calls,
        upstream=tuple(upstream or ()))


def image_block(image_description: str) -> str:
    """The vision agent's description of the user's reference image(s), as a
    block of the geom stage's first message."""
    return ("Description of the user's reference image(s), written by an "
            "image-analysis agent (you do not see the images themselves). Use it "
            "for proportions and placement; the stage's own params win over it:\n"
            + image_description)


def _retry_block(retry: dict) -> str:
    lines = []
    err = retry.get("last_error")
    if err:
        lines.append(f"Your previous attempt at this stage failed: {err.get('message')}")
    if retry.get("hint"):
        lines.append(f"User hint for this retry: {retry['hint']}")
    built = retry.get("built") or []
    if built:
        lines.append("Shapes your previous attempt already built and published — reuse "
                     "them if they are still correct, or rebuild only the ones the error "
                     "was actually in:")
        built_info = retry.get("built_info") or {}
        for name in built:
            info = built_info.get(name)
            if info:
                lines.append(_info_line(name, info))
            else:
                lines.append(f"  {name}")
    return "\n".join(lines)


def _result_names(args: dict) -> list[str]:
    result_names = args["result_names"]
    # A schema hint doesn't guarantee compliance — seen live, a model returned
    # this as a JSON-encoded string, which codegen's set() would then iterate
    # as individual characters.
    if isinstance(result_names, str):
        result_names = json.loads(result_names)
    return result_names


def _finish_stage(args: dict, call_log: list) -> dict:
    return {"call_log": call_log,
            "result_names": _leaf_result_names(call_log, _result_names(args))}


# Tools whose output is a part the model meant to keep or build on. Sketch
# helpers, subshape extractions and queries are routinely left unused.
_SOLID_CATEGORIES = {"primitive", "swept", "boolean", "transform"}
_SOLID_TOOLS = {"fillet_edges", "chamfer_edges"}


def _audit_finish(args: dict, calls: list, call_log: list, upstream: frozenset,
                  infos: dict) -> str | None:
    """Anomalies only — None on a clean stage, so finish_stage closes at no
    extra cost. Built on what the loop already has: no server round-trip."""
    try:
        names = list(dict.fromkeys(_result_names(args)))
    except (KeyError, ValueError, TypeError):
        return None
    built = {v for r in call_log for v in r.get("out_vars") or []}
    refs = {r["recv"] for r in call_log if r.get("recv")}
    for r in call_log:
        for a in r.get("args") or []:
            refs |= _refs_in(a)
    unknown = [n for n in names if n not in built and n not in upstream]
    categories = registry.categories()
    orphans = []
    for i, c in enumerate(calls):
        if not (categories.get(c["tool"]) in _SOLID_CATEGORIES or c["tool"] in _SOLID_TOOLS):
            continue
        # A later call of the same tool is taken as a redo, the first attempt
        # left unused on purpose (both golden runs and a live cup run do this).
        # The price: a forgotten part is only caught when no later call used
        # its tool.
        if any(later["tool"] == c["tool"] for later in calls[i + 1:]):
            continue
        orphans += [v for r in c["records"] for v in r.get("out_vars") or []
                    if v not in refs and v not in names and v not in orphans]
    empty = [n for n in names
             if infos.get(n) and all(abs(d) < 1e-6 for d in infos[n]["size"])]
    if not (unknown or orphans or empty):
        return None
    lines = ["Before this stage closes, check:"]
    if unknown:
        lines.append(f"- not a shape built in this run: {', '.join(unknown)} — "
                     "use the exact published name.")
    if orphans:
        lines.append(f"- built but neither in result_names nor used by another shape: "
                     f"{', '.join(orphans)} — add any that belong to the result; the rest "
                     "is removed from the study when the stage closes.")
    if empty:
        lines.append(f"- empty or a single point (size 0 x 0 x 0): {', '.join(empty)}.")
    known = [n for n in names if infos.get(n)]
    if known:
        lines.append("Your result shapes:")
        lines += [_info_line(n, infos[n]) for n in known]
    lines.append("Fix what needs fixing, then call finish_stage again — unchanged if "
                 "it is already right.")
    return "\n".join(lines)


def _info_line(name: str, info: dict) -> str:
    return f"  {name} — bbox {info['bbox']}, size {info['size']}, centre {info['centre']}"


def _geom_tools_for(client) -> dict:
    return {t["name"]: t for t in client.list_tools()["tools"]}


def _upstream_block(info: dict) -> str:
    lines = []
    for name, detail in info.items():
        if detail:
            lines.append(_info_line(name, detail))
        else:
            lines.append(f"  {name} — (geometry unavailable)")
    return ("Objects built by an EARLIER stage, given only for reference "
            "(e.g. to measure or align against) — reference them by these exact "
            "names, but do not modify, colour or finish_stage them unless this "
            "stage's own task says to:\n" + "\n".join(lines))


def _geom_upstream(client, names: list[str]) -> str | None:
    """The measured upstream block. Measuring belongs here, not in graph.py:
    knowing what a bbox is is geom's business."""
    if not names or client is None:
        return None
    info = None
    try:
        m = client.call_tool("measure", {"names": names})
        if m.get("ok"):
            info = m.get("measurements")
        else:
            log.warning("  measure(upstream) failed: %s", m.get("error"))
    except Exception as e:  # noqa: BLE001 — best-effort, must not abort the stage
        log.warning("  measure(upstream) errored: %s", e)
    if not info:
        return None
    return _upstream_block({n: info.get(n) for n in names})


def _geom_catalogue_str() -> str:
    # Names + one-liners only (no arg schemas) so the planner knows what a
    # single geom stage can actually do and stops over-splitting parts.
    lines = [f"  {n} -- {d}" for n, d in TOOL_DESCRIPTIONS.items()]
    return "geom tools available to a single geom stage:\n" + "\n".join(lines)


GEOM_SPEC = register(ModuleSpec(
    name="geom",
    prompt=_geom_prompt,
    terminator=Terminator(name="finish_stage", schema=FINISH_STAGE_SCHEMA,
                          description=FINISH_STAGE_DESCRIPTION, handle=_finish_stage),
    execute=lambda client, name, args: client.call_tool(name, args),
    render_result=lambda result: str(_compact_result(result)),
    tools_for=_geom_tools_for,
    kickoff=("Start with the first step now. Do not work out the later steps "
             "yet: each tool result tells you what you need for the next one, "
             "and the tool descriptions say how each tool behaves."),
    no_tool_nudge=("You did not call a tool. Call at least one tool now — "
                   "finish_stage once the geometry is done."),
    on_exhausted=lambda call_log, last_error: {"call_log": call_log, "last_error": last_error},
    core_tools=frozenset(CORE_TOOLS),
    categories=("primitive", "sketch", "swept", "boolean",
               "transform", "feature", "display", "query"),
    risky_categories=frozenset(_PLACEMENT_RISKY),
    measure_tools=frozenset(_MEASURE_TOOLS),
    render_upstream=_geom_upstream,
    render_retry=lambda retry: _retry_block(retry),
    audit=_audit_finish,
    forgets=True,
    call_prefix="geompy.",
    preamble=_GEOM_PREAMBLE,
    catalogue=_geom_catalogue_str(),
))
