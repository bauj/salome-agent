import json
import re

from salome_agent.llm import RepetitionLoopDetected
from salome_agent.logs import log
from salome_agent.modules import spec_for
from salome_agent.nodes.vision_spec import compact_spec, part_ids
from salome_agent.repetition import loop_nudge


_PLANNER_BASE = """You turn a natural-language SALOME request into an
ordered list of stages. Each stage has: module (geom|shaper|smesh), intent
(short description of the job for that stage), params (dims in millimetres,
unconverted), inputs (a plain-words list of what this stage needs from earlier ones), and
-- for a GEOM stage -- an optional "tools" list naming the GEOM tool names you expect
that stage to use (from the GEOM catalogue below; names only, e.g.
["make_box", "translate", "set_color"]). The "tools" list is a hint that
narrows what the GEOM specialist is offered; omit it if unsure. Respond as
JSON: {"stages": [...]}. Output only the JSON object -- no analysis, no
prose before or after it.

In a GEOM stage, a flat part whose outline has rounded corners, holes or
slots (a plate, a bracket, a gasket) is one make_profile_face (outline and
holes in one call) extruded with make_prism: hint those two, not the make_box +
fillet_edges + cut route. Add other tools only for work the outline cannot do,
such as rounding the top edges. A SHAPER stage draws such a part in a sketch.

Keep your reasoning brief -- a few sentences at most. Do not enumerate every
possible interpretation or walk through each stage in detail; decide and emit
the plan. You pick each stage's module and job; how the part is drawn is the
stage's work, not yours.

Object names are assigned automatically and you cannot know them. Never put
a guessed object name in `inputs` -- describe what the stage needs. Every
object built by an earlier stage is passed to each later stage regardless.

A **part** is one primitive plus only the operations that shape that same
piece of material -- a boolean that cuts or rounds it, its transform, its
colour. An **assembly** is several parts combined (fused, or just placed
together).

**Never split one part's construction across stages** -- all of its
operations belong to one stage; splitting breaks the object-name handoff.

For an assembly you MAY give each part (or each group of identical parts)
its own stage. Do that when the parts are distinct enough to name. A car is
an assembly: body, cabin, wheels are separate parts. Also use separate
stages whenever the work crosses modules (GEOM -> SMESH) or the
request asks for genuinely independent deliverables.

Fuse parts into one solid only when they are one piece of material (a mug
and its handle) or the request asks for a single solid. Never fuse parts
that should keep different colours (a red body, black wheels): an object
shows one colour, so the fuse loses them. Leave such parts as separate
objects.

Ask only for what the user asked for: no colour, transparency or extra
features they did not mention. Do not make a vague position or size precise
("near one end", "a small hole"): keep the user's words in params and let the
specialist pick the value. Any value you must fill in yourself must fit the
others -- a fillet radius below the wall thickness it rounds, no feature
overlapping another.

Module scope:
- shaper (SHAPER): all CAD geometry -- primitives, booleans, transforms,
  fillet/chamfer, sketches with or without constraints, faces and solids built
  from them (extrude, pocket, boss, revolve), AND display properties of
  geometry (colour, transparency). This is the default CAD module. A SHAPER
  stage builds the whole part on its own, a flat face included (make_face):
  never add a GEOM stage to finish a SHAPER part.
- geom (GEOM): the same CAD geometry, used only when the request names GEOM.
- smesh: meshing a geometry (algorithms, hypotheses, compute).

A module the request names ("in SHAPER", "using GEOM", "with SMESH")
always wins over these rules.

If the request comes with a "Reference image spec" block, an image-analysis
agent read it from the user's reference image(s): one entry per part, each with
an id (P1, P2, ...). Shape your stages around those parts and give every stage
that builds one of them a "parts" list of their ids, e.g. "parts": ["P1"]; that
stage is handed the part's exact data. Never restate a part's numbers in intent
or params -- refer to it by id. Several views described together are ONE object,
not separate parts. Explicit numeric dimensions in the text request still win:
put those in params. Mention listed uncertainties or conflicts in a stage's
params rather than inventing detail."""

# The smesh one-liner stays in _PLANNER_BASE (not restructured); it also
# appears in its spec.catalogue string so routable() includes it.
# Do NOT move that line out — the prompt's byte-identity depends on it.
PLANNER_SYSTEM_PROMPT = (_PLANNER_BASE + "\n\n" + spec_for("geom").catalogue
                         + "\n\n" + spec_for("shaper").catalogue)

# A CAD module the request names is pinned: the planner is told, and any CAD
# stage it still routes elsewhere is moved there.
_CAD = ("geom", "shaper")
_NAMED_MODULE = re.compile(
    r"\b(?:in|using|with|via)\s+(?:the\s+)?(geom|shaper)\b|\b(geom|shaper)\s+module\b",
    re.IGNORECASE)


def named_cad_module(request: str) -> str | None:
    named = {(a or b).lower() for a, b in _NAMED_MODULE.findall(request)}
    return named.pop() if len(named) == 1 else None

_DETAIL_HINT = {
    "coarse": "\n\nPrefer the fewest stages: keep an assembly's parts in one "
              "stage unless they are truly independent deliverables.",
    "balanced": "",
    "fine": "\n\nFor any assembly, give EACH part (or group of identical "
            "parts) its own stage, plus a final stage that fuses them when "
            "they are to be fused. Being fused at the end does not make the "
            "parts one part. Still never split a single part's own construction.",
}


# Told only when the request names SHAPER: GEOM keeps "never split one part".
_SHAPER_SPLIT = """

When the request describes one SHAPER part as several distinct pieces (a ring
and bars, a plate and its ribs), build it in stages: one stage per piece or
group of alike pieces, then a final stage that combines them (fuse, cut or
common) into the one result. For SHAPER this replaces "never split one part's
construction": a later SHAPER stage is given the earlier stages' results by
name. Each piece stage leaves its piece as its own result, fused to nothing;
only the final stage fuses, cuts or commons the pieces together. A piece stage
may look at an earlier piece to find where it stops, but does not fuse onto it.
Write each piece stage's intent as a complete brief: the piece's own
measurements, where it may run into another piece (bar ends may overlap into
the ring), where it must stop (never past the outer circle), and which pieces
it does not build. A cut-out that belongs to one piece goes in that piece's
stage. A part with one outline, such as a plate with holes, stays one stage."""


def blank_stages(stage_plans: list[dict]) -> dict:
    return {
        i: {"module": p["module"], "status": "pending", "script": None,
            "call_log": None, "static_retries": 0, "exec_retries": 0}
        for i, p in enumerate(stage_plans)
    }


def plan(request: str, llm, detail: str = "balanced",
         image_spec: dict | None = None, pin_from: str | None = None) -> list[dict]:
    """`pin_from` is the user's own text when `request` also carries earlier
    requests, which may name another module."""
    system = PLANNER_SYSTEM_PROMPT + _DETAIL_HINT.get(detail, "")
    pinned = named_cad_module(request if pin_from is None else pin_from)
    if pinned:
        system += (f"\n\nThe request names {pinned.upper()}: every CAD geometry stage "
                   f'uses module "{pinned}".')
        if pinned == "shaper":
            system += _SHAPER_SPLIT
    if image_spec:
        request = f"{request}\n\n{compact_spec(image_spec)}"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": request}]
    stages = _ask(llm, messages)
    if image_spec:
        stages = _check_parts(stages, image_spec, llm, messages)
    if pinned:
        stages = _pin(stages, pinned)
    return stages


def _pin(stages: list[dict], module: str) -> list[dict]:
    for i, s in enumerate(stages, 1):
        if s.get("module") in _CAD and s["module"] != module:
            log.warning("planner routed stage %d to %s; the request names %s",
                        i, s["module"], module)
            s["module"] = module
            s.pop("tools", None)
    return stages


def _ask(llm, messages: list[dict]) -> list[dict]:
    """A planner call that loops is re-asked without the loop, up to
    llm.loop_retries times; past that it propagates (a clean fatal in graph.py)."""
    nudge, loops = [], 0
    while True:
        try:
            return llm.chat(messages=messages + nudge, module="planner")["stages"]
        except RepetitionLoopDetected as e:
            loops += 1
            if loops > llm.loop_retries:
                raise
            log.warning("planner: %s; nudging it to answer", e)
            nudge = [{"role": "user", "content": loop_nudge(
                e.sample, e.before, "answer with the JSON plan now.")}]


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
        stages = _ask(llm, messages)
    for s in stages:
        if "parts" in s:
            s["parts"] = [i for i in part_ids(s) if i in known]
    return stages
