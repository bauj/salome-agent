import json

from salome_agent.logs import log
from salome_agent.modules import spec_for
from salome_agent.nodes.vision_spec import compact_spec, part_ids


_PLANNER_BASE = """You turn a natural-language SALOME request into an
ordered list of stages. Each stage has: module (geom|shaper|smesh), intent
(short description of the job for that stage), params (dims in millimetres,
unconverted), inputs (a plain-words list of what this stage needs from earlier ones), and
-- for a geom stage -- an optional "tools" list naming the geom tool names you expect
that stage to use (from the catalogue below; names only, e.g.
["make_box", "translate", "set_color"]). The "tools" list is a hint that
narrows what the geom specialist is offered; omit it if unsure. Respond as
JSON: {"stages": [...]}. Output only the JSON object -- no analysis, no
prose before or after it.

A flat part whose outline has rounded corners, holes or slots (a plate, a
bracket, a gasket) is one make_profile_face (outline and holes in one call)
extruded with make_prism: hint those two, not the make_box + fillet_edges +
cut route. Add other tools only for work the outline cannot do, such as
rounding the top edges.

Keep your reasoning brief -- a few sentences at most. Do not enumerate every
possible interpretation or walk through each stage in detail; decide and emit
the plan.

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
stages whenever the work crosses modules (geom -> smesh) or the
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
- geom: all CAD geometry -- primitives, booleans, transforms, fillet/chamfer,
  AND display properties of geometry (colour, transparency, display mode).
  "make it red", "colour the shape blue", "50% transparent" are geom.
- shaper: parametric sketches with constraints, and faces and solids built from them (extrude, pocket, boss, revolve).
  Use shaper only when the request names it or asks for a constrained or
  parametric sketch; all other CAD geometry stays geom.
- smesh: meshing a geometry (algorithms, hypotheses, compute).

A module the request names ("in shaper", "using GEOM", "with SMESH")
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
PLANNER_SYSTEM_PROMPT = _PLANNER_BASE + "\n\n" + spec_for("geom").catalogue

_DETAIL_HINT = {
    "coarse": "\n\nPrefer the fewest stages: keep an assembly's parts in one "
              "stage unless they are truly independent deliverables.",
    "balanced": "",
    "fine": "\n\nFor any assembly, give EACH part (or group of identical "
            "parts) its own stage, plus a final stage that fuses them when "
            "they are to be fused. Being fused at the end does not make the "
            "parts one part. Still never split a single part's own construction.",
}


def blank_stages(stage_plans: list[dict]) -> dict:
    return {
        i: {"module": p["module"], "status": "pending", "script": None,
            "call_log": None, "static_retries": 0, "exec_retries": 0}
        for i, p in enumerate(stage_plans)
    }


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
