from salome_agent.modules import spec_for


_PLANNER_BASE = """You turn a natural-language SALOME request into an
ordered list of stages. Each stage has: module (geom|smesh|paravis), intent
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
stages whenever the work crosses modules (geom -> smesh -> paravis) or the
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
- smesh: meshing a geometry (algorithms, hypotheses, compute).
- paravis: post-processing an existing simulation result (a MED file) --
  contours, cut planes, rendered scenes. NOT for colouring CAD geometry.
  Only use paravis when the request involves a results/field/MED file.

If the request comes with an "Image description" block, it was written by an
image-analysis agent from the user's reference image(s) -- treat it as the visual
target: use the parts it lists, their approximate proportions and placement to
shape your stage breakdown. Several views described together are ONE object, not
separate parts. Prefer explicit numeric dimensions given in the text request, then
dimensions the description says were read from the image; only fall back to values
marked as estimated (~) when nothing else gives one. Mention uncertainties in a
stage's params rather than inventing detail."""

# The smesh and paravis one-liners stay in _PLANNER_BASE (not restructured);
# they also appear in their spec.catalogue strings so routable() includes them.
# Do NOT move those three lines out — the prompt's byte-identity depends on them.
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
         image_description: str | None = None) -> list[dict]:
    system = PLANNER_SYSTEM_PROMPT + _DETAIL_HINT.get(detail, "")
    if image_description:
        request = f"{request}\n\nImage description:\n{image_description}"
    response = llm.chat(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": request}],
        module="planner",
    )
    return response["stages"]
