from executor.geom_tools import TOOL_DESCRIPTIONS
from salome_agent.llm import with_images


def _geom_catalogue() -> str:
    # Names + one-liners only (no arg schemas) so the planner knows what a
    # single geom stage can actually do and stops over-splitting parts.
    lines = [f"  {n} -- {d}" for n, d in TOOL_DESCRIPTIONS.items()]
    return "geom tools available to a single geom stage:\n" + "\n".join(lines)


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
its own stage, plus a final stage that combines them by name. Do that when
the parts are distinct enough to name. A car is an assembly: body, cabin,
wheels are separate parts even though they are fused at the end. Also use
separate stages whenever the work crosses modules (geom -> smesh -> paravis)
or the request asks for genuinely independent deliverables.

Module scope:
- geom: all CAD geometry -- primitives, booleans, transforms, fillet/chamfer,
  AND display properties of geometry (colour, transparency, display mode).
  "make it red", "colour the shape blue", "50% transparent" are geom.
- smesh: meshing a geometry (algorithms, hypotheses, compute).
- paravis: post-processing an existing simulation result (a MED file) --
  contours, cut planes, rendered scenes. NOT for colouring CAD geometry.
  Only use paravis when the request involves a results/field/MED file.

If reference image(s) are attached to the request, use them as the visual
target: identify the distinct parts/shapes shown and their approximate
relative proportions and placement to shape your stage breakdown. Several
images are usually different views/plans of the same part (e.g. front, top,
side) -- read them together, not as separate parts. Prefer explicit numeric
dimensions given in the text request; only estimate dimensions from the
image(s) when the text gives none."""

PLANNER_SYSTEM_PROMPT = _PLANNER_BASE + "\n\n" + _geom_catalogue()

_DETAIL_HINT = {
    "coarse": "\n\nPrefer the fewest stages: keep an assembly's parts in one "
              "stage unless they are truly independent deliverables.",
    "balanced": "",
    "fine": "\n\nFor any assembly, give EACH part (or group of identical "
            "parts) its own stage and add a final stage that combines them. "
            "The final result being one fused solid does not make the parts "
            "one part. Still never split a single part's own construction.",
}


def blank_stages(stage_plans: list[dict]) -> dict:
    return {
        i: {"module": p["module"], "status": "pending", "script": None,
            "call_log": None, "static_retries": 0, "exec_retries": 0}
        for i, p in enumerate(stage_plans)
    }


def plan(request: str, llm, detail: str = "balanced",
        images: list[str] | None = None) -> list[dict]:
    system = PLANNER_SYSTEM_PROMPT + _DETAIL_HINT.get(detail, "")
    response = llm.chat(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": with_images(request, images)}],
    )
    return response["stages"]
