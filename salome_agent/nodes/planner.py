PLANNER_SYSTEM_PROMPT = """You turn a natural-language SALOME request into an
ordered list of stages. Each stage has: module (geom|smesh|paravis), intent
(short description), params (dims in millimetres, unconverted), inputs (named
upstream objects this stage needs). Only emit stages the request actually
needs. Repeating a module is allowed. Respond as JSON: {"stages": [...]}.

Module scope:
- geom: all CAD geometry -- primitives, booleans, transforms, fillet/chamfer,
  AND display properties of geometry (colour, transparency, display mode).
  "make it red", "colour the shape blue", "50% transparent" are geom stages.
- smesh: meshing a geometry (algorithms, hypotheses, compute).
- paravis: post-processing an existing simulation result (a MED file) --
  contours, cut planes, rendered scenes. NOT for colouring CAD geometry.
  Only use paravis when the request involves a results/field/MED file.

If colour / transparency / display is its own stage, it must be the LAST
stage and act on the final shape -- a transform after colouring produces a
new, uncoloured shape."""


def blank_stages(stage_plans: list[dict]) -> dict:
    return {
        i: {"module": p["module"], "status": "pending", "script": None,
            "call_log": None, "static_retries": 0, "exec_retries": 0}
        for i, p in enumerate(stage_plans)
    }


def plan(request: str, llm) -> list[dict]:
    response = llm.chat(
        messages=[{"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                  {"role": "user", "content": request}],
    )
    return response["stages"]
