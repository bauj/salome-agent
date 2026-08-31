PLANNER_SYSTEM_PROMPT = """You turn a natural-language SALOME request into an
ordered list of stages. Each stage has: module (geom|smesh|paravis), intent
(short description of the WHOLE job for that module), params (dims in
millimetres, unconverted), inputs (named upstream objects this stage needs).
Respond as JSON: {"stages": [...]}.

**One stage per module.** A stage is a whole module's worth of work, not a
single operation. All the geometry -- every primitive, boolean, transform,
fillet and colour -- is ONE geom stage; the geom specialist builds it step
by step itself. Do NOT make "create the trunk", "create the leaves", "move
the leaves", "colour it" into separate stages -- that splits one shape's
construction across stages and the names stop lining up.

Emit more than one stage only when:
- the work crosses modules (geom -> smesh -> paravis), or
- the request is genuinely independent deliverables (e.g. "make part A,
  then separately make part B").

Module scope:
- geom: all CAD geometry -- primitives, booleans, transforms, fillet/chamfer,
  AND display properties of geometry (colour, transparency, display mode).
  "make it red", "colour the shape blue", "50% transparent" are geom.
- smesh: meshing a geometry (algorithms, hypotheses, compute).
- paravis: post-processing an existing simulation result (a MED file) --
  contours, cut planes, rendered scenes. NOT for colouring CAD geometry.
  Only use paravis when the request involves a results/field/MED file."""


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
