PLANNER_SYSTEM_PROMPT = """You turn a natural-language SALOME request into an
ordered list of stages. Each stage has: module (geom|smesh|paravis), intent
(short description of the job for that stage), params (dims in millimetres,
unconverted), inputs (named upstream objects this stage needs). Respond as
JSON: {"stages": [...]}.

**Never split the construction of a single connected shape across stages.**
Every primitive, boolean, transform, fillet and colour that goes into one
shape belongs to one stage -- the specialist builds it step by step itself.
Splitting it breaks the object-name handoff between stages.

Separate stages ARE fine (and expected) for: work that crosses modules
(geom -> smesh -> paravis); genuinely independent parts; and the distinct
named sub-assemblies of a larger model, as long as each such stage produces
one finished, clearly-named part that later stages combine by that name.

Module scope:
- geom: all CAD geometry -- primitives, booleans, transforms, fillet/chamfer,
  AND display properties of geometry (colour, transparency, display mode).
  "make it red", "colour the shape blue", "50% transparent" are geom.
- smesh: meshing a geometry (algorithms, hypotheses, compute).
- paravis: post-processing an existing simulation result (a MED file) --
  contours, cut planes, rendered scenes. NOT for colouring CAD geometry.
  Only use paravis when the request involves a results/field/MED file."""

_DETAIL_HINT = {
    "coarse": "\n\nPrefer the fewest stages possible: fold sub-assemblies "
              "into one stage unless they are truly independent.",
    "balanced": "",
    "fine": "\n\nLean towards a stage per named sub-assembly or major part "
            "of the model -- but only when each stage still produces exactly "
            "one finished, named part. Still never split one part's build.",
}


def blank_stages(stage_plans: list[dict]) -> dict:
    return {
        i: {"module": p["module"], "status": "pending", "script": None,
            "call_log": None, "static_retries": 0, "exec_retries": 0}
        for i, p in enumerate(stage_plans)
    }


def plan(request: str, llm, detail: str = "balanced") -> list[dict]:
    system = PLANNER_SYSTEM_PROMPT + _DETAIL_HINT.get(detail, "")
    response = llm.chat(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": request}],
    )
    return response["stages"]
