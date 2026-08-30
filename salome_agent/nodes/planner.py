PLANNER_SYSTEM_PROMPT = """You turn a natural-language SALOME request into an
ordered list of stages. Each stage has: module (geom|smesh|paravis), intent
(short description), params (dims in millimetres, unconverted), inputs (named
upstream objects this stage needs). Only emit stages the request actually
needs. Repeating a module is allowed. Respond as JSON: {"stages": [...]}."""


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
