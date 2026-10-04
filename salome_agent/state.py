from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict, NotRequired
import operator


class StagePlan(TypedDict):
    module: Literal["geom", "shaper", "smesh"]
    intent: str
    params: dict
    # Plain-words description of what this stage needs, never object names
    # (planner.py).
    inputs: list[str]
    # Optional planner hint: the geom tool names this stage expects to use
    # (task 193). Absent on smesh stages.
    tools: NotRequired[list[str]]
    # Optional: the vision spec part ids this stage builds (task 286); the stage
    # is handed those parts' exact data.
    parts: NotRequired[list[str]]


class StageError(TypedDict):
    phase: Literal["static", "exec"]
    message: str
    offending_line: Optional[int]
    traceback: Optional[str]


class StageRecord(TypedDict, total=False):
    module: str
    status: Literal["pending", "ok", "failed", "skipped"]
    script: Optional[str]
    call_log: Optional[list[dict]]
    # Task 5's per-call trace: [{tool, tool_args, records}] — what a build
    # node is made from. None for script modules.
    calls: Optional[list[dict]]
    static_retries: int
    exec_retries: int
    # {wall, llm_s, core_s, exec_s, turns, in_tok, out_tok} — accumulated
    # across retry re-entries. Written by graph nodes, read by report.py.
    timing: dict


class SessionHandle(TypedDict):
    mode: Literal["gui", "headless", "none"]
    pid: Optional[int]
    salome_port: Optional[int]
    name_prefix: str


class RunState(TypedDict):
    run_id: str
    user_request: str
    reference_images: Optional[list[str]]
    image_spec: Optional[dict]
    config: dict

    plan: list[StagePlan]
    plan_approved: bool
    stage_idx: int
    plan_timing: Optional[dict]

    session: SessionHandle
    study_ns: str

    stages: dict[int, StageRecord]
    last_error: Optional[StageError]
    history: Annotated[list[dict], operator.add]

    final_script: Optional[str]
    report: Optional[str]
    fatal: Optional[str]
    aborted: Optional[bool]
    build_graph: Optional[dict]

    # agentic chat (task 194) — absent in one-shot runs
    turn_idx: int
    selection: Optional[list]
    turn_outcome: Optional[dict]
    replan_context: Optional[str]
    degraded: Optional[bool]
