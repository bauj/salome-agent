from typing import Annotated, Literal, Optional, TypedDict
import operator


class StagePlan(TypedDict):
    module: Literal["geom", "smesh", "paravis"]
    intent: str
    params: dict
    inputs: list[str]


class StageError(TypedDict):
    phase: Literal["static", "exec"]
    message: str
    offending_line: Optional[int]
    traceback: Optional[str]


class StageRecord(TypedDict):
    module: str
    status: Literal["pending", "ok", "failed", "skipped"]
    script: Optional[str]
    call_log: Optional[list[dict]]
    static_retries: int
    exec_retries: int


class SessionHandle(TypedDict):
    mode: Literal["attach", "launch", "live", "none"]
    port: Optional[int]
    executor_pid: Optional[int]
    alive: bool
    owned: bool


class RunState(TypedDict):
    run_id: str
    user_request: str
    config: dict

    plan: list[StagePlan]
    plan_approved: bool
    stage_idx: int

    session: SessionHandle
    study_ns: str

    stages: dict[int, StageRecord]
    last_error: Optional[StageError]
    history: Annotated[list[dict], operator.add]

    final_script: Optional[str]
    report: Optional[str]
    fatal: Optional[str]
