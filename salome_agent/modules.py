from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Terminator:
    """The one tool that ends a stage.

    `handle` absorbs the entire difference between a tool-calling module and a
    script-emitting one: geom's returns {result_names, call_log}, a script
    module's returns {script, warning}. The loop never asks which mode it is in.
    """
    name: str
    schema: dict
    description: str
    handle: Callable[[dict, list], dict]


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    # verify_cadence -> system prompt. Modules with no cadence ignore the arg.
    prompt: Callable[[str], str]
    terminator: Terminator
    # client, tool name, args -> the raw result dict the loop reasons about
    # (needs "ok"; may carry "records", "error").
    execute: Callable[[Any, str, dict], dict]
    # result dict -> the string that goes back as the tool message.
    render_result: Callable[[dict], str]
    # client -> {tool name: tool def}. The terminator is never in here.
    tools_for: Callable[[Any], dict]
    # What to say when the model replies with prose and no tool call. geom's
    # exact wording is pinned by its golden baselines.
    no_tool_nudge: str
    # call_log, last_error -> the result when the iteration cap runs out.
    on_exhausted: Callable[[list, dict | None], dict]

    # Everything below is a lever that goes INERT when left empty — that is
    # how smesh and paravis run the geom loop without being changed by it.
    core_tools: frozenset = frozenset()
    categories: frozenset = frozenset()
    risky_categories: frozenset = frozenset()
    measure_tools: frozenset = frozenset()

    render_upstream: Callable[..., "str | None"] | None = None
    render_retry: Callable[[dict], "str | None"] | None = None

    call_prefix: str | None = None
    preamble: str = ""
    # Empty means the planner never offers this module as a stage.
    catalogue: str = ""


MODULES: dict[str, ModuleSpec] = {}


def register(spec: ModuleSpec) -> ModuleSpec:
    MODULES[spec.name] = spec
    return spec


def spec_for(name: str) -> ModuleSpec:
    return MODULES[name]


def routable() -> list[ModuleSpec]:
    """Modules the planner may put in a plan — those with catalogue text."""
    return [s for s in MODULES.values() if s.catalogue]
