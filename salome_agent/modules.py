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
    # Order-preserving: this list is serialized verbatim into the load_tools
    # tool schema's enum, which is prompt surface — a frozenset would silently
    # alphabetize it.
    categories: tuple = ()
    risky_categories: frozenset = frozenset()
    measure_tools: frozenset = frozenset()

    render_upstream: Callable[..., "str | None"] | None = None
    render_retry: Callable[[dict], "str | None"] | None = None

    call_prefix: str | None = None
    preamble: str = ""
    # Empty means the planner never offers this module as a stage.
    catalogue: str = ""


MODULES: dict[str, ModuleSpec] = {}
_SPECIALISTS_IMPORTED = False


def _ensure_specialists_registered() -> None:
    """Ensure specialists are imported so they register their specs.

    MODULES is populated as a side effect of importing specialist modules.
    This function is called on first use of spec_for/routable to guarantee
    that a fresh interpreter (without prior specialist imports) still works.

    Guard pattern: flag is set AFTER imports succeed (not before). Tension:
    re-entry protection wants early flag set, but failure recovery requires
    late set to avoid silent MODULES=={} if an import fails. We choose late
    set: a failed import must not leave the registry looking permanently empty,
    because routable() would silently return [] and produce wrong plans instead
    of raising a diagnostic error. Re-entry is still protected: on second call,
    if first call failed and didn't set flag, we retry; if it succeeded, flag
    is set and we skip. The failure case retries indefinitely (OK: imports are
    deterministic, and the error is loud, so operator sees and fixes it).
    """
    global _SPECIALISTS_IMPORTED
    if _SPECIALISTS_IMPORTED:
        return
    # Import in order to trigger registration via @register decorator.
    # Must be done here, not at module level, to avoid import cycles.
    try:
        import salome_agent.specialists.geom  # noqa: F401 — trigger registration
        import salome_agent.specialists.script_modules  # noqa: F401 — trigger registration
    except Exception:
        # Do not set flag; let caller retry on next spec_for/routable call.
        # This ensures MODULES is not left silently empty.
        raise
    _SPECIALISTS_IMPORTED = True


def register(spec: ModuleSpec) -> ModuleSpec:
    MODULES[spec.name] = spec
    return spec


def spec_for(name: str) -> ModuleSpec:
    _ensure_specialists_registered()
    return MODULES[name]


def routable() -> list[ModuleSpec]:
    """Modules the planner may put in a plan — those with catalogue text."""
    _ensure_specialists_registered()
    return [s for s in MODULES.values() if s.catalogue]
