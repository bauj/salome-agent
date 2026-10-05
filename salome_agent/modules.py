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
    # Its tools run in SALOME: with no server (--no-session) the stage ends
    # fatal instead of starting.
    needs_session: bool = False

    # Everything below is a lever that goes INERT when left empty — that is
    # how smesh runs the geom loop without being changed by it.
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
    # Closes the stage's first user message, next to the intent the model reads.
    kickoff: str = ""
    # Empty means the planner never offers this module as a stage.
    catalogue: str = ""
    # Tools that also end the loop, each with its own handle (amend's replan).
    extra_terminators: tuple = ()
    # result_names, this attempt's calls, full call_log, upstream names, last
    # known shape_info by name -> a report that reopens the stage once, or None
    # to let the terminator close it.
    audit: Callable[..., "str | None"] | None = None
    # args, call_log, last known shape_info by name -> a refusal sent back on
    # EVERY finish while it holds (audit reopens once), or None to let it close.
    gate: Callable[..., "str | None"] | None = None
    # result.py: closes the script after every module's body; how a published
    # name is emitted (geom: geompy.addToStudy).
    postamble: str = ""
    publish_line: str = "geompy.addToStudy({var}, {name!r})"
    # The module's published names are study objects the loop may remove:
    # offers the discard tool, and a clean finish removes what the results do
    # not reach, so the study matches result.py.
    forgets: bool = False
    # Offers the calc tool (sandboxed Python arithmetic, answered agent-side).
    calc: bool = False
    # On every finish the server re-checks result_names (exists, still builds,
    # valid, not empty); a bad one refuses the finish.
    checks_results: bool = False


MODULES: dict[str, ModuleSpec] = {}
_SPECIALISTS_IMPORTED = False


def _ensure_specialists_registered() -> None:
    """Import the specialists so MODULES gets populated, on first use.

    Flag is set only after the imports succeed, so a failed import retries
    on the next call instead of latching MODULES permanently empty.
    """
    global _SPECIALISTS_IMPORTED
    if _SPECIALISTS_IMPORTED:
        return
    # Import in order to trigger registration via @register decorator.
    # Must be done here, not at module level, to avoid import cycles.
    import salome_agent.specialists.geom  # noqa: F401 — trigger registration
    import salome_agent.specialists.script_modules  # noqa: F401 — trigger registration
    import salome_agent.specialists.shaper  # noqa: F401 — trigger registration
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


def llm_module_names() -> list[str]:
    """Every name llm.modules accepts: the planner, vision, each specialist, and
    amend (chat edits, built per turn so not in MODULES)."""
    _ensure_specialists_registered()
    return ["planner", "vision", *MODULES, "amend"]
