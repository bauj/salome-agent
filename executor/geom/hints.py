import difflib

from executor.geom.helpers import registry

# OCC / geompy failure text -> what to do about it. Seeded from the ok:false
# results in the -vv trace corpus (task 218); raw OCC messages name the
# failing algorithm, never the fix.
_HINTS = (
    # Checked live: an open curved face fails at any thickness (make_thick_solid
    # names it itself when it can); a filleted solid gives an invalid result at
    # any thickness.
    (("Offset driver failed",),
     "open a flat face (not a cylinder's or cone's side), and hollow the solid "
     "before filleting it"),
    (("make_thick_solid: CheckShape failed",),
     "hollow the solid before filleting or chamfering it — a rounded solid "
     "usually cannot be offset"),
    (("Offset construction failed",),
     "try a smaller offset, or offset the shape before filleting it"),
    (("no suitable edges",),
     "those edges cannot be rounded (a seam or degenerate edge, such as a "
     "cylinder's seam line or a cone's apex) — pick other edges"),
    (("non valid shape result",),
     "the operands probably share coincident or tangent faces — make the tool "
     "overlap the main shape (e.g. extend it 1 mm past the face it cuts) and retry"),
)


def with_hint(error: str) -> str:
    if "— hint:" in error:  # the tool already said something more specific
        return error
    for needles, hint in _HINTS:
        if any(n in error for n in needles):
            return f"{error} — hint: {hint}"
    return error


def key_error_message(executor, tool: str, args: dict, key) -> str:
    """A bare KeyError stringifies to just the key ("'cut_1'"), which reads as
    nothing. Say which of the two usual causes it is."""
    props = registry.schemas().get(tool, {}).get("properties", {})
    if key in props and key not in args:
        return f"{tool}: missing argument {key!r}"
    if not isinstance(key, str):
        return f"{tool}: KeyError {key!r}"
    msg = f"{tool}: no object named {key!r}"
    close = difflib.get_close_matches(key, list(executor.geom_bindings), n=3, cutoff=0.6)
    if close:
        msg += f" — did you mean {', '.join(close)}?"
    return msg + (" Use names exactly as a tool result published them "
                  "(list_study_objects lists them).")
