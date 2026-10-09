"""Named groups of sub-shapes: SHAPER's own make_group tool, and the code the
SMESH tools reuse to name faces and edges of a SHAPER shape."""
import builtins
import keyword
import re

from executor.shaper.features import _build
from executor.shaper.state import SUB_TYPES, state, tool
from executor.shaper.subshapes import _pick, _sels, _shape_name

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*$")
# Names result.py binds at module level, Python keywords and built-ins: a group
# named like one would replace it in the live namespace and in result.py.
RESERVED = frozenset({
    "salome", "SALOMEDS", "KernelBasis", "geomBuilder", "geompy", "gg",
    "model", "shaper_doc", "SketchAPI", "SketchAPI_Point",
    "smesh", "smeshBuilder", "SMESH", "StdMeshers",
    *keyword.kwlist, *keyword.softkwlist, *dir(builtins)})


def group_sub_type(ids) -> str:
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"ids must be a list like [\"F1\", \"F2\"], got {ids!r}")
    types = {v: k for k, v in SUB_TYPES.items()}
    letters = {i[:1] for i in ids}
    if len(letters) != 1 or next(iter(letters)) not in types:
        raise ValueError("ids must all be of one type (all E.., all F.., ...)")
    return types[next(iter(letters))]


def check_group_name(name, taken) -> None:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"name {name!r} must start with a letter and use only "
                         "letters, digits and _")
    if name in RESERVED:
        raise ValueError(f"the name {name!r} is reserved (a SALOME or Python name); "
                         "pick another")
    if name in taken:
        raise ValueError(f"the name {name!r} is taken; pick another")


def make_group(executor, shape: str, ids, name) -> dict:
    st = state(executor)
    check_group_name(name, set(st.objects) | set(st.sketches) | set(st.ns))
    sub_type = group_sub_type(ids)
    picked = _pick(executor, shape, sub_type, ids, None)
    code = (f'{name} = model.addGroup(shaper_doc, "{sub_type.upper()}", '
            f"[{_sels(sub_type, picked)}])")
    r = _build(executor, name, "addGroup", code, [shape], "group")
    if r["ok"]:
        st.groups[name] = shape
        r["sub_type"] = sub_type
        r["count"] = len(picked)
    return r


@tool("make_group", category="feature",
      description=("Publish sub-shapes of `shape` (`ids` from get_subshapes, all of one "
                   "type) as one named group, e.g. a face to name for a later stage."),
      schema={"type": "object", "properties": {
          "shape": {"type": "string"},
          "ids": {"type": "array", "items": {"type": "string"}},
          "name": {"type": "string"}}, "required": ["shape", "ids", "name"]})
def tool_make_group(executor, args: dict):
    return make_group(executor, _shape_name(executor, args), args.get("ids"), args.get("name"))
