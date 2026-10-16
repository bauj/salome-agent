"""Every free string argument of every GEOM, SHAPER and SMESH tool, given a value
that tries to break out of the generated code and create a marker directory.
The other arguments name real objects, so the payload reaches the code it targets."""
import os

import pytest

from ._salome_root import SALOME_ROOT

pytestmark = [pytest.mark.salome,
              pytest.mark.skipif(SALOME_ROOT is None,
                                 reason="set SALOME_ROOT or salome.root in config.json")]

_PLATE = [
    {"id": "L1", "type": "line", "start": [0, 0], "end": [40, 0]},
    {"id": "L2", "type": "line", "start": [40, 0], "end": [40, 20]},
    {"id": "L3", "type": "line", "start": [40, 20], "end": [0, 20]},
    {"id": "L4", "type": "line", "start": [0, 20], "end": [0, 0]},
]
_CLOSE = [{"type": "coincident", "refs": [a, b]} for a, b in
          (("L1.end", "L2.start"), ("L2.end", "L3.start"),
           ("L3.end", "L4.start"), ("L4.end", "L1.start"))]
_SHAPE_KEYS = {"shape", "base", "main", "target", "entity", "to", "shape1", "shape2",
               "first", "second", "objects", "tools", "shapes", "names", "path"}
_FIXED = {"axis": "OZ", "plane": "XOY", "where": "all", "id": "L1", "ids": "L1",
          "refs": "L1.end", "at": "L1.end", "of": "L1", "edges": "1", "name": "fz_new",
          "group": "fz_lid", "on": "fz_lid", "end_face": "fz_lid", "faces": "fz_lid",
          "order": "fz_lid", "normal": "OZ", "major_dir": "OX", "vector": "OX",
          "vector2": "OY", "start_tangent": "OX", "end_tangent": "OY",
          "pnt1": "0,0,0", "point": "0,0,0", "center": "0,0,0", "start": "0,0,0",
          "origin": "0,0,0", "pnt2": "10,0,0", "end": "10,0,0", "through": "5,5,0",
          "via": "5,5,0"}
_POINTS = ["0,0,0", "10,0,0", "0,10,0"]
_NUMBERS = {"count": 2, "count2": 2, "factor": 2, "size": 5, "radius": 2,
            "radius_major": 10, "radius_minor": 2, "base_radius": 4, "top_radius": 2,
            "step": 15, "step2": 15, "angle_deg": 30, "angle": 30}
# Tools that work inside an open sketch; every other SHAPER tool needs it closed.
_IN_SKETCH = {"sketch_add", "sketch_constrain", "sketch_remove", "sketch_status",
              "sketch_fillet", "sketch_copy", "close_sketch"}


def _ok(client, module, tool, **args):
    r = client.call_tool(tool, args, module=module)
    assert r["ok"] is True, (tool, r.get("error"))
    return r


def _sketch(client) -> str:
    sk = _ok(client, "shaper", "create_sketch", plane="XOY")["sketch"]
    _ok(client, "shaper", "sketch_add", sketch=sk, entities=_PLATE)
    _ok(client, "shaper", "sketch_constrain", sketch=sk, constraints=_CLOSE)
    return sk


@pytest.fixture(scope="module")
def objects(running_server):
    import json
    from salome_agent.executor_client import ExecutorClient
    from .conftest import _HS
    hs = json.loads(open(_HS).read())
    client = ExecutorClient(hs["host"], hs["port"], hs["token"], "")
    sb = _ok(client, "shaper", "make_box", dx=10, dy=20, dz=30, center=[5, 10, 15])["name"]
    gb = _ok(client, "geom", "make_box", dx=10, dy=20, dz=30)["result_shape"]
    sk = _sketch(client)
    _ok(client, "shaper", "close_sketch", sketch=sk)
    top = _ok(client, "smesh", "get_subshapes", shape=sb, sub_type="face",
              where={"sorted_by": "z", "take": "max", "count": 1})["subshapes"][0]["id"]
    _ok(client, "smesh", "name_subshapes", shape=sb, ids=[top], name="fz_lid")
    mesh = _ok(client, "smesh", "create_mesh", shape=sb)["name"]
    yield client, {"geom": gb, "shaper": sb, "smesh": sb}, sk, mesh
    client.close()


def _default(module, key, objs, sk, mesh, path=()):
    if key == "sketch":
        return sk
    if key == "mesh":
        return mesh
    if key == "to" and any(st[0] == "k" and st[1] in ("segments", "slot") for st in path):
        return "10,0,0"
    if key in _SHAPE_KEYS:
        return objs[module]
    return _FIXED.get(key, "OZ")


def _fill(s, key, ctx):
    if "enum" in s:
        return s["enum"][0]
    for k in ("anyOf", "oneOf"):
        if k in s:
            strs = [v for v in s[k] if v.get("type") == "string"]
            return _fill(strs[0] if strs else s[k][0], key, ctx)
    t = s.get("type")
    t = next((x for x in t if x != "null"), None) if isinstance(t, list) else t
    if t in ("number", "integer"):
        return _NUMBERS.get(key, 1)
    if t == "boolean":
        return False
    if t == "array":
        items = s.get("items") or {}
        if key == "points" and items.get("type") == "string":
            return list(_POINTS)
        n = 3 if items.get("type") in ("number", "integer") else 1
        return [_fill(items, key, ctx) for _ in range(n)]
    if t == "object" or "properties" in s:
        props = s.get("properties") or {}
        return {k: _fill(props[k], k, ctx) for k in s.get("required", props) if k in props}
    return _default(*ctx[:1], key, *ctx[1:])


def _slots(s, path=()):
    """Paths to every place a free string can go: ("v", i) picks a schema
    variant, ("k", name) a property, ("i",) a list item."""
    if "enum" in s:
        return
    for k in ("anyOf", "oneOf"):
        for i, v in enumerate(s.get(k, ())):
            yield from _slots(v, path + (("v", i),))
    t = s.get("type")
    ts = t if isinstance(t, list) else [t]
    if "string" in ts or (t is None and not s.get("properties") and "items" not in s
                          and not s.get("anyOf") and not s.get("oneOf")):
        yield path
    for k, v in (s.get("properties") or {}).items():
        yield from _slots(v, path + (("k", k),))
    if isinstance(s.get("items"), dict):
        yield from _slots(s["items"], path + (("i",),))


def _place(args, schema, path, value, ctx, key=None):
    """args with `value` at `path`, building the containers on the way."""
    if not path:
        return value
    step, rest = path[0], path[1:]
    if step[0] == "v":
        sub = (schema.get("anyOf") or schema.get("oneOf"))[step[1]]
        return _place(_fill(sub, key, ctx), sub, rest, value, ctx, key)
    if step[0] == "i":
        items = schema.get("items") or {}
        cur = args if isinstance(args, list) and args else [_fill(items, key, ctx)]
        return [_place(cur[0], items, rest, value, ctx, key)] + cur[1:]
    name = step[1]
    sub = schema["properties"][name]
    cur = dict(args) if isinstance(args, dict) else {}
    cur[name] = _place(cur.get(name, _fill(sub, name, ctx)), sub, rest, value, ctx, name)
    return cur


def _payloads(marker, default):
    forms = ['{d}\n__import__("os").mkdir({p})\n',
             '{d}")\n__import__("os").mkdir({p})\n#',
             "{d}')\n__import__('os').mkdir({p})\n#",
             '{d})\n__import__("os").mkdir({p})\n#',
             '{d}"]\n__import__("os").mkdir({p})\n#']
    return [f.format(d=default, p=repr(f"{marker}-{i}")) for i, f in enumerate(forms)]


def _attack(client, module, tool, ctx, tmp_path) -> int:
    _, objs, sk, mesh = ctx
    schema = tool["parameters"]
    base = _fill(schema, None, ctx) or {}
    calls = 0
    for path in set(_slots(schema)):
        key = next((st[1] for st in reversed(path) if st[0] == "k"), None)
        default = _default(module, key, objs, sk, mesh, path)
        marker = tmp_path / f"{tool['name']}-{'.'.join(str(st[-1]) for st in path)}"
        for payload in _payloads(marker, default):
            r = client.call_tool(tool["name"], _place(base, schema, path, payload, ctx),
                                 module=module)
            calls += 1
            if r.get("ok") and tool["name"] == "create_sketch":
                client.call_tool("close_sketch", {"sketch": r["sketch"]}, module=module)
    return calls


@pytest.mark.parametrize("module", ["geom", "shaper", "smesh"])
def test_no_string_argument_runs_code(objects, module, tmp_path):
    client, objs, sk, mesh = objects
    tools = client.list_tools(module)["tools"]
    calls = 0
    for t in tools:
        if module == "shaper" and t["name"] in _IN_SKETCH:
            open_sk = _sketch(client)
            calls += _attack(client, module, t, (module, objs, open_sk, mesh), tmp_path)
            client.call_tool("close_sketch", {"sketch": open_sk}, module=module)
        else:
            calls += _attack(client, module, t, (module, objs, sk, mesh), tmp_path)
    made = sorted(os.listdir(tmp_path))
    assert calls > 0
    assert made == [], f"tool arguments ran code: {made}"
