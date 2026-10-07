"""One sketch is edited at a time, as in SHAPER's GUI: a sketch is being edited
from create_sketch, or from the first change to a closed one, until
close_sketch, and nothing else is built meanwhile. A sketch changed while
another one is being edited makes SHAPER update it in an endless loop when
result.py runs."""
from executor.shaper.sketch import _reply, exec_records
from executor.shaper.state import (
    check_results, history_index, measure, record, registry, sketch_of, state, tool,
    transaction,
)

EDITS = frozenset({"sketch_add", "sketch_constrain", "sketch_remove", "sketch_copy",
                   "sketch_fillet"})
_ALWAYS = frozenset({"sketch_status", "close_sketch"})


def refusal(executor, name: str, args: dict) -> str | None:
    """Why `name` cannot run now, or None."""
    st = executor.__dict__.get("shaper_state")
    current = st.editing if st is not None else None
    if current is None or name in _ALWAYS:
        return None
    if name in EDITS:
        target = args.get("sketch")
        if target == current or target not in st.sketches:
            return None
        return (f"{current} is still being edited: call close_sketch on it before "
                f"changing {target}")
    if registry.categories().get(name) == "query":
        return None
    return f"{current} is still being edited: call close_sketch on it before building anything else"


def call(executor, name: str, args: dict, handler) -> dict:
    """Run a tool under the rule: refuse it, or track which sketch it opens."""
    why = refusal(executor, name, args)
    if why:
        return {"ok": False, "error": why}
    reopened = None
    if name in EDITS:
        st = state(executor)
        sk = st.sketches.get(args.get("sketch"))
        if sk is not None and st.editing is None:
            reopened = sk
            sk.baseline = _health(executor, _later(st, sk.name))
    result = handler()
    st = executor.__dict__.get("shaper_state")
    if not result.get("ok"):
        if reopened is not None:
            reopened.baseline = None
        return result
    if name == "create_sketch":
        st.editing = result["sketch"]
    elif name in EDITS:
        sk = st.sketches[args["sketch"]]
        st.editing = sk.name
        broken = _breaks(executor, sk)
        if broken:
            result["breaks"] = broken
    return result


def _later(st, name: str) -> list:
    """Published features made after `name`, in history order."""
    order = history_index(st)
    pos = order.get(name, -1)
    return sorted((n for n in st.objects if order.get(n, -1) > pos), key=order.get)


def _health(executor, names: list) -> dict:
    """name -> (check status, holds a solid, detail). A solid whose sketch lost a
    curve can become an open shell with no SHAPER error: only its volume tells."""
    out = {}
    for n, r in (check_results(executor, names).items() if names else ()):
        solid = r["status"] == "ok" and measure(executor, n)["volume"] > 1e-9
        out[n] = (r["status"], solid, r.get("detail"))
    return out


def _breaks(executor, sk) -> dict:
    """Later features that built before this edit of `sk` and no longer do."""
    if not sk.baseline:
        return {}
    with transaction() as m:
        m.do()
    out = {}
    for n, (status, solid, detail) in _health(executor, list(sk.baseline)).items():
        was, was_solid, _ = sk.baseline[n]
        if was != "ok":
            continue
        if status != "ok":
            out[n] = detail or status
        elif was_solid and not solid:
            out[n] = "it is no longer a solid (volume 0)"
    return out


@tool("close_sketch", category="sketch",
      description=(
          "Finish editing the sketch `sketch`: features can then be built from it. A "
          "sketch is being edited from create_sketch until this call. While a sketch "
          "is being edited, only the sketch tools on it and the queries work; a new "
          "sketch, a change to another sketch or any other feature is refused. Adding "
          "to or changing a finished sketch makes it edited again."),
      schema={"type": "object", "properties": {"sketch": {"type": "string"}},
              "required": ["sketch"]})
def tool_close_sketch(executor, args: dict):
    st = state(executor)
    sk = sketch_of(executor, args.get("sketch"))
    if st.editing != sk.name:
        now = f"; {st.editing} is" if st.editing else ""
        raise ValueError(f"{sk.name} is not being edited{now}")
    rec = record("close_sketch", "model.do()", recv=sk.name)
    exec_records(st, [rec])
    broken = _breaks(executor, sk)
    if broken:
        lines = "; ".join(f"{n}: {d}" for n, d in broken.items())
        raise ValueError(
            f"{sk.name} stays edited: with this change, features built from it no "
            f"longer build ({lines}). Change the sketch until they do, then close it again")
    st.editing = None
    sk.baseline = None
    return _reply(executor, sk, [rec], None)


def check_finish(executor, names: list) -> dict:
    """finish_stage's re-check, and the sketch still being edited."""
    out = check_results(executor, names)
    st = executor.__dict__.get("shaper_state")
    if st is not None and st.editing:
        out[st.editing] = {"status": "editing"}
    return out
