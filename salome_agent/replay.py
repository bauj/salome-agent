from salome_agent.build_graph import BuildGraph, _num


class ReplayFailed(RuntimeError):
    def __init__(self, node_id: str, error: str):
        super().__init__(f"rebuilding {node_id} failed: {error}")
        self.node_id = node_id
        self.error = error


class RollbackFailed(RuntimeError):
    def __init__(self, message: str, graph: BuildGraph):
        super().__init__(message)
        self.graph = graph


def _forget(client, names) -> None:
    names = list(dict.fromkeys(names))
    if not names:
        return
    r = client.forget(names)
    if not r.get("ok"):
        raise ReplayFailed("forget", r.get("error") or "forget failed")


def _vars(graph: BuildGraph, ids) -> list[str]:
    return [v for nid in ids if nid in graph.nodes for v in graph.nodes[nid].out_vars]


def replay(client, graph: BuildGraph, ids) -> None:
    for nid in sorted(ids, key=_num):
        node = graph.nodes[nid]
        if node.module == "geom":
            r = client.call_tool(node.tool, node.tool_args, pin=node.out_vars)
        else:
            r = client.call_tool(node.tool, node.tool_args, pin=node.out_vars,
                                 module=node.module)
        if not r.get("ok"):
            raise ReplayFailed(nid, r.get("error") or "unknown error")
        graph.set_records(nid, r.get("records") or [])


def _swap(client, graph: BuildGraph, new: BuildGraph, ids, removed=()) -> None:
    """Forget `graph`'s nodes `ids` and `removed`, then build `ids` from `new`;
    on failure put `graph`'s nodes back."""
    old = sorted(set(ids) | set(removed), key=_num)
    _forget(client, _vars(graph, old))
    try:
        replay(client, new, ids)
    except ReplayFailed:
        restored = graph.copy()
        try:
            _forget(client, _vars(new, ids) + _vars(graph, old))
            replay(client, restored, old)
        except ReplayFailed as e2:
            raise RollbackFailed(f"could not restore after a failed edit: {e2}", new) from e2
        raise


def apply_edit(client, graph: BuildGraph, node_id: str, param: str, value):
    new = graph.copy()
    new.edit_arg(node_id, param, value)
    ids = new.rebuilt_by(node_id)
    _swap(client, graph, new, ids)
    return new, ids


def apply_delete(client, graph: BuildGraph, node_id: str):
    owners = {o for nid in graph.affected(node_id) for o in graph.receivers(nid)}
    new = graph.copy()
    removed = [n.id for n in new.delete(node_id)]
    ids = sorted({nid for o in owners if o in new.nodes for nid in new.affected(o)}, key=_num)
    _swap(client, graph, new, ids, removed)
    return new, removed


def sync(client, current: BuildGraph, target: BuildGraph) -> BuildGraph:
    cur, tgt = current.nodes, target.nodes
    changed = [nid for nid, n in tgt.items()
               if nid not in cur or cur[nid].tool_args != n.tool_args]
    rebuild: set[str] = set()
    for nid in changed:
        rebuild |= set(target.rebuilt_by(nid))
    for nid in cur:
        if nid not in tgt:
            for o in current.receivers(nid):
                if o in tgt:
                    rebuild |= set(target.affected(o))
    drop = [nid for nid in cur if nid not in tgt] + [nid for nid in rebuild if nid in cur]
    _forget(client, _vars(current, drop))
    out = target.copy()
    replay(client, out, rebuild)
    return out


def rebuild_all(client, graph: BuildGraph, extra_names=()) -> BuildGraph:
    _forget(client, sorted(set(graph.all_out_vars()) | set(extra_names)))
    out = graph.copy()
    replay(client, out, list(out.nodes))
    return out
