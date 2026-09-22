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
        r = client.call_tool(node.tool, node.tool_args, pin=node.out_vars)
        if not r.get("ok"):
            raise ReplayFailed(nid, r.get("error") or "unknown error")
        graph.set_records(nid, r.get("records") or [])


def apply_edit(client, graph: BuildGraph, node_id: str, param: str, value):
    new = graph.copy()
    new.edit_arg(node_id, param, value)
    ids = new.affected(node_id)
    _forget(client, _vars(graph, ids))
    try:
        replay(client, new, ids)
    except ReplayFailed:
        restored = graph.copy()
        try:
            _forget(client, _vars(new, ids) + _vars(graph, ids))
            replay(client, restored, ids)
        except ReplayFailed as e2:
            raise RollbackFailed(f"could not restore after a failed edit: {e2}", new) from e2
        raise
    return new, ids


def apply_delete(client, graph: BuildGraph, node_id: str):
    new = graph.copy()
    removed = new.delete(node_id)
    _forget(client, [v for n in removed for v in n.out_vars])
    return new, [n.id for n in removed]


def sync(client, current: BuildGraph, target: BuildGraph) -> BuildGraph:
    cur, tgt = current.nodes, target.nodes
    changed = [nid for nid, n in tgt.items()
               if nid not in cur or cur[nid].tool_args != n.tool_args]
    rebuild: set[str] = set()
    for nid in changed:
        rebuild |= set(target.affected(nid))
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
