import copy
import json
from dataclasses import asdict, dataclass, field

from salome_agent.codegen import _refs_in


class GraphError(ValueError):
    pass


def _num(node_id: str) -> int:
    return int(node_id[1:])


@dataclass
class BuildNode:
    id: str
    turn: int
    block: str
    module: str
    tool: str
    tool_args: dict
    records: list
    deps: list = field(default_factory=list)

    @property
    def out_vars(self) -> list[str]:
        return [v for r in self.records for v in r.get("out_vars", [])]

    @property
    def published(self) -> list[str]:
        return [n for r in self.records for n in (r.get("publish") or [])]


@dataclass
class Block:
    id: str
    turn: int
    label: str
    module: str
    result_names: list
    script: str | None = None


def _record_refs(records: list) -> set[str]:
    refs: set[str] = set()
    for r in records:
        for arg in r.get("args", []):
            refs |= _refs_in(arg)
        if r.get("recv"):
            refs.add(r["recv"])
    return refs


def _mentions(value, names: set[str]) -> bool:
    if isinstance(value, str):
        return value in names
    if isinstance(value, (list, tuple)):
        return any(_mentions(v, names) for v in value)
    return False


class BuildGraph:
    """The conversation's accumulated tool calls. result.py and the study are
    both projections of it."""

    def __init__(self):
        self.nodes: dict[str, BuildNode] = {}
        self.blocks: list[Block] = []
        self.requests: list[str] = []
        self._next_node = 1
        self._next_block = 1

    def add_block(self, turn: int, label: str, module: str, result_names: list,
                  script: str | None = None, before: str | None = None) -> str:
        """`before`: the id of a block the new one goes in front of (else last)."""
        bid = f"b{self._next_block}"
        self._next_block += 1
        block = Block(bid, turn, label, module, list(result_names), script)
        at = next((i for i, b in enumerate(self.blocks) if b.id == before), len(self.blocks))
        self.blocks.insert(at, block)
        return bid

    def owner(self, var: str) -> str | None:
        for node in self.nodes.values():
            if var in node.out_vars:
                return node.id
        return None

    def _deps(self, node_id: str, records: list) -> list[str]:
        own = {v for r in records for v in r.get("out_vars", [])}
        deps = {self.owner(ref) for ref in _record_refs(records) - own}
        deps.discard(None)
        deps.discard(node_id)
        return sorted(deps, key=_num)

    def append(self, block: str, tool: str, tool_args: dict, records: list, turn: int,
               module: str = "geom") -> str:
        nid = f"n{self._next_node}"
        self._next_node += 1
        self.nodes[nid] = BuildNode(nid, turn, block, module, tool, dict(tool_args),
                                    list(records), self._deps(nid, records))
        return nid

    def _live(self, node_id: str) -> BuildNode:
        node = self.nodes.get(node_id)
        if node is None:
            live = ", ".join(self.nodes) or "(none)"
            raise GraphError(f"{node_id} does not exist (deleted or never built); "
                             f"current nodes are {live}")
        return node

    def edit_arg(self, node_id: str, param: str, value) -> None:
        node = self._live(node_id)
        names = set(self.all_out_vars())
        if _mentions(node.tool_args.get(param), names) or _mentions(value, names):
            raise GraphError(
                f"{node_id}.{param} names a shape; edit_arg only changes literal values. "
                f"To change which shapes {node_id} uses, delete_node it and build it again.")
        node.tool_args[param] = value

    def affected(self, node_id: str) -> list[str]:
        self._live(node_id)
        hit = {node_id}
        for nid in sorted(self.nodes, key=_num):
            if hit & set(self.nodes[nid].deps):
                hit.add(nid)
        return sorted(hit, key=_num)

    def receivers(self, node_id: str) -> list[str]:
        """The nodes that made what `node_id` changes in place (the sketch it
        adds to, the mesh it sets up, the shape it colours)."""
        node = self._live(node_id)
        own = set(node.out_vars)
        owners = {self.owner(r["recv"]) for r in node.records
                  if r.get("recv") and r["recv"] not in own}
        owners.discard(None)
        return sorted(owners, key=_num)

    def rebuilt_by(self, node_id: str) -> list[str]:
        """What a change to `node_id` rebuilds: from the objects it changes in
        place, since forgetting its own outputs leaves its parts in them."""
        hit: set[str] = set()
        for nid in [node_id, *self.receivers(node_id)]:
            hit |= set(self.affected(nid))
        return sorted(hit, key=_num)

    def delete(self, node_id: str) -> list[BuildNode]:
        ids = self.affected(node_id)
        removed = [self.nodes.pop(nid) for nid in ids]
        gone = {v for n in removed for v in n.out_vars}
        for block in self.blocks:
            block.result_names = [n for n in block.result_names if n not in gone]
        self.blocks = [b for b in self.blocks
                       if b.script is not None or self.block_nodes(b.id)]
        return removed

    def set_records(self, node_id: str, records: list) -> None:
        node = self._live(node_id)
        node.records = list(records)
        node.deps = self._deps(node_id, records)

    def block_nodes(self, block_id: str) -> list[BuildNode]:
        return [n for nid, n in sorted(self.nodes.items(), key=lambda kv: _num(kv[0]))
                if n.block == block_id]

    def result_names(self) -> list[str]:
        out: list[str] = []
        for b in self.blocks:
            if b.script is not None:
                continue
            out += b.result_names or [p for n in self.block_nodes(b.id) for p in n.published]
        return out

    def all_out_vars(self) -> list[str]:
        return [v for nid in sorted(self.nodes, key=_num) for v in self.nodes[nid].out_vars]

    def copy(self) -> "BuildGraph":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {"version": 1, "requests": list(self.requests),
                "next_node": self._next_node, "next_block": self._next_block,
                "blocks": [asdict(b) for b in self.blocks],
                "nodes": [asdict(self.nodes[nid]) for nid in sorted(self.nodes, key=_num)]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "BuildGraph":
        g = cls()
        if not d:
            return g
        g.requests = list(d.get("requests") or [])
        g._next_node = d.get("next_node", 1)
        g._next_block = d.get("next_block", 1)
        g.blocks = [Block(**b) for b in d.get("blocks") or []]
        g.nodes = {n["id"]: BuildNode(**n) for n in d.get("nodes") or []}
        return g

    def render(self) -> str:
        lines = []
        for b in self.blocks:
            lines.append(f"# {b.label}")
            if b.script is not None:
                lines.append("    (script stage — not editable)")
                continue
            for n in self.block_nodes(b.id):
                args = ", ".join(f"{k}={json.dumps(v)}" for k, v in n.tool_args.items())
                lhs = f"{', '.join(n.published)} = " if n.published else ""
                deps = f"  <- {', '.join(n.deps)}" if n.deps else ""
                lines.append(f"{n.id:<4}{lhs}{n.tool}({args}){deps}")
        return "\n".join(lines)
