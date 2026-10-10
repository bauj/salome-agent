import json

from salome_agent.build_graph import GraphError, _num
from salome_agent.modules import ModuleSpec, Terminator
from salome_agent.replay import ReplayFailed, apply_delete, apply_edit
from salome_agent.modules import spec_for
from salome_agent.specialists.geom import _compact_result
from salome_agent.specialists.tool_specialist import ToolSpecialist

AMEND_SYSTEM_PROMPT = """You amend a CAD model the user is building with you in
a live SALOME study. The model is a build graph: each node is one tool call,
shown as `id  name = tool(args)  <- the nodes it uses`. Study names are the
handles; node ids are what edit_arg and delete_node take.

To change something that exists, call edit_arg on the node that made it —
"make the roof taller" is an edit of the roof node's height argument, not a
new shape. The node and everything built on it is rebuilt immediately with
the same names, and you get the rebuilt shapes' measurements back. edit_arg
only changes literal values. To change which shapes a node uses, delete_node
it and build the replacement with the {module} tools.

To add something, call the {module} tools as usual; they are added to the graph.
Measure before you place anything relative to an existing shape. Lengths are
millimetres; angles are degrees.

Meshes are part of the model. Editing a shape rebuilds the SMESH nodes built
on it. To mesh or remesh, use the SMESH tools; if they are not in your list,
call load_tools with the category smesh.

Never call delete_node unless the user asked to remove or replace something.
A request for a new object ("make me a prism") is added alongside what is
already there, not swapped in. If you cannot tell whether they want to
add or replace, change nothing and ask.

Do only what the user asked. If the request could mean two different nodes,
change nothing and call finish_turn with your question in `question`. If the
request is too big to express as edits plus a few new shapes (a whole new
assembly, a different object), call replan before changing anything.

Each turn: state your reasoning in at most 2-3 sentences, then call one or
more tools. When the change is done, call finish_turn(summary) with one
sentence saying what changed.

You are given only a subset of the {module} and SMESH tools. If you need one
that is not in your list, call load_tools with its name or category."""

_LITERAL_HELP = ('the new value as a JSON literal, e.g. 120, 7.5, [0, 0, 1] or "OZ"')

EDIT_ARG_DEF = {
    "name": "edit_arg", "category": "amend",
    "description": ("Change one literal argument of an existing node; the node and "
                    "everything built on it are rebuilt with the same names."),
    "parameters": {"type": "object", "properties": {
        "node": {"type": "string", "description": "node id, e.g. n2"},
        "param": {"type": "string", "description": "argument name as shown in the build graph"},
        "value": {"type": "string", "description": _LITERAL_HELP}},
        "required": ["node", "param", "value"]}}

DELETE_NODE_DEF = {
    "name": "delete_node", "category": "amend",
    "description": "Remove a node and every node built on it, from the model and the study.",
    "parameters": {"type": "object", "properties": {
        "node": {"type": "string", "description": "node id, e.g. n4"}},
        "required": ["node"]}}

FINISH_TURN = Terminator(
    name="finish_turn",
    schema={"type": "object", "properties": {
        "summary": {"type": "string", "description": "one sentence: what changed"},
        "question": {"type": "string",
                     "description": "only if you changed nothing because the request was "
                                    "ambiguous: your question to the user"}},
        "required": ["summary"]},
    description="End this turn.",
    handle=lambda args, call_log: {"summary": args.get("summary") or "",
                                   "question": args.get("question") or None})

REPLAN = Terminator(
    name="replan",
    schema={"type": "object", "properties": {"reason": {"type": "string"}},
            "required": ["reason"]},
    description=("Hand this request to the planner because it is too big for edits. "
                 "Call it before changing anything."),
    handle=lambda args, call_log: {"replan": args.get("reason") or ""})


def _literal(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _call(client, module: str, name: str, args: dict) -> dict:
    if module == "geom":
        return client.call_tool(name, args)
    return client.call_tool(name, args, module=module)


class AmendSession:
    """`module` is the CAD module whose tools this turn adds shapes with; the
    SMESH tools come with it."""

    def __init__(self, graph, client, turn: int, module: str = "geom"):
        self.graph = graph
        self.client = client
        self.turn = turn
        self.module = module
        self.events: list[dict] = []
        self._blocks: dict[str, str] = {}
        # tool name -> the module that runs it (tools_for fills it).
        self.tool_module: dict[str, str] = {}
        self.meshed = any(n.module == "smesh" for n in graph.nodes.values())
        # The SMESH how-to went out (system prompt, or the first SMESH load_tools).
        self.smesh_told = self.meshed
        self._last_module = module

    def _amend_block(self, module: str) -> str:
        """One block per module per turn; the CAD block goes before the SMESH
        one, since a mesh uses the CAD shapes and never the other way round."""
        if module not in self._blocks:
            before = self._blocks.get("smesh") if module != "smesh" else None
            label = f"turn {self.turn + 1}: amend"
            if module == "smesh":
                label += " (SMESH)"
            self._blocks[module] = self.graph.add_block(self.turn, label, module, [],
                                                        before=before)
        return self._blocks[module]

    def _measure(self, ids) -> dict | None:
        by_module: dict[str, list] = {}
        for nid in ids:
            if nid in self.graph.nodes:
                node = self.graph.nodes[nid]
                by_module.setdefault(node.module, []).extend(node.published)
        out = {}
        for module, names in by_module.items():
            if names:
                m = _call(self.client, module, "measure", {"names": names})
                if m.get("ok"):
                    out.update(m.get("measurements") or {})
        return out or None

    def execute(self, client, name: str, args: dict) -> dict:
        self._last_module = self.module
        try:
            if name == "edit_arg":
                node, param = args["node"], args["param"]
                value = _literal(args["value"])
                old = self.graph.nodes[node].tool_args.get(param) if node in self.graph.nodes else None
                self.graph, ids = apply_edit(self.client, self.graph, node, param, value)
                self.events += [{"event": "graph_edit", "op": "edit", "node": node,
                                 "param": param, "old": old, "new": value},
                                {"event": "replay", "nodes": ids}]
                return {"ok": True, "changed": True, "edited": node, "rebuilt": ids,
                        "measurements": self._measure(ids)}
            if name == "delete_node":
                before = self.graph
                self.graph, ids = apply_delete(self.client, self.graph, args["node"])
                self.events.append({"event": "graph_edit", "op": "delete", "node": args["node"],
                                    "param": None, "old": ids, "new": None})
                out = {"ok": True, "changed": True, "deleted": ids}
                note = _other_module_note(before, args["node"], ids)
                if note:
                    out["note"] = note
                return out
        except GraphError as e:
            return {"ok": False, "error": str(e)}
        except ReplayFailed as e:
            return {"ok": False, "error": f"{e} — the model was restored to before this edit"}
        module = self._last_module = self.tool_module.get(name, self.module)
        r = _call(self.client, module, name, args)
        if r.get("ok") and r.get("records") and not self._recompute(module, name, args):
            bid = self._amend_block(module)
            nid = self.graph.append(bid, name, args, r["records"], self.turn, module=module)
            if module == "smesh" and name == "create_mesh":
                # SMESH publishes nothing: a mesh reaches result.py as a result name
                # of its block, as finish_stage gives a planned SMESH stage.
                block = next(b for b in self.graph.blocks if b.id == bid)
                block.result_names += self.graph.nodes[nid].out_vars
            self.events.append({"event": "graph_edit", "op": "append", "node": nid,
                                "param": None, "old": None, "new": name})
            r = {**r, "node": nid}
        return r


    def _recompute(self, module: str, name: str, args: dict) -> bool:
        """A compute of a mesh nothing touched since its last compute: it changes
        nothing, so it stays out of the graph and result.py."""
        if module != "smesh" or name != "compute":
            return False
        mesh = args.get("mesh")
        touched = [n for n in self.graph.nodes.values()
                   if mesh in n.out_vars or any(r.get("recv") == mesh for r in n.records)]
        return bool(touched) and max(touched, key=lambda n: _num(n.id)).tool == "compute"

    def render(self, result: dict) -> str:
        if self._last_module == "smesh":
            return spec_for("smesh").render_result(result)
        return _render(result) if self.module == "geom" else _render_shaper(result)

    def load_note(self, added) -> str | None:
        """The SMESH how-to, once, with the first load_tools that brings SMESH tools."""
        if self.smesh_told or not any(self.tool_module.get(n) == "smesh" for n in added):
            return None
        self.smesh_told = True
        return "\n\n" + _smesh_howto()


def _other_module_note(before, node_id: str, ids) -> str | None:
    """Says which nodes of another module a delete took with it, by their names."""
    module = before.nodes[node_id].module
    other = [before.nodes[i] for i in ids if before.nodes[i].module != module]
    if not other:
        return None
    names = list(dict.fromkeys(p for n in other for p in (n.published or n.out_vars)))
    mods = " and ".join(dict.fromkeys(n.module.upper() for n in other))
    named = f" ({', '.join(names)})" if names else ""
    return (f"This also removed the {mods} nodes {', '.join(n.id for n in other)}{named}, "
            f"which were built on what you deleted. Rebuild them on the new shape with "
            f"the {mods} tools unless the user asked to remove them.")


def _smesh_howto() -> str:
    return ("How the SMESH tools work (in this turn finish_turn ends it, "
            "not finish_stage):\n\n" + spec_for("smesh").prompt("final"))


_AMEND_KEYS = ("node", "edited", "rebuilt", "deleted", "note", "measurements")


def _render(result: dict) -> str:
    out = _compact_result(result)
    for k in _AMEND_KEYS:
        if result.get(k) is not None:
            out[k] = result[k]
    return str(out)


def _render_shaper(result: dict) -> str:
    hidden = {"records", "result_shape", "shape_info"}
    out = {k: v for k, v in result.items() if k not in hidden}
    return str(out)


def render_transcript(history) -> str:
    turns = [e for e in history if e.get("event") == "turn"][-10:]
    return "\n".join(f"turn {e['turn'] + 1} ({e.get('status')}): {e.get('text')} -> "
                     f"{e.get('summary') or ''}" for e in turns)


def _snapshot(client, graph) -> str | None:
    if client is None:
        return None
    from salome_agent.specialists.shaper import _upstream_line
    lines = []
    for b in graph.blocks:
        names = b.result_names or [p for n in graph.block_nodes(b.id) for p in n.published]
        if not names or b.module not in ("geom", "shaper"):
            continue
        m = _call(client, b.module, "measure", {"names": names})
        if not m.get("ok"):
            continue
        for n, d in (m.get("measurements") or {}).items():
            if not d or d.get("concealed"):
                continue
            if b.module == "geom":
                lines.append(f"  {n} — bbox {d['bbox']}, size {d['size']}, centre {d['centre']}")
            else:
                lines.append(_upstream_line(n, d))
    return "Current study, measured now:\n" + "\n".join(lines) if lines else None


def _prompt(module: str, cadence: str, meshed: bool = False) -> str:
    text = AMEND_SYSTEM_PROMPT.format(module=module)
    if module == "shaper":
        text += ("\n\nHow the SHAPER tools work (in this turn finish_turn ends it, "
                 "not finish_stage):\n\n" + spec_for("shaper").prompt(cadence))
    if meshed:
        text += "\n\n" + _smesh_howto()
    return text


def _spec(session: AmendSession) -> ModuleSpec:
    base = spec_for(session.module)
    smesh = spec_for("smesh")

    def tools_for(client):
        defs = dict(base.tools_for(client))
        session.tool_module = {n: session.module for n in defs}
        # The CAD module keeps a name both have (get_subshapes: SMESH's hands
        # CAD shapes to the CAD module's own). Core SMESH tools load as the
        # category "smesh", the rest by name.
        for n, d in smesh.tools_for(client).items():
            if n in defs:
                continue
            cat = "smesh" if n in smesh.core_tools else f"smesh_{d.get('category')}"
            defs[n] = {**d, "category": cat}
            session.tool_module[n] = "smesh"
        defs["edit_arg"] = EDIT_ARG_DEF
        defs["delete_node"] = DELETE_NODE_DEF
        return defs

    core = base.core_tools | (smesh.core_tools if session.meshed else frozenset())
    return ModuleSpec(
        name="amend",
        prompt=lambda cadence: _prompt(session.module, cadence, session.meshed),
        terminator=FINISH_TURN,
        extra_terminators=(REPLAN,),
        execute=session.execute,
        render_result=session.render,
        tools_for=tools_for,
        load_note=session.load_note,
        no_tool_nudge="You did not call a tool. Call a tool now — finish_turn when done.",
        on_exhausted=lambda call_log, last_error: {"exhausted": True, "last_error": last_error},
        core_tools=core,
        categories=base.categories + ("smesh",),
        risky_categories=base.risky_categories,
        measure_tools=base.measure_tools,
    )


def run_amend(text: str, graph, llm, client, turn: int, policy: dict, *,
              history=(), selection=None, session=None, cancel=None,
              image_description=None):
    session = session or AmendSession(graph, client, turn)
    blocks = ["Build graph:\n" + graph.render()]
    snap = _snapshot(client, graph)
    if snap:
        blocks.append(snap)
    transcript = render_transcript(history)
    if transcript:
        blocks.append("Conversation so far:\n" + transcript)
    if selection:
        blocks.append("The user currently has these selected in the viewer:\n"
                      + "\n".join(json.dumps(s) for s in selection))
    if image_description:
        blocks.append("The user attached reference image(s) to this message; the "
                      "image-analysis agent describes them as:\n" + image_description)
    blocks.append(f"The user says: {text}")
    stage_plan = {"module": "amend", "intent": text, "tools": ["edit_arg", "delete_node"]}
    outcome = ToolSpecialist(_spec(session)).run(
        stage_plan, llm, client, policy["tool_iteration_cap"], blocks=tuple(blocks),
        tool_selection=policy.get("tool_selection", "hints"),
        verify_cadence=policy.get("verify_cadence", "final"), cancel=cancel)
    return session, outcome
