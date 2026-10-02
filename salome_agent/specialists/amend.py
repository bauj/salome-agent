import json

from salome_agent.build_graph import GraphError
from salome_agent.modules import ModuleSpec, Terminator
from salome_agent.replay import ReplayFailed, apply_delete, apply_edit
from salome_agent.specialists.geom import _compact_result, _geom_tools_for
from salome_agent.specialists.tool_selection import CORE_TOOLS
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
it and build the replacement with the geom tools.

To add something, call the geom tools as usual; they are added to the graph.
Measure before you place anything relative to an existing shape. Lengths are
millimetres; angles are degrees.

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

You are given only a subset of the geom tools. If you need one that is not in
your list, call load_tools with its name or category."""

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


class AmendSession:
    def __init__(self, graph, client, turn: int):
        self.graph = graph
        self.client = client
        self.turn = turn
        self.events: list[dict] = []
        self._block = None

    def _amend_block(self) -> str:
        if self._block is None:
            self._block = self.graph.add_block(self.turn, f"turn {self.turn + 1}: amend",
                                               "geom", [])
        return self._block

    def _measure(self, ids) -> dict | None:
        names = [p for nid in ids if nid in self.graph.nodes
                 for p in self.graph.nodes[nid].published]
        if not names:
            return None
        m = self.client.call_tool("measure", {"names": names})
        return m.get("measurements") if m.get("ok") else None

    def execute(self, client, name: str, args: dict) -> dict:
        try:
            if name == "edit_arg":
                node, param = args["node"], args["param"]
                value = _literal(args["value"])
                old = self.graph.nodes[node].tool_args.get(param) if node in self.graph.nodes else None
                self.graph, ids = apply_edit(self.client, self.graph, node, param, value)
                self.events += [{"event": "graph_edit", "op": "edit", "node": node,
                                 "param": param, "old": old, "new": value},
                                {"event": "replay", "nodes": ids}]
                return {"ok": True, "edited": node, "rebuilt": ids,
                        "measurements": self._measure(ids)}
            if name == "delete_node":
                self.graph, ids = apply_delete(self.client, self.graph, args["node"])
                self.events.append({"event": "graph_edit", "op": "delete", "node": args["node"],
                                    "param": None, "old": ids, "new": None})
                return {"ok": True, "deleted": ids}
        except GraphError as e:
            return {"ok": False, "error": str(e)}
        except ReplayFailed as e:
            return {"ok": False, "error": f"{e} — the model was restored to before this edit"}
        r = self.client.call_tool(name, args)
        if r.get("ok") and r.get("records"):
            nid = self.graph.append(self._amend_block(), name, args, r["records"], self.turn)
            self.events.append({"event": "graph_edit", "op": "append", "node": nid,
                                "param": None, "old": None, "new": name})
            r = {**r, "node": nid}
        return r


def _render(result: dict) -> str:
    out = _compact_result(result)
    for k in ("node", "edited", "rebuilt", "deleted", "measurements"):
        if result.get(k) is not None:
            out[k] = result[k]
    return str(out)


def render_transcript(history) -> str:
    turns = [e for e in history if e.get("event") == "turn"][-10:]
    return "\n".join(f"turn {e['turn'] + 1} ({e.get('status')}): {e.get('text')} -> "
                     f"{e.get('summary') or ''}" for e in turns)


def _snapshot(client, graph) -> str | None:
    names = graph.result_names()
    if not names or client is None:
        return None
    m = client.call_tool("measure", {"names": names})
    if not m.get("ok"):
        return None
    lines = [f"  {n} — bbox {d['bbox']}, size {d['size']}, centre {d['centre']}"
             for n, d in (m.get("measurements") or {}).items() if d]
    return "Current study, measured now:\n" + "\n".join(lines) if lines else None


def _spec(session: AmendSession) -> ModuleSpec:
    def tools_for(client):
        defs = _geom_tools_for(client)
        defs["edit_arg"] = EDIT_ARG_DEF
        defs["delete_node"] = DELETE_NODE_DEF
        return defs

    return ModuleSpec(
        name="amend",
        prompt=lambda cadence: AMEND_SYSTEM_PROMPT,
        terminator=FINISH_TURN,
        extra_terminators=(REPLAN,),
        execute=session.execute,
        render_result=_render,
        tools_for=tools_for,
        no_tool_nudge="You did not call a tool. Call a tool now — finish_turn when done.",
        on_exhausted=lambda call_log, last_error: {"exhausted": True, "last_error": last_error},
        core_tools=frozenset(CORE_TOOLS),
        categories=("primitive", "sketch", "swept", "boolean",
                    "transform", "feature", "display", "query"),
        risky_categories=frozenset({"boolean", "transform"}),
        measure_tools=frozenset({"measure", "check_shape"}),
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
