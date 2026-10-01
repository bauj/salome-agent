import os
import signal
import sys
import threading

from salome_agent.interactive import terminal_gate

_HELP = """commands:
  /undo      revert the last turn
  /redo      re-apply the last undone turn
  /show      print the build graph
  /result    print result.py
  /rebuild   rebuild the whole study from the model (after a failed rollback)
  /quit      leave (Ctrl-D too)
anything else is a request to the agent. Ctrl-C cancels the running turn."""


def _print_event(stream):
    def on_event(ev):
        d = ev["data"]
        if ev["kind"] == "graph_edit":
            if d["op"] == "edit":
                stream.write(f"  edit {d['node']}.{d['param']}: {d['old']} -> {d['new']}\n")
            elif d["op"] == "delete":
                stream.write(f"  deleted {', '.join(d['old'])}\n")
            else:
                stream.write(f"  added {d['node']} ({d['new']})\n")
        elif ev["kind"] == "replay":
            stream.write(f"  rebuilding {len(d['nodes'])} object(s): {', '.join(d['nodes'])}\n")
        stream.flush()
    return on_event


def run_turn(conv, text, gate, stream):
    cancel = threading.Event()

    def on_sigint(signum, frame):
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()
        stream.write("\n(cancelling this turn — Ctrl-C again to quit)\n")

    prev = signal.signal(signal.SIGINT, on_sigint)
    try:
        return conv.turn(text, on_event=_print_event(stream), on_gate=gate, cancel=cancel)
    finally:
        signal.signal(signal.SIGINT, prev)


def _say(stream, result):
    stream.write(f"[{result.status}] {result.message}\n")
    stream.flush()


def repl(conv, first_request, *, read=input, stream=sys.stderr, gate=terminal_gate,
         run_turn=run_turn) -> int:
    try:
        import readline  # noqa: F401 — history and line editing for input()
    except ImportError:
        pass
    _say(stream, run_turn(conv, first_request, gate, stream))
    stream.write(_HELP + "\n")
    while not conv.ended:
        try:
            line = read("chat> ").strip()
        except (EOFError, KeyboardInterrupt):
            stream.write("\n")
            return 0
        if not line:
            continue
        if not line.startswith("/"):
            _say(stream, run_turn(conv, line, gate, stream))
            continue
        cmd = line.split()[0]
        if cmd == "/quit":
            return 0
        if cmd == "/help":
            stream.write(_HELP + "\n")
        elif cmd == "/show":
            stream.write(conv.graph.render() + "\n")
        elif cmd == "/undo":
            _say(stream, conv.undo())
        elif cmd == "/redo":
            _say(stream, conv.redo())
        elif cmd == "/rebuild":
            _say(stream, conv.rebuild())
        elif cmd == "/result":
            path = os.path.join(conv.run_dir, "result.py")
            stream.write(f"{path}\n")
            if os.path.exists(path):
                with open(path) as f:
                    stream.write(f.read())
        else:
            stream.write(f"unknown command {cmd} — /help\n")
        stream.flush()
    return 1
