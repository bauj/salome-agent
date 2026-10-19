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
  /image PATH[,PATH] [request]
             a request with reference image(s); quote a path with spaces
  /quit      leave (Ctrl-D too)
anything else is a request to the agent. Ctrl-C cancels the running turn."""


IMAGE_ONLY_REQUEST = "Build what the attached image(s) show."
_IMAGE_USAGE = "usage: /image PATH[,PATH] [request] — quote a path with spaces"


def _image_turn(rest: str) -> tuple[str, list[str]]:
    """`/image` arguments as (request, data: URIs); ValueError / OSError on a
    missing or unreadable image, as --images gives."""
    from salome_agent import cli

    rest = rest.strip()
    if rest[:1] in ("'", '"'):
        end = rest.find(rest[0], 1)
        if end < 0:
            raise ValueError(f"no closing {rest[0]} — {_IMAGE_USAGE}")
        raw, text = rest[1:end], rest[end + 1:]
    else:
        raw, _, text = rest.partition(" ")
    paths = [os.path.expanduser(p) for p in cli.parse_image_paths(raw)]
    if not paths:
        raise ValueError(_IMAGE_USAGE)
    return text.strip() or IMAGE_ONLY_REQUEST, cli.load_reference_images(paths)


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


def run_turn(conv, text, gate, stream, images=None):
    cancel = threading.Event()

    def on_sigint(signum, frame):
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()
        stream.write("\n(cancelling this turn — Ctrl-C again to quit)\n")

    prev = signal.signal(signal.SIGINT, on_sigint)
    try:
        return conv.turn(text, images=images, on_event=_print_event(stream), on_gate=gate,
                         cancel=cancel)
    finally:
        signal.signal(signal.SIGINT, prev)


def _say(stream, result):
    stream.write(f"[{result.status}] {result.message}\n")
    stream.flush()


def repl(conv, first_request, *, images=None, read=input, stream=sys.stderr,
         gate=terminal_gate, run_turn=run_turn) -> int:
    try:
        import readline  # noqa: F401 — history and line editing for input()
    except ImportError:
        pass
    _say(stream, run_turn(conv, first_request, gate, stream, images=images))
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
        elif cmd == "/image":
            try:
                text, uris = _image_turn(line[len(cmd):])
            except (ValueError, OSError) as e:
                stream.write(f"{str(e).removeprefix('--images: ')}\n")
            else:
                _say(stream, run_turn(conv, text, gate, stream, images=uris))
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
