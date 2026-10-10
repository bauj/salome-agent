"""How long the server lets each kind of work run before it gives up. The
client reads these too, so it waits for a reply at least as long as the
server may legitimately take to send one."""

TOOL_TIMEOUT = 120
MAIN_THREAD_TIMEOUT = 30
REFRESH_TIMEOUT = 10
# Tool modules besides GEOM; they run on the GUI main thread (core.py).
SIDE_MODULES = ("shaper", "smesh")
