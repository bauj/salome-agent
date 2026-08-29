import glob
import os
import re
from typing import Optional

_PORT_RE = re.compile(r"NameService=corbaname::[\w.\-]+:(\d+)")


def discover_port(user_path: Optional[str] = None) -> Optional[int]:
    base = user_path or os.environ.get("OMNIORB_USER_PATH", "/tmp")
    for path in glob.glob(os.path.join(base, ".omniORB_*_*.cfg")):
        with open(path) as f:
            contents = f.read()
        m = _PORT_RE.search(contents)
        if m:
            return int(m.group(1))
    return None
