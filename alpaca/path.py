from __future__ import annotations
import pathlib
import os
def path_helper(path: str) -> pathlib.Path:
    """Helper to resolve paths."""
    if home := os.environ.get("ALPACA_HOME"):
        p = pathlib.Path(home) / path
    elif home := os.environ.get("XDG_DATA_HOME"):
        p = pathlib.Path(home) / path
    else:
        p = pathlib.Path("~/.local/share/alpaca") / path
    if not p.exists():
        p.expanduser().mkdir(parents=True, exist_ok=True)
    return p.expanduser()
