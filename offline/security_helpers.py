"""Pure filesystem and access-log safety helpers (no application imports)."""
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit


def confined_path(root, relative):
    """Resolve a relative name inside root; reject escapes and absolute names."""
    relative = relative.replace("\\", "/")
    if relative.startswith("/") or PureWindowsPath(relative).drive:
        raise ValueError("absolute path is not allowed")
    base = Path(root).resolve()
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError("path escapes document root") from None
    return str(candidate)


def safe_log_target(target):
    """Do not persist URL query/fragment values or terminal control characters."""
    try:
        path = urlsplit(target).path
    except ValueError:
        return "[invalid request target]"
    return "".join(c if c.isprintable() else "?" for c in path)
