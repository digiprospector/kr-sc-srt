from __future__ import annotations

import hashlib
import re
from pathlib import Path


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def default_root() -> Path:
    return Path.cwd() / "kr-sc-srt-work"


def job_name_from_source(source: str) -> str:
    local = Path(source)
    if local.exists():
        stem = local.stem
    else:
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
        stem = source.rstrip("/").split("/")[-1] or "job"
        stem = f"{stem}-{digest}"
    cleaned = _SAFE_RE.sub("_", stem).strip("._-")
    return cleaned or "job"
