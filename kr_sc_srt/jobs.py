from __future__ import annotations

import re
from pathlib import Path


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def default_root() -> Path:
    return Path.cwd() / "kr-sc-srt-work"


def job_name_from_source(source: str) -> str:
    local = Path(source)
    if local.exists():
        return local.stem

    url_path = source.split("?")[0].split("#")[0]
    parts = url_path.rstrip("/").split("/")
    if parts:
        last_part = parts[-1]
        match = re.search(r"\d+", last_part)
        if match:
            return match.group(0)
        return _SAFE_RE.sub("_", last_part).strip("._-") or "job"
    return "job"
