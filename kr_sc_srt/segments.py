from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .timecode import parse_time


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Segment:
    name: str
    safe_name: str
    start_ms: int
    end_ms: int


def safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value.strip()).strip("._-")
    return cleaned or "segment"


def read_segments(path: Path) -> list[Segment]:
    if not path.exists():
        raise FileNotFoundError(f"Segments CSV not found: {path}")

    segments: list[Segment] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if row[0].strip().lower() == "name":
                continue
            if len(row) != 3:
                raise ValueError(f"{path}:{line_no}: expected name,start,end")

            name, start_text, end_text = [cell.strip() for cell in row]
            if not name:
                raise ValueError(f"{path}:{line_no}: segment name is required")
            start_ms = parse_time(start_text)
            end_ms = parse_time(end_text)
            if end_ms <= start_ms:
                raise ValueError(f"{path}:{line_no}: end must be after start")

            base_safe = safe_name(name)
            candidate = base_safe
            suffix = 2
            while candidate in seen:
                candidate = f"{base_safe}_{suffix}"
                suffix += 1
            seen.add(candidate)
            segments.append(Segment(name=name, safe_name=candidate, start_ms=start_ms, end_ms=end_ms))

    if not segments:
        raise ValueError(f"No segments found in {path}")
    return segments
