from __future__ import annotations

import re


_TIME_RE = re.compile(
    r"^\s*(?:(?P<hours>\d{1,2}):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2})(?P<fraction>[,.]\d{1,3})?\s*$"
)


def parse_time(value: str) -> int:
    """Parse HH:MM:SS.mmm or MM:SS into milliseconds."""
    match = _TIME_RE.match(value)
    if not match:
        raise ValueError(f"Invalid time value: {value!r}")

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    fraction = match.group("fraction")
    millis = 0
    if fraction:
        millis = int(fraction[1:].ljust(3, "0")[:3])

    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid time value: {value!r}")

    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def format_srt_time(milliseconds: int) -> str:
    milliseconds = max(0, int(round(milliseconds)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def format_ffmpeg_time(milliseconds: int) -> str:
    milliseconds = max(0, int(round(milliseconds)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
