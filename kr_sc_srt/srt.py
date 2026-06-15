from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .timecode import format_srt_time, parse_time


@dataclass(frozen=True)
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str


def parse_srt(content: str) -> list[Cue]:
    blocks = content.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n\n")
    cues: list[Cue] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if len(lines) < 2:
            raise ValueError(f"无效的 SRT 数据块: {block!r}")

        try:
            index = int(lines[0].strip())
            timing = lines[1]
            text_lines = lines[2:]
        except ValueError:
            index = len(cues) + 1
            timing = lines[0]
            text_lines = lines[1:]

        if "-->" not in timing:
            raise ValueError(f"无效的 SRT 时间轴行: {timing!r}")
        start_text, end_text = [part.strip() for part in timing.split("-->", 1)]
        start_ms = parse_time(start_text)
        end_ms = parse_time(end_text)
        if end_ms <= start_ms:
            raise ValueError(f"无效的 SRT 时间轴范围: {timing!r}")
        cues.append(Cue(index=index, start_ms=start_ms, end_ms=end_ms, text="\n".join(text_lines)))
    return renumber(cues)


def read_srt(path: Path) -> list[Cue]:
    return parse_srt(path.read_text(encoding="utf-8-sig"))


def write_srt(path: Path, cues: Iterable[Cue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_srt(cues), encoding="utf-8")


def render_srt(cues: Iterable[Cue]) -> str:
    parts: list[str] = []
    for index, cue in enumerate(cues, start=1):
        parts.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_time(cue.start_ms)} --> {format_srt_time(cue.end_ms)}",
                    cue.text.strip(),
                ]
            )
        )
    return "\n\n".join(parts).strip() + "\n"


def renumber(cues: Iterable[Cue]) -> list[Cue]:
    return [
        Cue(index=index, start_ms=cue.start_ms, end_ms=cue.end_ms, text=cue.text)
        for index, cue in enumerate(cues, start=1)
    ]


def replace_text(cues: Iterable[Cue], texts: Iterable[str]) -> list[Cue]:
    cue_list = list(cues)
    text_list = list(texts)
    if len(cue_list) != len(text_list):
        raise ValueError(f"预期有 {len(cue_list)} 条翻译，但实际收到 {len(text_list)} 条")
    return [
        Cue(index=cue.index, start_ms=cue.start_ms, end_ms=cue.end_ms, text=text.strip())
        for cue, text in zip(cue_list, text_list)
    ]


def crop(cues: Iterable[Cue], start_ms: int, end_ms: int) -> list[Cue]:
    if end_ms <= start_ms:
        raise ValueError("分段结束时间必须在开始时间之后")

    cropped: list[Cue] = []
    for cue in cues:
        overlap_start = max(cue.start_ms, start_ms)
        overlap_end = min(cue.end_ms, end_ms)
        if overlap_end <= overlap_start:
            continue
        cropped.append(
            Cue(
                index=len(cropped) + 1,
                start_ms=overlap_start - start_ms,
                end_ms=overlap_end - start_ms,
                text=cue.text,
            )
        )
    return cropped
