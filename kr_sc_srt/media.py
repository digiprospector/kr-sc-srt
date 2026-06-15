from __future__ import annotations

from pathlib import Path

from . import runner
from .timecode import format_ffmpeg_time


def extract_audio(video: Path, audio: Path, limit_s: int | None = None) -> Path:
    audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vn",
    ]
    if limit_s is not None:
        cmd.extend(["-t", str(limit_s)])
    cmd.extend(["-c:a", "copy", str(audio)])
    
    runner.run(cmd)
    return audio


def probe_duration_ms(media: Path) -> int:
    output = runner.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media),
        ],
        capture=True,
    ).strip()
    return int(float(output) * 1000)


def cut_video(source: Path, target: Path, start_ms: int, end_ms: int) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    runner.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            format_ffmpeg_time(start_ms),
            "-to",
            format_ffmpeg_time(end_ms),
            "-i",
            str(source),
            "-map",
            "0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(target),
        ]
    )
    return target


def burn_subtitles(video: Path, srt: Path, target: Path, font: str = "Noto Sans CJK SC") -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    filter_arg = f"subtitles={_escape_subtitle_path(srt)}:force_style='FontName={font},FontSize=20,Outline=1'"
    runner.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            filter_arg,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(target),
        ]
    )
    return target


def _escape_subtitle_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    return value.replace(":", r"\:").replace("'", r"\'").replace(",", r"\,")
