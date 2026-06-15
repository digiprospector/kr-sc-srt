from __future__ import annotations

from pathlib import Path

from . import runner


LOW_SELECTOR = "worstvideo*+worstaudio/worst"
HIGH_SELECTOR = "bestvideo*+bestaudio/best"


def download_with_ytdlp(source: str, out_dir: Path, stem: str, quality: str, cookies: Path | None = None) -> Path:
    selector = LOW_SELECTOR if quality == "low" else HIGH_SELECTOR
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / f"{stem}.%(ext)s")
    command = [
        "yt-dlp",
        "-f",
        selector,
        "--merge-output-format",
        "mp4",
        "-o",
        template,
        "--print",
        "after_move:filepath",
    ]
    if cookies:
        command.extend(["--cookies", str(cookies)])
    command.append(source)

    output = runner.run(command, capture=True)
    candidates = [Path(line.strip()) for line in output.splitlines() if line.strip()]
    for candidate in reversed(candidates):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    for candidate in sorted(out_dir.glob(f"{stem}.*"), key=lambda path: path.stat().st_mtime, reverse=True):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    raise FileNotFoundError(f"yt-dlp completed but no output was found for {source}")


def resolve_source(source: str, out_dir: Path, quality: str, cookies: Path | None = None) -> Path:
    local = Path(source)
    if local.exists():
        return local.resolve()
    return download_with_ytdlp(source, out_dir, f"source.{quality}", quality, cookies)
