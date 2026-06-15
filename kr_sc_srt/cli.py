from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import asr
from .jobs import default_root
from .pipeline import Pipeline
from .srt import read_srt, write_srt
from .state import load_last_job
from .translate import DEFAULT_API_BASE, DEFAULT_MODEL, translate_cues


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "translate-srt":
        source_srt = Path(args.input).expanduser().resolve()
        output_srt = Path(args.output).expanduser().resolve()
        translated = translate_cues(
            read_srt(source_srt),
            api_key=args.api_key or os.environ.get(args.api_key_env),
            api_base=args.api_base,
            model=args.translation_model,
        )
        write_srt(output_srt, translated)
        print(f"wrote {output_srt}")
        return 0

    root = Path(args.root).expanduser().resolve()
    source = args.source
    out_dir = Path(args.out).expanduser().resolve() if args.out else None

    if args.resume_last:
        try:
            last = load_last_job(root)
        except FileNotFoundError as exc:
            parser.error(f"{exc}. Set the source URL once, then use --resume-last on later runs.")
        source = source or last["source"]
        out_dir = out_dir or Path(last["out_dir"]).expanduser().resolve()

    if not source:
        parser.error("source URL/file is required unless --resume-last is used")

    model_cache_dir = Path(args.model_cache_dir).expanduser().resolve() if args.model_cache_dir else None
    cookies = Path(args.cookies).expanduser().resolve() if args.cookies else None
    pipeline = Pipeline(
        root=root,
        source=source,
        out_dir=out_dir,
        model_cache_dir=model_cache_dir,
        cookies=cookies,
        force_all=args.force_all,
        force_stage=set(args.force_stage or []),
    )

    if args.command == "prepare":
        pipeline.prepare(asr_model=args.asr_model)
    elif args.command == "render":
        segments = Path(args.segments).expanduser().resolve() if args.segments else pipeline.out_dir / f"{pipeline.job_name}.csv"
        pipeline.render(segments, font=args.font)
    else:
        parser.error(f"Unsupported command: {args.command}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kr-sc-srt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="First pass: low-res download and Korean ASR.")
    _add_common(prepare)
    prepare.add_argument("--asr-model", default=asr.DEFAULT_MODEL, help="FunASR model id.")

    render = subparsers.add_parser("render", help="Second pass: high-res download, split, and burn subtitles.")
    _add_common(render)
    render.add_argument("--segments", help="CSV file with name,start,end rows. Defaults to <out>/<job-name>.csv.")
    render.add_argument("--font", default="Noto Sans CJK SC", help="Font used for burned Chinese subtitles.")

    translate = subparsers.add_parser("translate-srt", help="Local helper: translate Korean SRT into Chinese SRT.")
    translate.add_argument("input", help="Input Korean SRT path.")
    translate.add_argument("output", help="Output Chinese SRT path.")
    translate.add_argument("--api-key", help="OpenAI-compatible API key. Prefer --api-key-env.")
    translate.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the API key.")
    translate.add_argument("--api-base", default=DEFAULT_API_BASE, help="OpenAI-compatible API base URL.")
    translate.add_argument("--translation-model", default=DEFAULT_MODEL, help="Chat model for Korean-to-Chinese translation.")

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", nargs="?", help="SOOP VOD URL or local video file.")
    parser.add_argument("--root", default=str(default_root()), help="Persistent root directory.")
    parser.add_argument("--out", help="Job output directory.")
    parser.add_argument("--model-cache-dir", help="Persistent FunASR/model cache directory.")
    parser.add_argument("--cookies", help="Optional cookies.txt for yt-dlp.")
    parser.add_argument("--resume-last", action="store_true", help="Reuse the last URL and output directory from root.")
    parser.add_argument("--force-stage", action="append", help="Force one stage to rerun. Can be repeated.")
    parser.add_argument("--force-all", action="store_true", help="Rerun all stages.")


if __name__ == "__main__":
    raise SystemExit(main())
