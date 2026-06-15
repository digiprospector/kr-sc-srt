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
        print(f"已写入 {output_srt}")
        return 0

    root = Path(args.root).expanduser().resolve()
    source = args.source
    out_dir = Path(args.out).expanduser().resolve() if args.out else None

    if args.resume_last:
        try:
            last = load_last_job(root)
        except FileNotFoundError as exc:
            parser.error(f"{exc}。请先设置一次 source URL，之后运行便可使用 --resume-last。")
        source = source or last["source"]
        out_dir = out_dir or Path(last["out_dir"]).expanduser().resolve()

    if not source:
        parser.error("除非指定了 --resume-last，否则必须提供 source URL 或本地视频文件")

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
        test=args.test,
    )

    if args.command == "prepare":
        pipeline.prepare(asr_model=args.asr_model, asr_chunk_s=args.asr_chunk_s)
    elif args.command == "render":
        segments = Path(args.segments).expanduser().resolve() if args.segments else pipeline.out_dir / f"{pipeline.job_name}.csv"
        pipeline.render(segments, font=args.font)
    else:
        parser.error(f"不支持的命令: {args.command}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kr-sc-srt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="第一阶段：低画质下载和韩语 ASR 语音识别。")
    _add_common(prepare)
    prepare.add_argument("--asr-model", default=asr.DEFAULT_MODEL, help="FunASR 模型 ID。")
    prepare.add_argument(
        "--asr-chunk-s",
        type=int,
        default=asr.DEFAULT_CHUNK_S,
        help="ASR 语音识别音频分片时长（秒）。",
    )

    render = subparsers.add_parser("render", help="第二阶段：高画质下载、分段并烧录字幕。")
    _add_common(render)
    render.add_argument("--segments", help="CSV 文件路径，包含 name,start,end。默认为 <out>/<job-name>.csv。")
    render.add_argument("--font", default="Noto Sans CJK SC", help="烧录中文硬字幕所使用的字体。")

    translate = subparsers.add_parser("translate-srt", help="本地助手：将韩语 SRT 字幕翻译为中文 SRT 字幕。")
    translate.add_argument("input", help="输入韩语 SRT 路径。")
    translate.add_argument("output", help="输出中文 SRT 路径。")
    translate.add_argument("--api-key", help="兼容 OpenAI 的 API-Key。更推荐使用 --api-key-env。")
    translate.add_argument("--api-key-env", default="OPENAI_API_KEY", help="包含 API-Key 的环境变量名。")
    translate.add_argument("--api-base", default=DEFAULT_API_BASE, help="兼容 OpenAI 的 API 基础 URL。")
    translate.add_argument("--translation-model", default=DEFAULT_MODEL, help="用于韩语转中文翻译的聊天模型。")

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", nargs="?", help="SOOP VOD URL 或本地视频文件。")
    parser.add_argument("--root", default=str(default_root()), help="持久化根目录。")
    parser.add_argument("--out", help="任务输出目录。")
    parser.add_argument("--model-cache-dir", help="持久化 FunASR/模型缓存目录。")
    parser.add_argument("--cookies", help="可选，用于 yt-dlp 的 cookies.txt 路径。")
    parser.add_argument("--resume-last", action="store_true", help="复用根目录中上一次的 URL 和输出目录。")
    parser.add_argument("--force-stage", action="append", help="强制重新运行某个阶段。可多次指定。")
    parser.add_argument("--force-all", action="store_true", help="重新运行所有阶段。")
    parser.add_argument("--test", action="store_true", help="测试模式：仅提取前 10 分钟的音频进行处理。")


if __name__ == "__main__":
    raise SystemExit(main())
