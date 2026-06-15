from __future__ import annotations

import shutil
from pathlib import Path

from . import runner

WHISPER_BINARY = "/content/drive/MyDrive/Faster-Whisper-XXL/faster-whisper-xxl"
DEFAULT_MODEL = "large-v2"


def transcribe_to_srt(
    audio: Path,
    output_srt: Path,
    model_name: str = DEFAULT_MODEL,
) -> Path:
    """使用 Faster-Whisper-XXL 将音频 *audio* 转录为 SRT 字幕文件。

    Whisper 二进制直接输出 SRT，无需在 Python 端进行时间戳解析或句度分割。
    """
    output_dir = output_srt.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        WHISPER_BINARY,
        str(audio),
        "-m", model_name,
        "-l", "Korean",
        "--vad_method", "pyannote_v3",
        "--ff_vocal_extract", "mdx_kim2",
        "--sentence",
        "-v", "true",
        "-o", str(output_dir),
        "-f", "srt",
    ]

    print(f"[asr] 正在运行 Whisper: 模型={model_name}", flush=True)
    runner.run(cmd)

    # Whisper 输出文件名基于输入文件的 stem（如 198391511.aac → 198391511.srt）
    generated = output_dir / f"{audio.stem}.srt"
    if generated.resolve() != output_srt.resolve():
        if generated.exists():
            shutil.move(str(generated), str(output_srt))
        else:
            raise FileNotFoundError(
                f"Whisper 未在预期位置生成 SRT 文件: {generated}"
            )

    if not output_srt.exists():
        raise FileNotFoundError(
            f"Whisper 未生成预期的 SRT 文件: {output_srt}"
        )

    print(f"[asr] 韩语字幕已生成: {output_srt}", flush=True)
    return output_srt
