from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from . import media
from .srt import Cue, write_srt


DEFAULT_MODEL = "iic/SenseVoiceSmall"
_LANG_TAG_RE = re.compile(r"<\|[^>]+?\|>")


def transcribe_to_srt(
    audio: Path,
    output_srt: Path,
    model_cache_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    language: str = "ko",
) -> Path:
    if model_cache_dir:
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MODELSCOPE_CACHE", str(model_cache_dir))
        os.environ.setdefault("HF_HOME", str(model_cache_dir / "huggingface"))

    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError("FunASR is not installed. Install requirements.txt first.") from exc

    model = AutoModel(model=model_name, trust_remote_code=True)
    result = model.generate(
        input=str(audio),
        language=language,
        use_itn=True,
        sentence_timestamp=True,
        merge_vad=True,
        merge_length_s=15,
    )
    cues = _result_to_cues(result, audio)
    write_srt(output_srt, cues)
    return output_srt


def _result_to_cues(result: Any, audio: Path) -> list[Cue]:
    items = result if isinstance(result, list) else [result]
    cues: list[Cue] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sentence_info = item.get("sentence_info") or item.get("sentences") or []
        for sentence in sentence_info:
            text = _clean_text(str(sentence.get("text", "")))
            if not text:
                continue
            start = int(float(sentence.get("start", 0)))
            end = int(float(sentence.get("end", 0)))
            if end > start:
                cues.append(Cue(index=len(cues) + 1, start_ms=start, end_ms=end, text=text))

        if not cues and item.get("text"):
            text = _clean_text(str(item["text"]))
            if text:
                duration = media.probe_duration_ms(audio)
                cues.append(Cue(index=1, start_ms=0, end_ms=max(1000, duration), text=text))

    if not cues:
        raise RuntimeError("FunASR returned no transcribable text")
    return cues


def _clean_text(text: str) -> str:
    return _LANG_TAG_RE.sub("", text).strip()
