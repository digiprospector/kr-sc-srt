from __future__ import annotations

import gc
import os
import re
import shutil
from pathlib import Path
from typing import Any

from . import media
from .srt import Cue, renumber, write_srt

DEFAULT_MODEL = "iic/SenseVoiceSmall"
_LANG_TAG_RE = re.compile(r"<\|[^>]+?\|>")

# 30 seconds per chunk – small enough to avoid OOM, large enough for good
# sentence boundary detection with merge_vad.
DEFAULT_CHUNK_S = 30


def transcribe_to_srt(
    audio: Path,
    output_srt: Path,
    model_cache_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    language: str = "ko",
    chunk_duration_s: int = DEFAULT_CHUNK_S,
) -> Path:
    """Transcribe *audio* to an SRT file using FunASR SenseVoice.

    The audio is split into fixed-length chunks (default 30 s) to keep peak
    memory low enough for Colab free-tier (≈12 GB RAM).
    """
    if model_cache_dir:
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MODELSCOPE_CACHE", str(model_cache_dir))
        os.environ.setdefault("HF_HOME", str(model_cache_dir / "huggingface"))

    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError("FunASR is not installed. Install requirements.txt first.") from exc

    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    print(f"[asr] device={device}, chunk={chunk_duration_s}s", flush=True)

    model = AutoModel(
        model=model_name,
        trust_remote_code=True,
        disable_update=True,
        device=device,
    )

    cues = _transcribe_chunked(model, audio, language, chunk_duration_s)
    write_srt(output_srt, cues)
    return output_srt


# ----------------------------------------------------------------------- #
# Chunked processing
# ----------------------------------------------------------------------- #

def _transcribe_chunked(
    model: Any,
    audio: Path,
    language: str,
    chunk_s: int,
) -> list[Cue]:
    """Split *audio* with pydub, run ASR per chunk, merge results."""
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError(
            "pydub is not installed.  pip install pydub"
        ) from exc

    print(f"[asr] loading audio: {audio}", flush=True)
    full_audio = AudioSegment.from_file(str(audio))
    total_ms = len(full_audio)
    chunk_ms = chunk_s * 1000
    n_chunks = (total_ms + chunk_ms - 1) // chunk_ms
    print(
        f"[asr] total={total_ms / 1000:.1f}s, splitting into {n_chunks} chunks",
        flush=True,
    )

    tmp_dir = audio.parent / ".asr_chunks"
    tmp_dir.mkdir(exist_ok=True)

    all_cues: list[Cue] = []
    try:
        for idx, start_ms in enumerate(range(0, total_ms, chunk_ms), 1):
            end_ms = min(start_ms + chunk_ms, total_ms)
            segment = full_audio[start_ms:end_ms]

            temp_path = tmp_dir / f"chunk_{start_ms}.wav"
            segment.export(str(temp_path), format="wav")

            print(
                f"[asr] chunk {idx}/{n_chunks}: "
                f"{start_ms / 1000:.1f}s – {end_ms / 1000:.1f}s ...",
                flush=True,
            )

            try:
                result = model.generate(
                    input=str(temp_path),
                    language=language,
                    use_itn=True,
                    sentence_timestamp=True,
                    merge_vad=True,
                    merge_length_s=15,
                )
                chunk_cues = _result_to_cues(
                    result,
                    offset_ms=start_ms,
                    fallback_duration_ms=end_ms - start_ms,
                )
                all_cues.extend(chunk_cues)
            except Exception as exc:
                print(f"[asr] chunk {idx} failed: {exc}", flush=True)
                raise
            finally:
                temp_path.unlink(missing_ok=True)

            # hint the GC to free temporary tensors between chunks
            gc.collect()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not all_cues:
        raise RuntimeError("FunASR returned no transcribable text")

    return renumber(all_cues)


# ----------------------------------------------------------------------- #
# Result parsing
# ----------------------------------------------------------------------- #

def _result_to_cues(
    result: Any,
    offset_ms: int = 0,
    fallback_duration_ms: int = 0,
    audio: Path | None = None,
) -> list[Cue]:
    """Convert FunASR generate() output to a list of :class:`Cue`.

    When *offset_ms* is non-zero, all timestamps are shifted forward so that
    per-chunk results land on the correct global timeline.
    """
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
            start = int(float(sentence.get("start", 0))) + offset_ms
            end = int(float(sentence.get("end", 0))) + offset_ms
            if end > start:
                cues.append(Cue(index=len(cues) + 1, start_ms=start, end_ms=end, text=text))

        # Fallback: model returned text but no sentence-level timestamps.
        if not cues and item.get("text"):
            text = _clean_text(str(item["text"]))
            if text:
                if fallback_duration_ms:
                    duration = fallback_duration_ms
                elif audio:
                    duration = media.probe_duration_ms(audio)
                else:
                    duration = 1000
                cues.append(
                    Cue(
                        index=1,
                        start_ms=offset_ms,
                        end_ms=offset_ms + max(1000, duration),
                        text=text,
                    )
                )

    return cues


def _clean_text(text: str) -> str:
    return _LANG_TAG_RE.sub("", text).strip()
