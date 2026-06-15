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

# 每个分片 180 秒 - 在 ASR 上下文与内存安全之间取得良好平衡。
DEFAULT_CHUNK_S = 180


def transcribe_to_srt(
    audio: Path,
    output_srt: Path,
    model_cache_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    language: str = "ko",
    chunk_duration_s: int = DEFAULT_CHUNK_S,
) -> Path:
    """使用 FunASR SenseVoice 将音频 *audio* 转录为 SRT 文件。

    音频会被切分为固定长度的分片（默认 30 秒），以保持较低的内存占用峰值，
    适配 Colab 免费层（约 12 GB 内存）。
    """
    if model_cache_dir:
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MODELSCOPE_CACHE", str(model_cache_dir))
        os.environ.setdefault("HF_HOME", str(model_cache_dir / "huggingface"))

    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError("未安装 FunASR。请先运行 pip install -r requirements.txt 安装依赖。") from exc

    try:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    print(f"[asr] 设备={device}, 分片={chunk_duration_s}秒", flush=True)

    model = AutoModel(
        model=model_name,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        trust_remote_code=True,
        disable_update=True,
        device=device,
        output_timestamp=True,
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
    """使用 pydub 分割音频 *audio*，对每个分片运行 ASR 语音识别，然后合并结果。"""
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError(
            "未安装 pydub。请运行 pip install pydub 安装。"
        ) from exc

    print(f"[asr] 正在加载音频: {audio}", flush=True)
    full_audio = AudioSegment.from_file(str(audio))
    total_ms = len(full_audio)
    chunk_ms = chunk_s * 1000
    n_chunks = (total_ms + chunk_ms - 1) // chunk_ms
    print(
        f"[asr] 总时长={total_ms / 1000:.1f}秒，正在分割为 {n_chunks} 个分片",
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
                f"[asr] 分片 {idx}/{n_chunks}: "
                f"{start_ms / 1000:.1f}秒 – {end_ms / 1000:.1f}秒 ...",
                flush=True,
            )
            try:
                result = model.generate(
                    input=str(temp_path),
                    language=language,
                    use_itn=True,
                    merge_vad=True,
                    merge_length_s=15,
                    output_timestamp=True,
                )
                print(f"[asr] 分片 {idx} 原始结果: {result}", flush=True)
                chunk_cues = _result_to_cues(
                    result,
                    offset_ms=start_ms,
                    fallback_duration_ms=end_ms - start_ms,
                )
                all_cues.extend(chunk_cues)
            except Exception as exc:
                print(f"[asr] 分片 {idx} 失败: {exc}", flush=True)
                raise
            finally:
                temp_path.unlink(missing_ok=True)

            # 提示 GC 在分片之间释放临时张量
            gc.collect()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not all_cues:
        raise RuntimeError("FunASR 未返回任何可转录的文本")

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
    """将 FunASR generate() 的输出转换为 :class:`Cue` 列表。

    当 *offset_ms* 不为零时，所有时间戳都会向前偏移，
    以便每个分片的结果能够对应到正确的全局时间线上。
    """
    items = result if isinstance(result, list) else [result]
    cues: list[Cue] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        
        # 1. 优先使用句子级时间戳（如 sentence_info 或 sentences）
        sentence_info = item.get("sentence_info") or item.get("sentences") or []
        for sentence in sentence_info:
            text = _clean_text(str(sentence.get("text", "")))
            if not text:
                continue
            start = int(float(sentence.get("start", 0))) + offset_ms
            end = int(float(sentence.get("end", 0))) + offset_ms
            if end > start:
                cues.append(Cue(index=len(cues) + 1, start_ms=start, end_ms=end, text=text))

        # 2. 如果没有句子级时间戳，但有字词级时间戳（timestamp 字段），在本地进行句度分割聚合
        if not cues:
            raw_timestamp = item.get("timestamp") or []
            cleaned_text = _clean_text(item.get("text", ""))
            
            raw_tokens: list[dict[str, Any]] = []
            if raw_timestamp:
                first_elem = raw_timestamp[0]
                # 2.1 支持 [word, start, end] 格式
                if isinstance(first_elem, (list, tuple)) and len(first_elem) >= 3:
                    for elem in raw_timestamp:
                        if isinstance(elem, (list, tuple)) and len(elem) >= 3:
                            w = _clean_text(str(elem[0]))
                            if w:
                                raw_tokens.append({
                                    "text": w,
                                    "start": int(elem[1]),
                                    "end": int(elem[2])
                                })
                # 2.2 支持 [start, end] 格式，需与 cleaned_text 的词或字进行绑定
                elif isinstance(first_elem, (list, tuple)) and len(first_elem) == 2:
                    words = cleaned_text.split()
                    if len(words) == len(raw_timestamp):
                        for w, elem in zip(words, raw_timestamp):
                            raw_tokens.append({
                                "text": w,
                                "start": int(elem[0]),
                                "end": int(elem[1])
                            })
                    else:
                        # 如果非空格分隔（如中文或字符级时间戳），则与去空格后的字符绑定
                        chars = [c for c in cleaned_text if not c.isspace()]
                        if len(chars) == len(raw_timestamp):
                            for c, elem in zip(chars, raw_timestamp):
                                raw_tokens.append({
                                    "text": c,
                                    "start": int(elem[0]),
                                    "end": int(elem[1])
                                })
                        else:
                            # 长度仍不匹配时，进行最小长度对齐截断
                            min_len = min(len(chars), len(raw_timestamp))
                            for i in range(min_len):
                                raw_tokens.append({
                                    "text": chars[i],
                                    "start": int(raw_timestamp[i][0]),
                                    "end": int(raw_timestamp[i][1])
                                })

            # 2.3 将解析出的字词级 token 按停顿（Gap）、时长和字数聚合为字幕行
            if raw_tokens:
                max_gap_ms = 800        # 静音停顿超过 0.8 秒切分
                max_duration_ms = 4000  # 单行字幕最长 4 秒
                max_text_len = 36       # 单行字幕最长 36 字符

                curr_tokens: list[dict[str, Any]] = []
                for tok in raw_tokens:
                    if not curr_tokens:
                        curr_tokens.append(tok)
                        continue

                    prev_tok = curr_tokens[-1]
                    gap = tok["start"] - prev_tok["end"]
                    curr_duration = tok["end"] - curr_tokens[0]["start"]
                    curr_text = " ".join([t["text"] for t in curr_tokens]) + " " + tok["text"]

                    if gap > max_gap_ms or curr_duration > max_duration_ms or len(curr_text) > max_text_len:
                        cue_text = " ".join([t["text"] for t in curr_tokens])
                        start = curr_tokens[0]["start"] + offset_ms
                        end = curr_tokens[-1]["end"] + offset_ms
                        cues.append(Cue(index=len(cues) + 1, start_ms=start, end_ms=end, text=cue_text))
                        curr_tokens = [tok]
                    else:
                        curr_tokens.append(tok)

                if curr_tokens:
                    cue_text = " ".join([t["text"] for t in curr_tokens])
                    start = curr_tokens[0]["start"] + offset_ms
                    end = curr_tokens[-1]["end"] + offset_ms
                    cues.append(Cue(index=len(cues) + 1, start_ms=start, end_ms=end, text=cue_text))

        # 3. 兜底方案：模型仅返回了 text，但既没有句子级时间戳也没有有效词级时间戳。
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
                        index=len(cues) + 1,
                        start_ms=offset_ms,
                        end_ms=offset_ms + max(1000, duration),
                        text=text,
                    )
                )

    return cues


def _clean_text(text: str) -> str:
    return _LANG_TAG_RE.sub("", text).strip()
