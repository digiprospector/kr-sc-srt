from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from . import media, runner
from .srt import Cue, read_srt, renumber, write_srt

WHISPER_BINARY = "/content/drive/MyDrive/Faster-Whisper-XXL/faster-whisper-xxl"
DEFAULT_MODEL = "large-v2"
DEFAULT_CHUNK_MINUTES = 30
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_MIN_SILENCE_MS = 700
DEFAULT_VAD_SPEECH_PAD_MS = 300

VAD_SAMPLE_RATE = 16_000
SPLIT_SEARCH_WINDOW_MS = 5 * 60 * 1000
MIN_CHUNK_MS = 60 * 1000


@dataclass(frozen=True)
class SpeechRange:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start_ms: int
    end_ms: int
    path: Path


def transcribe_to_srt(
    audio: Path,
    output_srt: Path,
    model_name: str = DEFAULT_MODEL,
    chunk_minutes: int = DEFAULT_CHUNK_MINUTES,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    vad_min_silence_ms: int = DEFAULT_VAD_MIN_SILENCE_MS,
    vad_speech_pad_ms: int = DEFAULT_VAD_SPEECH_PAD_MS,
) -> Path:
    """Transcribe Korean audio to SRT with chunking for long inputs."""
    output_dir = output_srt.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    duration_ms = media.probe_duration_ms(audio)
    target_ms = max(1, chunk_minutes) * 60 * 1000
    if duration_ms <= target_ms:
        return _transcribe_single_audio(audio, output_srt, model_name)

    chunk_ranges = _plan_chunk_ranges_with_windowed_vad(
        audio=audio,
        duration_ms=duration_ms,
        target_ms=target_ms,
        threshold=vad_threshold,
        min_silence_ms=vad_min_silence_ms,
        speech_pad_ms=vad_speech_pad_ms,
        temp_dir=output_dir / ".asr_chunks",
    )
    chunks = _cut_audio_chunks(audio, output_dir / ".asr_chunks", chunk_ranges)

    merged: list[Cue] = []
    for chunk in chunks:
        chunk_srt = chunk.path.with_suffix(".srt")
        _transcribe_single_audio(chunk.path, chunk_srt, model_name)
        merged.extend(_read_offset_cues(chunk_srt, chunk.start_ms))

    if not merged:
        raise RuntimeError("Whisper did not return any subtitles")

    write_srt(output_srt, renumber(sorted(merged, key=lambda cue: (cue.start_ms, cue.end_ms))))
    print(f"[asr] Korean subtitles generated: {output_srt}", flush=True)
    return output_srt


def _transcribe_single_audio(audio: Path, output_srt: Path, model_name: str) -> Path:
    output_dir = output_srt.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_srt.unlink(missing_ok=True)

    cmd = [
        WHISPER_BINARY,
        str(audio),
        "-m",
        model_name,
        "-l",
        "Korean",
        "--vad_method",
        "pyannote_v3",
        "--ff_vocal_extract",
        "mdx_kim2",
        "--sentence",
        "-v",
        "true",
        "-o",
        str(output_dir),
        "-f",
        "srt",
    ]

    print(f"[asr] Running Whisper: model={model_name}, audio={audio}", flush=True)
    runner.run(cmd)

    generated = output_dir / f"{audio.stem}.srt"
    if generated.resolve() != output_srt.resolve():
        if generated.exists():
            shutil.move(str(generated), str(output_srt))
        else:
            raise FileNotFoundError(f"Whisper did not create the expected SRT: {generated}")

    if not output_srt.exists():
        raise FileNotFoundError(f"Whisper did not create the expected SRT: {output_srt}")
    return output_srt


def _detect_speech_ranges(
    audio: Path,
    threshold: float,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> list[SpeechRange]:
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    except ImportError as exc:
        raise RuntimeError(
            "silero-vad is required for long audio chunking. "
            "Install dependencies with pip install -r requirements.txt."
        ) from exc

    model = load_silero_vad()
    wav = read_audio(str(audio), sampling_rate=VAD_SAMPLE_RATE)
    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=VAD_SAMPLE_RATE,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    return [
        SpeechRange(
            start_ms=_samples_to_ms(item["start"]),
            end_ms=_samples_to_ms(item["end"]),
        )
        for item in timestamps
        if item["end"] > item["start"]
    ]


def _plan_chunk_ranges_with_windowed_vad(
    audio: Path,
    duration_ms: int,
    target_ms: int,
    threshold: float,
    min_silence_ms: int,
    speech_pad_ms: int,
    temp_dir: Path,
) -> list[tuple[int, int]]:
    if duration_ms <= target_ms:
        return [(0, duration_ms)]

    ranges: list[tuple[int, int]] = []
    start_ms = 0
    temp_dir.mkdir(parents=True, exist_ok=True)

    while duration_ms - start_ms > target_ms:
        target_split = start_ms + target_ms
        lower_ms = max(start_ms + MIN_CHUNK_MS, target_split - SPLIT_SEARCH_WINDOW_MS)
        upper_ms = min(duration_ms, target_split + SPLIT_SEARCH_WINDOW_MS)

        if upper_ms > lower_ms:
            temp_vad_wav = temp_dir / f"vad_temp_{lower_ms}_{upper_ms}.wav"
            try:
                media.cut_audio(audio, temp_vad_wav, lower_ms, upper_ms)
                relative_ranges = _detect_speech_ranges(
                    temp_vad_wav,
                    threshold=threshold,
                    min_silence_ms=min_silence_ms,
                    speech_pad_ms=speech_pad_ms,
                )
                speech_ranges = [
                    SpeechRange(r.start_ms + lower_ms, r.end_ms + lower_ms)
                    for r in relative_ranges
                ]
                candidates = [
                    midpoint
                    for midpoint in _silence_midpoints_in_window(speech_ranges, lower_ms, upper_ms)
                    if lower_ms <= midpoint <= upper_ms
                ]
                if candidates:
                    split_ms = min(candidates, key=lambda point: abs(point - target_split))
                else:
                    split_ms = min(target_split, duration_ms)
            except Exception as exc:
                print(f"[asr] Warning: VAD failed on window [{lower_ms}, {upper_ms}]: {exc}. Falling back to hard split.")
                split_ms = min(target_split, duration_ms)
            finally:
                temp_vad_wav.unlink(missing_ok=True)
        else:
            split_ms = min(target_split, duration_ms)

        ranges.append((start_ms, split_ms))
        start_ms = split_ms

    ranges.append((start_ms, duration_ms))
    return ranges


def _silence_midpoints_in_window(
    speech_ranges: list[SpeechRange],
    window_start_ms: int,
    window_end_ms: int,
) -> list[int]:
    if not speech_ranges:
        return [(window_start_ms + window_end_ms) // 2]

    ordered = sorted(speech_ranges, key=lambda item: item.start_ms)
    midpoints: list[int] = []
    previous_end = window_start_ms
    for speech in ordered:
        if speech.start_ms > previous_end:
            midpoints.append((previous_end + speech.start_ms) // 2)
        previous_end = max(previous_end, speech.end_ms)
    if previous_end < window_end_ms:
        midpoints.append((previous_end + window_end_ms) // 2)
    return midpoints


def _cut_audio_chunks(
    audio: Path,
    chunk_dir: Path,
    ranges: list[tuple[int, int]],
) -> list[AudioChunk]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[AudioChunk] = []
    for index, (start_ms, end_ms) in enumerate(ranges, start=1):
        chunk_path = chunk_dir / f"{audio.stem}.chunk{index:03d}.wav"
        media.cut_audio(audio, chunk_path, start_ms, end_ms)
        chunks.append(AudioChunk(index=index, start_ms=start_ms, end_ms=end_ms, path=chunk_path))
    return chunks


def _read_offset_cues(srt_path: Path, offset_ms: int) -> list[Cue]:
    return [
        Cue(
            index=cue.index,
            start_ms=cue.start_ms + offset_ms,
            end_ms=cue.end_ms + offset_ms,
            text=cue.text,
        )
        for cue in read_srt(srt_path)
    ]


def _samples_to_ms(samples: int) -> int:
    return round(samples * 1000 / VAD_SAMPLE_RATE)
