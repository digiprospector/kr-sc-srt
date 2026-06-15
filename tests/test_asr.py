from pathlib import Path
from unittest.mock import patch

import pytest

from kr_sc_srt import asr
from kr_sc_srt.srt import read_srt


def test_transcribe_short_audio_builds_correct_command(tmp_path: Path):
    audio = tmp_path / "test.aac"
    audio.write_bytes(b"fake")
    output_srt = tmp_path / "test.ko.srt"
    generated_srt = tmp_path / "test.srt"

    captured: dict = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        generated_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
        return ""

    with (
        patch("kr_sc_srt.asr.media.probe_duration_ms", return_value=1_000),
        patch("kr_sc_srt.asr.runner.run", side_effect=fake_run),
    ):
        result = asr.transcribe_to_srt(audio, output_srt, model_name="large-v2")

    assert result == output_srt
    assert output_srt.exists()
    cmd = captured["cmd"]
    assert cmd[0] == asr.WHISPER_BINARY
    assert str(audio) in cmd
    assert cmd[cmd.index("-m") + 1] == "large-v2"
    assert cmd[cmd.index("-l") + 1] == "Korean"
    assert "--vad_method" in cmd
    assert "--ff_vocal_extract" in cmd
    assert "--sentence" in cmd
    assert cmd[cmd.index("-f") + 1] == "srt"


def test_transcribe_short_audio_renames_output(tmp_path: Path):
    audio = tmp_path / "198391511.aac"
    audio.write_bytes(b"fake")
    output_srt = tmp_path / "198391511.ko.srt"

    def fake_run(cmd, **_kwargs):
        out_dir = Path(cmd[cmd.index("-o") + 1])
        (out_dir / "198391511.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8"
        )
        return ""

    with (
        patch("kr_sc_srt.asr.media.probe_duration_ms", return_value=1_000),
        patch("kr_sc_srt.asr.runner.run", side_effect=fake_run),
    ):
        result = asr.transcribe_to_srt(audio, output_srt)

    assert result == output_srt
    assert output_srt.exists()
    assert "hello" in output_srt.read_text(encoding="utf-8")
    assert not (tmp_path / "198391511.srt").exists()


def test_transcribe_short_audio_same_path_no_rename(tmp_path: Path):
    audio = tmp_path / "test.aac"
    audio.write_bytes(b"fake")
    output_srt = tmp_path / "test.srt"

    def fake_run(cmd, **_kwargs):
        out_dir = Path(cmd[cmd.index("-o") + 1])
        (out_dir / "test.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nsame\n", encoding="utf-8"
        )
        return ""

    with (
        patch("kr_sc_srt.asr.media.probe_duration_ms", return_value=1_000),
        patch("kr_sc_srt.asr.runner.run", side_effect=fake_run),
    ):
        result = asr.transcribe_to_srt(audio, output_srt)

    assert result == output_srt
    assert output_srt.exists()


def test_transcribe_short_audio_missing_output_raises(tmp_path: Path):
    audio = tmp_path / "test.aac"
    audio.write_bytes(b"fake")
    output_srt = tmp_path / "test.ko.srt"

    def fake_run(cmd, **_kwargs):
        return ""

    with (
        patch("kr_sc_srt.asr.media.probe_duration_ms", return_value=1_000),
        patch("kr_sc_srt.asr.runner.run", side_effect=fake_run),
    ):
        with pytest.raises(FileNotFoundError, match="Whisper"):
            asr.transcribe_to_srt(audio, output_srt)


def test_silence_midpoints_in_window():
    minute = 60 * 1000
    # 窗口: [25分钟, 35分钟]
    # 语音区间: [26分钟, 29分钟], [31分钟, 34分钟]
    # 静音中点应为: 25.5分钟, 30.0分钟, 34.5分钟
    speech_ranges = [
        asr.SpeechRange(26 * minute, 29 * minute),
        asr.SpeechRange(31 * minute, 34 * minute),
    ]
    midpoints = asr._silence_midpoints_in_window(
        speech_ranges,
        window_start_ms=25 * minute,
        window_end_ms=35 * minute,
    )
    assert midpoints == [25.5 * minute, 30.0 * minute, 34.5 * minute]


def test_silence_midpoints_in_window_empty():
    minute = 60 * 1000
    # 窗口: [25分钟, 35分钟]，无语音区间
    # 应返回窗口的几何中点: 30分钟
    midpoints = asr._silence_midpoints_in_window(
        [],
        window_start_ms=25 * minute,
        window_end_ms=35 * minute,
    )
    assert midpoints == [30 * minute]


def test_transcribe_long_audio_merges_offset_chunk_srts(tmp_path: Path):
    minute = 60 * 1000
    audio = tmp_path / "long.aac"
    audio.write_bytes(b"fake")
    output_srt = tmp_path / "long.ko.srt"
    cut_calls: list[tuple[int, int]] = []

    def fake_cut_audio(_source, target, start_ms, end_ms):
        cut_calls.append((start_ms, end_ms))
        target.write_bytes(b"chunk")
        return target

    def fake_run(cmd, **_kwargs):
        chunk_audio = Path(cmd[1])
        out_dir = Path(cmd[cmd.index("-o") + 1])
        (out_dir / f"{chunk_audio.stem}.srt").write_text(
            f"1\n00:00:01,000 --> 00:00:02,000\n{chunk_audio.stem}\n",
            encoding="utf-8",
        )
        return ""

    with (
        patch("kr_sc_srt.asr.media.probe_duration_ms", return_value=65 * minute),
        patch(
            "kr_sc_srt.asr._detect_speech_ranges",
            return_value=[
                asr.SpeechRange(0, 4 * minute),
                asr.SpeechRange(6 * minute, 10 * minute),
            ],
        ),
        patch("kr_sc_srt.asr.media.cut_audio", side_effect=fake_cut_audio),
        patch("kr_sc_srt.asr.runner.run", side_effect=fake_run),
    ):
        result = asr.transcribe_to_srt(audio, output_srt, chunk_minutes=30)

    cues = read_srt(result)
    assert cut_calls == [
        (25 * minute, 35 * minute),
        (55 * minute, 65 * minute),
        (0, 30 * minute),
        (30 * minute, 60 * minute),
        (60 * minute, 65 * minute),
    ]
    assert [cue.start_ms for cue in cues] == [1_000, 30 * minute + 1_000, 60 * minute + 1_000]
    assert [cue.index for cue in cues] == [1, 2, 3]


def test_default_model():
    assert asr.DEFAULT_MODEL == "large-v2"
