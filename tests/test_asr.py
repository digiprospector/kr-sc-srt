from pathlib import Path
from unittest.mock import patch

import pytest

from kr_sc_srt import asr


def test_transcribe_to_srt_builds_correct_command(tmp_path: Path):
    audio = tmp_path / "test.aac"
    audio.write_bytes(b"fake")
    output_srt = tmp_path / "test.ko.srt"
    generated_srt = tmp_path / "test.srt"

    captured: dict = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        generated_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n안녕\n", encoding="utf-8")
        return ""

    with patch("kr_sc_srt.asr.runner.run", side_effect=fake_run):
        result = asr.transcribe_to_srt(audio, output_srt, model_name="large-v2")

    assert result == output_srt
    assert output_srt.exists()
    cmd = captured["cmd"]
    assert cmd[0] == asr.WHISPER_BINARY
    assert str(audio) in cmd
    assert "-m" in cmd
    assert cmd[cmd.index("-m") + 1] == "large-v2"
    assert "-l" in cmd
    assert cmd[cmd.index("-l") + 1] == "Korean"
    assert "--vad_method" in cmd
    assert "--ff_vocal_extract" in cmd
    assert "--sentence" in cmd
    assert "-f" in cmd
    assert cmd[cmd.index("-f") + 1] == "srt"


def test_transcribe_to_srt_renames_output(tmp_path: Path):
    audio = tmp_path / "198391511.aac"
    audio.write_bytes(b"fake")
    output_srt = tmp_path / "198391511.ko.srt"

    def fake_run(cmd, **_kwargs):
        out_dir = Path(cmd[cmd.index("-o") + 1])
        (out_dir / "198391511.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n테스트\n", encoding="utf-8"
        )
        return ""

    with patch("kr_sc_srt.asr.runner.run", side_effect=fake_run):
        result = asr.transcribe_to_srt(audio, output_srt)

    assert result == output_srt
    assert output_srt.exists()
    assert "테스트" in output_srt.read_text(encoding="utf-8")
    assert not (tmp_path / "198391511.srt").exists()


def test_transcribe_to_srt_same_path_no_rename(tmp_path: Path):
    audio = tmp_path / "test.aac"
    audio.write_bytes(b"fake")
    output_srt = tmp_path / "test.srt"

    def fake_run(cmd, **_kwargs):
        out_dir = Path(cmd[cmd.index("-o") + 1])
        (out_dir / "test.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n동일\n", encoding="utf-8"
        )
        return ""

    with patch("kr_sc_srt.asr.runner.run", side_effect=fake_run):
        result = asr.transcribe_to_srt(audio, output_srt)

    assert result == output_srt
    assert output_srt.exists()


def test_transcribe_to_srt_missing_output_raises(tmp_path: Path):
    audio = tmp_path / "test.aac"
    audio.write_bytes(b"fake")
    output_srt = tmp_path / "test.ko.srt"

    def fake_run(cmd, **_kwargs):
        return ""

    with patch("kr_sc_srt.asr.runner.run", side_effect=fake_run):
        with pytest.raises(FileNotFoundError, match="Whisper"):
            asr.transcribe_to_srt(audio, output_srt)


def test_default_model():
    assert asr.DEFAULT_MODEL == "large-v2"
