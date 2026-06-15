from pathlib import Path

import pytest

from kr_sc_srt import cli
from kr_sc_srt.state import save_last_job


def test_cli_prepare_resume_last_uses_saved_url(monkeypatch, tmp_path: Path):
    out_dir = tmp_path / "outputs" / "job"
    save_last_job(tmp_path, "job", out_dir, "https://example.test/vod")
    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def prepare(self, **kwargs):
            captured["prepare"] = kwargs

    monkeypatch.setattr(cli, "Pipeline", FakePipeline)

    code = cli.main(["prepare", "--root", str(tmp_path), "--resume-last"])

    assert code == 0
    assert captured["source"] == "https://example.test/vod"
    assert captured["out_dir"] == out_dir.resolve()
    assert captured["prepare"]["asr_model"]


def test_cli_prepare_resume_last_without_saved_job_explains_first_run(tmp_path: Path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["prepare", "--root", str(tmp_path), "--resume-last"])

    captured = capsys.readouterr()
    assert "Set the source URL once" in captured.err


def test_cli_translate_srt_writes_output(monkeypatch, tmp_path: Path):
    source = tmp_path / "ko.srt"
    target = tmp_path / "zh.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n안녕\n",
        encoding="utf-8",
    )

    def fake_translate(cues, **kwargs):
        return [type(cue)(index=cue.index, start_ms=cue.start_ms, end_ms=cue.end_ms, text="你好") for cue in cues]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "translate_cues", fake_translate)

    code = cli.main(["translate-srt", str(source), str(target)])

    assert code == 0
    assert "你好" in target.read_text(encoding="utf-8")
