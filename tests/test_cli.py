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
    assert captured["prepare"]["translation_model"]


def test_cli_prepare_resume_last_without_saved_job_explains_first_run(tmp_path: Path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["prepare", "--root", str(tmp_path), "--resume-last"])

    captured = capsys.readouterr()
    assert "Set the source URL once" in captured.err
