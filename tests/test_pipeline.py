from pathlib import Path

from kr_sc_srt.pipeline import Pipeline
from kr_sc_srt.state import StageResult


def test_stage_skips_completed_output(tmp_path: Path):
    calls = []
    output = tmp_path / "out.txt"
    pipeline = Pipeline(root=tmp_path, source="https://example.test/vod", out_dir=tmp_path / "job", log=lambda _: None)

    def action():
        calls.append("run")
        output.write_text("done", encoding="utf-8")
        return StageResult({"file": str(output)})

    pipeline._stage("custom", {"x": 1}, [output], action)
    pipeline._stage("custom", {"x": 1}, [output], action)

    assert calls == ["run"]


def test_stage_reruns_when_params_change(tmp_path: Path):
    calls = []
    output = tmp_path / "out.txt"
    pipeline = Pipeline(root=tmp_path, source="https://example.test/vod", out_dir=tmp_path / "job", log=lambda _: None)

    def action():
        calls.append("run")
        output.write_text("done", encoding="utf-8")
        return StageResult({"file": str(output)})

    pipeline._stage("custom", {"x": 1}, [output], action)
    pipeline._stage("custom", {"x": 2}, [output], action)

    assert calls == ["run", "run"]
