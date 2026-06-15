from pathlib import Path

from kr_sc_srt.state import JobState, StageResult, load_last_job, save_last_job


def test_state_skips_when_dynamic_output_exists(tmp_path: Path):
    output = tmp_path / "source.low.mp4"
    output.write_bytes(b"video")
    state = JobState(tmp_path / "run.json")
    params = {"quality": "low"}

    state.mark_completed("download_low", params, StageResult({"video": str(output)}))

    resumed = JobState(tmp_path / "run.json")
    assert resumed.is_complete("download_low", params, [])


def test_state_does_not_skip_missing_dynamic_output(tmp_path: Path):
    missing = tmp_path / "missing.mp4"
    state = JobState(tmp_path / "run.json")
    params = {"quality": "low"}

    state.mark_completed("download_low", params, StageResult({"video": str(missing)}))

    resumed = JobState(tmp_path / "run.json")
    assert not resumed.is_complete("download_low", params, [])


def test_last_job_roundtrip(tmp_path: Path):
    out_dir = tmp_path / "outputs" / "job"
    save_last_job(tmp_path, "job", out_dir, "https://example.test/video")

    data = load_last_job(tmp_path)

    assert data["job_name"] == "job"
    assert data["source"] == "https://example.test/video"
    assert Path(data["out_dir"]) == out_dir
