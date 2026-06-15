from pathlib import Path

from kr_sc_srt import download


def test_resolve_source_returns_local_path(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")

    resolved = download.resolve_source(str(video), tmp_path / "out", "low")

    assert resolved == video.resolve()
