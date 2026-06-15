from pathlib import Path

import pytest

from kr_sc_srt.segments import read_segments


def test_read_segments_with_header_and_duplicate_names(tmp_path: Path):
    path = tmp_path / "job.csv"
    path.write_text(
        "name,start,end\npart,00:01,00:02\npart,00:02,00:03.500\n",
        encoding="utf-8",
    )

    segments = read_segments(path)

    assert [segment.safe_name for segment in segments] == ["part", "part_2"]
    assert segments[1].start_ms == 2000
    assert segments[1].end_ms == 3500


def test_read_segments_rejects_invalid_range(tmp_path: Path):
    path = tmp_path / "job.csv"
    path.write_text("name,start,end\nbad,00:05,00:03\n", encoding="utf-8")

    with pytest.raises(ValueError, match="end must be after start"):
        read_segments(path)
