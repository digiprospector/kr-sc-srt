from kr_sc_srt.srt import crop, parse_srt, render_srt


def test_parse_render_and_crop_retimes_cues():
    cues = parse_srt(
        """1
00:00:01,000 --> 00:00:03,000
hello

2
00:00:04,000 --> 00:00:06,000
world
"""
    )

    cropped = crop(cues, 2500, 5000)

    assert len(cropped) == 2
    assert cropped[0].start_ms == 0
    assert cropped[0].end_ms == 500
    assert cropped[1].start_ms == 1500
    assert cropped[1].end_ms == 2500
    assert render_srt(cropped).startswith("1\n00:00:00,000 --> 00:00:00,500")


def test_parse_srt_allows_missing_numeric_index():
    cues = parse_srt(
        """00:01:00.000 --> 00:01:02.000
line
"""
    )

    assert cues[0].index == 1
    assert cues[0].start_ms == 60_000
    assert cues[0].text == "line"
