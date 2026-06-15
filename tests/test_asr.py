from kr_sc_srt.asr import _result_to_cues


def test_result_to_cues_sentence_info():
    result = {
        "sentence_info": [
            {"text": "hello", "start": 100, "end": 500},
            {"text": "world", "start": 600, "end": 1000},
        ]
    }
    cues = _result_to_cues(result, offset_ms=1000)
    assert len(cues) == 2
    assert cues[0].text == "hello"
    assert cues[0].start_ms == 1100
    assert cues[0].end_ms == 1500
    assert cues[1].text == "world"
    assert cues[1].start_ms == 1600
    assert cues[1].end_ms == 2000


def test_result_to_cues_word_timestamp():
    # Test [word, start, end] format
    result = {
        "text": "hello world from pytest",
        "timestamp": [
            ["hello", 100, 500],
            ["world", 600, 1000],
            ["from", 1100, 1500],
            # Simulate a silence gap of 1000ms (> 800ms)
            ["pytest", 2500, 3000],
        ]
    }
    cues = _result_to_cues(result, offset_ms=1000)
    # Should split before "pytest" due to gap
    assert len(cues) == 2
    assert cues[0].text == "hello world from"
    assert cues[0].start_ms == 1100
    assert cues[0].end_ms == 2500
    assert cues[1].text == "pytest"
    assert cues[1].start_ms == 3500
    assert cues[1].end_ms == 4000


def test_result_to_cues_simple_timestamp_space_separated():
    # Test [start, end] format aligned with space-separated words
    result = {
        "text": "hello world python",
        "timestamp": [
            [100, 500],
            [600, 1000],
            [1100, 1500],
        ]
    }
    cues = _result_to_cues(result, offset_ms=1000)
    assert len(cues) == 1
    assert cues[0].text == "hello world python"
    assert cues[0].start_ms == 1100
    assert cues[0].end_ms == 2500


def test_result_to_cues_simple_timestamp_cjk():
    # Test [start, end] format aligned character by character (CJK) without spaces
    result = {
        "text": "안녕하세요",
        "timestamp": [
            [100, 300],
            [300, 500],
            [500, 700],
            [700, 900],
            # Simulate gap > 800ms
            [1800, 2000],
        ]
    }
    cues = _result_to_cues(result, offset_ms=1000)
    assert len(cues) == 2
    assert cues[0].text == "안녕하세"
    assert cues[0].start_ms == 1100
    assert cues[0].end_ms == 1900
    assert cues[1].text == "요"
    assert cues[1].start_ms == 2800
    assert cues[1].end_ms == 3000


def test_result_to_cues_simple_timestamp_cjk_with_spaces():
    # Test [start, end] format aligned character by character (CJK) preserving actual spaces
    result = {
        "text": "안녕 하세요",
        "timestamp": [
            [100, 300],
            [300, 500],
            [500, 700],
            [700, 900],
            [900, 1100],
        ]
    }
    cues = _result_to_cues(result, offset_ms=1000)
    assert len(cues) == 1
    assert cues[0].text == "안녕 하세요"
    assert cues[0].start_ms == 1100
    assert cues[0].end_ms == 2100


def test_result_to_cues_duration_split():
    # Test splitting when duration exceeds max_duration_ms (4000ms)
    result = {
        "text": "a b c d e",
        "timestamp": [
            ["a", 0, 1000],
            ["b", 1000, 2000],
            ["c", 2000, 3000],
            ["d", 3000, 4100],  # total duration becomes 4100 (> 4000)
            ["e", 4100, 5000],
        ]
    }
    cues = _result_to_cues(result, offset_ms=0)
    # Should split before "d"
    assert len(cues) == 2
    assert cues[0].text == "a b c"
    assert cues[0].start_ms == 0
    assert cues[0].end_ms == 3000
    assert cues[1].text == "d e"
    assert cues[1].start_ms == 3000
    assert cues[1].end_ms == 5000


def test_result_to_cues_fallback():
    result = {
        "text": "hello fallback",
    }
    cues = _result_to_cues(result, offset_ms=1000, fallback_duration_ms=5000)
    assert len(cues) == 1
    assert cues[0].text == "hello fallback"
    assert cues[0].start_ms == 1000
    assert cues[0].end_ms == 6000
