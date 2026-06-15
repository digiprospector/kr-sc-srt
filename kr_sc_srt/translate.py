from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .srt import Cue, replace_text


DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class TranslationError(RuntimeError):
    pass


def translate_cues(
    cues: list[Cue],
    api_key: str | None = None,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    batch_size: int = 40,
    retries: int = 3,
) -> list[Cue]:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise TranslationError("OPENAI_API_KEY is required for translation")

    translated: list[str] = []
    for start in range(0, len(cues), batch_size):
        batch = cues[start : start + batch_size]
        translated.extend(_translate_batch(batch, key, api_base, model, retries))
    return replace_text(cues, translated)


def _translate_batch(cues: list[Cue], api_key: str, api_base: str, model: str, retries: int) -> list[str]:
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate Korean subtitle lines into natural Simplified Chinese. "
                    "Return only a JSON array of strings, with the same length and order."
                ),
            },
            {
                "role": "user",
                "content": json.dumps([cue.text for cue in cues], ensure_ascii=False),
            },
        ],
    }
    url = api_base.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_body = response.read().decode("utf-8")
            data = json.loads(response_body)
            content = data["choices"][0]["message"]["content"]
            translated = _parse_json_array(content)
            if len(translated) != len(cues):
                raise TranslationError(f"Expected {len(cues)} translations, got {len(translated)}")
            return translated
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TranslationError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)

    raise TranslationError(f"Translation failed after {retries} attempts: {last_error}")


def _parse_json_array(content: str) -> list[str]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    value = json.loads(text)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TranslationError("Translation response must be a JSON array of strings")
    return value
