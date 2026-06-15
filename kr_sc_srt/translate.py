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
        raise TranslationError("翻译需要提供 OPENAI_API_KEY")

    total_cues = len(cues)
    total_batches = (total_cues + batch_size - 1) // batch_size
    print(
        f"[translate] 开始翻译共 {total_cues} 条字幕 (模型={model}, api_base={api_base})",
        flush=True,
    )
    print(
        f"[translate] 分批大小: {batch_size}, 总批次数: {total_batches}",
        flush=True,
    )

    translated: list[str] = []
    for batch_idx, start in enumerate(range(0, total_cues, batch_size), 1):
        end = min(start + batch_size, total_cues)
        batch = cues[start:end]
        print(
            f"[translate] [{batch_idx}/{total_batches}] 正在翻译字幕 {start + 1} 到 {end} ...",
            end="",
            flush=True,
        )
        batch_translated = _translate_batch(batch, key, api_base, model, retries)
        translated.extend(batch_translated)
        print(" 完成。", flush=True)

    print(
        f"[translate] 所有 {total_cues} 条字幕已成功翻译。",
        flush=True,
    )
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
                raise TranslationError(f"预期有 {len(cues)} 条翻译，但实际收到 {len(translated)} 条")
            return translated
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TranslationError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                sleep_s = 2**attempt
                print(
                    f"\n[translate] 警告: 第 {attempt + 1} 次尝试失败: {exc}。将在 {sleep_s}秒后重试...",
                    flush=True,
                )
                time.sleep(sleep_s)

    raise TranslationError(f"在尝试 {retries} 次后翻译失败: {last_error}")


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
        raise TranslationError("翻译响应必须是一个 JSON 格式的字符串数组")
    return value
