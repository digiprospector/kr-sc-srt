from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"无效的 JSON 文件: {path}") from exc


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


@dataclass
class StageResult:
    outputs: dict[str, str]
    metadata: dict[str, Any] | None = None


class JobState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = read_json(path, default={"stages": {}})
        self.data.setdefault("stages", {})

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        write_json_atomic(self.path, self.data)

    def set_base(self, **values: Any) -> None:
        self.data.update(values)
        self.save()

    def stage(self, name: str) -> dict[str, Any]:
        return self.data.setdefault("stages", {}).setdefault(name, {})

    def output_path(self, stage_name: str, output_name: str) -> Path | None:
        output = self.stage(stage_name).get("outputs", {}).get(output_name)
        return Path(output) if output else None

    def is_complete(self, name: str, params: dict[str, Any], required_outputs: list[Path]) -> bool:
        stage = self.stage(name)
        if required_outputs and all(output.exists() and output.stat().st_size > 0 for output in required_outputs):
            # 如果参数发生明确变更，则不跳过（需要重新运行）；否则直接认为任务已完成
            if not (stage and stage.get("params") != params):
                return True

        if stage.get("status") != "completed":
            return False
        if stage.get("params") != params:
            return False
        if not required_outputs:
            required_outputs = [Path(value) for value in stage.get("outputs", {}).values()]
            if not required_outputs:
                return False
        for output in required_outputs:
            if not output.exists() or output.stat().st_size <= 0:
                return False
        return True

    def mark_running(self, name: str, params: dict[str, Any]) -> None:
        self.data.setdefault("stages", {})[name] = {
            "status": "running",
            "params": params,
            "started_at": utc_now(),
        }
        self.save()

    def mark_completed(self, name: str, params: dict[str, Any], result: StageResult) -> None:
        self.data.setdefault("stages", {})[name] = {
            "status": "completed",
            "params": params,
            "outputs": result.outputs,
            "metadata": result.metadata or {},
            "completed_at": utc_now(),
        }
        self.save()

    def mark_failed(self, name: str, params: dict[str, Any], error: BaseException) -> None:
        self.data.setdefault("stages", {})[name] = {
            "status": "failed",
            "params": params,
            "error": str(error),
            "failed_at": utc_now(),
        }
        self.save()


def save_last_job(root: Path, job_name: str, out_dir: Path, source: str) -> None:
    write_json_atomic(
        root / "last_job.json",
        {
            "job_name": job_name,
            "out_dir": str(out_dir),
            "source": source,
            "updated_at": utc_now(),
        },
    )


def load_last_job(root: Path) -> dict[str, Any]:
    path = root / "last_job.json"
    data = read_json(path)
    if not data.get("source") or not data.get("out_dir"):
        raise FileNotFoundError(f"在 {path} 下未找到可恢复的任务")
    return data
