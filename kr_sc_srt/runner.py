from __future__ import annotations

import subprocess
from pathlib import Path


class CommandError(RuntimeError):
    def __init__(self, command: list[str], returncode: int, stdout: str, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"命令执行失败，退出状态码 {returncode}: {' '.join(command)}\n{stderr.strip() or stdout.strip()}"
        )


def run(
    command: list[str],
    cwd: Path | None = None,
    capture: bool = False,
    capture_stdout_only: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    """运行命令并在被捕获时返回标准输出（stdout）。

    参数：
        capture: 捕获标准输出和标准错误（命令行静默运行）。
        capture_stdout_only: 仅捕获标准输出用于解析，而让标准错误直接输出到终端，
            以保持进度显示可见。
    """
    print(f"+ {' '.join(command)}", flush=True)
    if capture_stdout_only:
        pipe_stdout = subprocess.PIPE
        pipe_stderr = None  # 继承 - 在终端中可见
    elif capture:
        pipe_stdout = subprocess.PIPE
        pipe_stderr = subprocess.PIPE
    else:
        pipe_stdout = None
        pipe_stderr = None
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=pipe_stdout,
        stderr=pipe_stderr,
        check=False,
    )
    stdout = process.stdout or ""
    stderr = process.stderr or ""
    if process.returncode != 0:
        raise CommandError(command, process.returncode, stdout, stderr)
    return stdout
