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
            f"Command failed with exit code {returncode}: {' '.join(command)}\n{stderr.strip() or stdout.strip()}"
        )


def run(
    command: list[str],
    cwd: Path | None = None,
    capture: bool = False,
    capture_stdout_only: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    """Run a command and return stdout when captured.

    Args:
        capture: Capture both stdout and stderr (command runs silently).
        capture_stdout_only: Capture stdout for parsing but let stderr pass
            through to the terminal so progress output remains visible.
    """
    print(f"+ {' '.join(command)}", flush=True)
    if capture_stdout_only:
        pipe_stdout = subprocess.PIPE
        pipe_stderr = None  # inherit – visible in terminal
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
