from __future__ import annotations

import os
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _log_path() -> Path:
    path = _project_root() / "logs" / "command_history.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_line(line: str) -> None:
    with _log_path().open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _append_separator(tag: str) -> None:
    _append_line(f"{'=' * 24} {tag} {'=' * 24}")


def _ts() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _cmdline() -> str:
    return " ".join(sys.argv)


@contextmanager
def command_run_logger(entrypoint: str) -> Iterator[None]:
    start = time.perf_counter()
    run_id = f"{int(time.time() * 1000)}-{os.getpid()}"
    host = socket.gethostname()
    cwd = os.getcwd()
    _append_line("")
    _append_separator(f"START run_id={run_id}")
    _append_line(
        f"[{_ts()}] START run_id={run_id} entrypoint={entrypoint} host={host} cwd={cwd} cmd=\"{_cmdline()}\""
    )
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        elapsed = time.perf_counter() - start
        _append_line(
            f"[{_ts()}] END   run_id={run_id} entrypoint={entrypoint} status={status} elapsed_s={elapsed:.2f}"
        )
        _append_separator(f"END run_id={run_id}")
