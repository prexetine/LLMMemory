"""进程内终端镜像；不同用户分别落盘，追加、实时刷新，不读取模型凭据。"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import json
import logging
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Iterator, TextIO

_ACTIVE: set[Path] = set()


class Tee:
    def __init__(self, original: TextIO, sink: TextIO, lock: threading.RLock):
        self.original, self.sink, self.lock = original, sink, lock

    def write(self, text: str) -> int:
        with self.lock:
            self.original.write(text)
            self.sink.write(text)
            self.sink.flush()
        return len(text)

    def flush(self) -> None:
        with self.lock:
            self.original.flush()
            self.sink.flush()

    def __getattr__(self, name: str):
        return getattr(self.original, name)


@contextmanager
def terminal_log(folder: Path, label: str) -> Iterator[None]:
    """记录logging、stdout/stderr及逃逸异常；不捕获PowerShell自身的输入。"""
    path = (folder / "logs" / "terminal.log").resolve()
    if path in _ACTIVE:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    original_level = root.level
    original_out, original_err = sys.stdout, sys.stderr
    console = None
    if not root.handlers:
        console = logging.StreamHandler(original_err)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(console)
    root.setLevel(min(root.level, logging.INFO))
    with path.open("a", encoding="utf-8", newline="\n", buffering=1) as sink:
        handler = logging.StreamHandler(sink)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [pid=%(process)d] %(message)s"))
        root.addHandler(handler)
        _ACTIVE.add(path)
        lock = threading.RLock()
        sys.stdout, sys.stderr = Tee(original_out, sink, lock), Tee(original_err, sink, lock)
        sink.write(f"\n=== START {datetime.now().astimezone().isoformat()} pid={os.getpid()} {label} ===\n")
        sink.flush()
        outcome = "returned"
        try:
            yield
        except BaseException:
            outcome = "exception"
            traceback.print_exc(file=sink)
            sink.flush()
            raise
        finally:
            sys.stdout, sys.stderr = original_out, original_err
            root.removeHandler(handler)
            handler.close()
            root.setLevel(original_level)
            if console:
                root.removeHandler(console)
                console.close()
            _ACTIVE.remove(path)
            sink.write(f"=== END {datetime.now().astimezone().isoformat()} {outcome} ===\n")
            sink.flush()


def logged_run(function):
    """适配已有run_batch(config_path,...)及run_one(config,...)签名。"""
    @wraps(function)
    def wrapped(config_or_path, mode, output_override=None, resume=False):
        config = config_or_path if isinstance(config_or_path, dict) else json.loads(Path(config_or_path).read_text(encoding="utf-8"))
        base = Path(__file__).resolve().parents[2]
        folder = Path(output_override) if output_override else base / config["output_dir"]
        label = f"{function.__name__} user={config.get('device_id', 'batch')} mode={mode} resume={resume}"
        with terminal_log(folder, label):
            result = function(config_or_path, mode, output_override, resume)
            logging.info("运行返回 user=%s exit_code=%s", config.get("device_id", "batch"), result)
            return result
    return wrapped
