"""用户级进程并行与批次互斥；同一用户仍串行运行。"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_STOP = None


def stop_requested() -> bool:
    return _STOP is not None and _STOP.is_set()


def initialize_worker(event) -> None:
    global _STOP
    _STOP = event
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@contextmanager
def batch_lock(folder: Path) -> Iterator[None]:
    """Windows内核文件锁；退出即释放，保留小锁文件不代表仍占用。"""
    import msvcrt
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".batch.lock").open("a+", encoding="utf-8") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write("0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("此批次目录正在被另一进程使用，请勿重复启动") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def execute_user(config: dict, mode: str, folder: str, device: str, resume: bool) -> dict:
    """每个进程独占一个用户的模型客户端、数据库与计时。"""
    from legacy_memory_agent import run_one
    if multiprocessing.current_process().name != "MainProcess":
        logging.basicConfig(level=logging.INFO,
                            format=f"%(asctime)s %(levelname)s [user={device}] %(message)s", force=True)
    child = Path(folder) / device
    single = {k: v for k, v in config.items() if k != "user_selection"}
    single["device_id"] = device
    started = time.perf_counter()
    error = None
    logging.info("开始用户 %s", device)
    try:
        if stop_requested():
            raise KeyboardInterrupt()
        continuing = resume and child.exists() and any(child.iterdir())
        code = run_one(single, mode, str(child), continuing)
        status = ("prepared" if mode == "prepare" else "completed") if code == 0 else "failed_or_partial"
    except KeyboardInterrupt:
        code, status, error = 2, "interrupted", "用户中断"
    except Exception as exc:
        code, status, error = 2, "failed", f"{type(exc).__name__}: {exc}"
        logging.error("用户%s失败：%s", device, error)
    report_path = child / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            error = error or report.get("error")
            if report.get("flow_version") in ("generic-flow-v1", "generic-flow-v2") and not error:
                status = "awaiting_retry" if report.get("retryable_tasks") else report.get("status", status)
            if str(error).startswith("interrupted:"):
                status = "interrupted"
        except (OSError, ValueError) as exc:
            code, status, error = 2, "failed", f"报告读取失败：{exc}"
    logging.info("结束用户 %s status=%s", device, status)
    return {"device_id": device, "status": status, "exit_code": code, "error": error,
            "worker_pid": os.getpid(),
            "output": str(child), "html": None,
            "elapsed_seconds": round(time.perf_counter() - started, 3)}


def iter_users(config: dict, mode: str, folder: Path, devices: list[str],
               resume: bool, workers: int) -> Iterator[dict]:
    """只保留workers个在途任务；Ctrl+C停止派发，在模型调用边界安全退出。"""
    if workers == 1:
        for device in devices:
            row = execute_user(config, mode, str(folder), device, resume)
            yield row
            if row["status"] == "interrupted":
                return
        return
    context = multiprocessing.get_context("spawn")
    event = context.Event()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                               initializer=initialize_worker, initargs=(event,))
    pending = {}
    todo = iter(devices)
    def fill() -> None:
        while len(pending) < workers and not event.is_set():
            device = next(todo, None)
            if device is None:
                break
            pending[pool.submit(execute_user, config, mode, str(folder), device, resume)] = device
    try:
        fill()
        while pending:
            try:
                done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                event.set()
                logging.warning("停止派发新用户；等待当前模型请求返回并保存，请勿强制关闭窗口")
                continue
            for future in done:
                device = pending.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"device_id": device, "status": "failed", "exit_code": 2,
                           "error": f"工作进程失败：{exc}", "output": str(folder / device),
                           "html": None, "elapsed_seconds": None}
                if row["status"] == "interrupted":
                    event.set()
                yield row
            fill()
    finally:
        event.set()
        pool.shutdown(wait=True, cancel_futures=True)
