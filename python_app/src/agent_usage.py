"""按本次session新增调用日志汇总供应商token，不混入续跑历史。"""
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import time
from typing import Iterator

LOGGER = logging.getLogger(__name__)


def _tokens(usage: dict, *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


@contextmanager
def session_usage(path: Path, device_id: str, session_id: str) -> Iterator[dict]:
    """打印当前执行的调用消耗；缺失usage显示未知，部分返回显示已知小计。"""
    offset = path.stat().st_size if path.exists() else 0
    started = time.perf_counter()
    state = {"status": "completed"}
    try:
        yield state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        raise
    finally:
        totals = [0, 0, 0]
        missing = [0, 0, 0]
        calls = 0
        try:
            if path.exists():
                # 二进制定位文件边界，再按UTF-8逐行解码；不扫描以前session的日志。
                with path.open("rb") as stream:
                    stream.seek(offset)
                    for line in stream:
                        if not line.strip():
                            continue
                        calls += 1
                        try:
                            record = json.loads(line.decode("utf-8"))
                            usage = record.get("usage") or {}
                            if not isinstance(usage, dict):
                                usage = {}
                        except (ValueError, AttributeError):
                            LOGGER.warning("session=%s 调用日志不可解析，本条token未知", session_id)
                            usage = {}
                        prompt = _tokens(usage, "prompt_tokens", "input_tokens")
                        completion = _tokens(usage, "completion_tokens", "output_tokens")
                        total = _tokens(usage, "total_tokens")
                        if total is None and prompt is not None and completion is not None:
                            total = prompt + completion
                        # reasoning/cache细项属于供应商usage明细，不重复加到总量。
                        for i, value in enumerate((prompt, completion, total)):
                            if value is None:
                                missing[i] += 1
                            else:
                                totals[i] += value
            labels = []
            for value, unknown in zip(totals, missing):
                if unknown == 0:
                    labels.append(str(value))
                elif unknown == calls:
                    labels.append(f"未知(缺失{unknown}次)")
                else:
                    labels.append(f"已知{value}(另{unknown}次未知)")
            LOGGER.info(
                "用户=%s session=%s 分析结束 status=%s 本次模型调用=%s "
                "输入token=%s 输出token=%s 总token=%s 耗时=%.2fs（仅本次执行）",
                device_id, session_id, state["status"], calls, *labels,
                time.perf_counter() - started,
            )
        except Exception:
            # 统计故障不能覆盖原始提取异常，也不能误报零消耗。
            LOGGER.exception("用户=%s session=%s token统计失败，消耗未知", device_id, session_id)
