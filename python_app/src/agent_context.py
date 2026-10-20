"""有界上下文：保守UTF-8字节估算、完整分块、按需历史和请求预检。

不依赖供应商私有tokenizer；字节数是保守调度估值，不是供应商实测token数。
"""
from __future__ import annotations

import json
import math
from typing import Any, Iterator

from agent_schema import response_schema

CONTEXT_VERSION = "bounded-context-v1"
DEFAULTS = {
    "context_window_tokens": 65536,
    "context_safety_tokens": 4096,
    "session_chunk_tokens": 8192,
    "history_context_tokens": 4096,
    "rule_page_tokens": 8192,
    "context_split_retries": 6,
}


class ContextCapacityError(RuntimeError):
    """必须缩小任务后重试，不能原样重试或假装完成。"""


def size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def setting(config: dict, name: str) -> int:
    value = config.get(name, DEFAULTS[name])
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}必须为正整数")
    return value


def input_budget(config: dict) -> int:
    # 缺少显式输出上限时仍保留调度空间；此预留不发送给模型、不限制生成长度。
    budget = setting(config, "context_window_tokens") - config.get("max_output_tokens", 12000) - setting(config, "context_safety_tokens")
    if budget < 4096:
        raise ValueError("上下文窗口扣除输出预留和安全余量后不足4096")
    return budget


def request_size(role: str, prompt: str, data: dict) -> int:
    # 同时计入system/user文本、结构化输出schema和消息封装余量。
    return size({"messages": [{"role": "system", "content": prompt},
                              {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
                 "response_schema": response_schema(role, data)}) + 512


def check_request(config: dict, role: str, prompt: str, data: dict) -> int:
    estimated = request_metrics(config, role, prompt, data)["estimated_input_tokens"]
    budget = input_budget(config)
    if estimated > budget:
        raise ContextCapacityError(f"role={role} 估算输入={estimated} 预算={budget}，需拆分任务")
    return estimated


def request_metrics(config: dict, role: str, prompt: str, data: dict) -> dict:
    """显式区分传输字节与本地token估算；非供应商tokenizer或硬上界。"""
    mode = config.get("token_estimator", "utf8_bytes_legacy")
    byte_count = request_size(role, prompt, data)
    if mode == "utf8_bytes_legacy":
        estimate = byte_count
    elif mode == "mixed_chars_v1":
        # 计算实际消息内容及Schema，不重复计算外层JSON转义。
        parts = [prompt, json.dumps(data, ensure_ascii=False),
                 json.dumps(response_schema(role, data), ensure_ascii=False)]
        weight = 0.0
        for text in parts:
            ascii_count = 0
            for ch in text:
                if ord(ch) < 128:
                    ascii_count += 1
                elif '\u3000' <= ch <= '\u9fff' or '\uff00' <= ch <= '\uffef':
                    weight += 1.0
                else:
                    weight += len(ch.encode('utf-8'))
            weight += ascii_count / 3.0
        estimate = math.ceil(weight * 1.25) + 512
    else:
        raise ValueError("未知token_estimator")
    return dict(estimated_input_tokens=estimate, request_utf8_bytes=byte_count,
                token_estimator=mode, input_budget_tokens=input_budget(config))


class BudgetClient:
    """所有中心调用共用的发送前检查；不默默裁剪任何必需字段。"""

    def __init__(self, client: Any, config: dict):
        self.client, self.config = client, config

    def ask(self, role: str, prompt: str, data: dict) -> dict:
        check_request(self.config, role, prompt, data)
        return self.client.ask(role, prompt, data)


def compact_turn(turn: dict) -> dict:
    return {k: turn[k] for k in ("evidence_id", "occurred_at", "user_text", "assistant_text", "source_spans",
                                "source_kind", "photo_policy", "image_sha256", "image_asset") if k in turn}


def view(session: dict, turns: list[dict]) -> dict:
    return {"device_id": session["device_id"], "session_id": session["session_id"],
            "start": session["start"], "end": session["end"], "input_hash": session["input_hash"],
            "turns": turns, "context_scope": "bounded_window_of_session",
            **{k: session[k] for k in ("source_kind", "image_url") if k in session}}


def bisect_window(window: dict) -> tuple[dict, dict]:
    turns = window["turns"]
    if len(turns) > 1:
        middle = len(turns) // 2
        return view(window, turns[:middle]), view(window, turns[middle:])
    if not turns:
        raise ContextCapacityError("空窗口仍超预算：规则/候选/固定提示需要人工检查")
    turn = turns[0]
    # 超长单话轮按字段拆分；保留原证据ID及精确字符区间，不截断原数据。
    fields = [k for k in ("user_text", "assistant_text") if turn.get(k)]
    base = {k: v for k, v in turn.items() if k not in ("user_text", "assistant_text", "source_spans")}
    spans = turn.get("source_spans", {k: [0, len(turn.get(k, ""))] for k in fields})
    if len(fields) == 2:
        pieces = []
        for field in fields:
            pieces.append(dict(base, user_text=turn["user_text"] if field == "user_text" else "",
                               assistant_text=turn["assistant_text"] if field == "assistant_text" else "",
                               source_spans={field: spans[field]}))
    elif fields and len(turn[fields[0]]) > 1:
        field = fields[0]
        middle = len(turn[field]) // 2
        start, end = spans[field]
        pieces = []
        for text, interval in ((turn[field][:middle], [start, start + middle]),
                               (turn[field][middle:], [start + middle, end])):
            pieces.append(dict(base, user_text=text if field == "user_text" else "",
                               assistant_text=text if field == "assistant_text" else "",
                               source_spans={field: interval}))
    else:
        raise ContextCapacityError("最小证据片段仍超预算，未丢弃原文，等待调整规则或预算")
    return view(window, [pieces[0]]), view(window, [pieces[1]])


def windows(session: dict, budget: int, *, unit: str = "bytes") -> Iterator[dict]:
    """完整覆盖所有原始user/assistant字符；局部重叠仅在预算允许时增加。"""
    if type(budget) is not int or budget < 1 or unit not in ("bytes", "chars"):
        raise ValueError("窗口预算必须为正整数，单位必须为bytes或chars")
    def measure(window: dict) -> int:
        if unit == "chars":
            return sum(len(t.get(k, "")) for t in window["turns"] for k in ("user_text", "assistant_text"))
        return size(window)
    turns = []
    for original in session["turns"]:
        turn = compact_turn(original)
        if "source_spans" not in turn:
            turn["source_spans"] = {k: [0, len(turn.get(k, ""))] for k in ("user_text", "assistant_text")}
        turns.append(turn)
    pending = [view(session, turns)]
    previous = None
    while pending:
        current = pending.pop()
        if measure(current) > budget:
            left, right = bisect_window(current)
            pending.extend([right, left])
            continue
        # 不合并同ID的不同片段，避免字典证据校验覆盖。
        if previous and all(t["evidence_id"] != previous["evidence_id"] for t in current["turns"]):
            overlap = view(current, [previous] + current["turns"])
            if measure(overlap) <= budget:
                current = overlap
        if current["turns"]:
            previous = current["turns"][-1]
        yield current


def pages(items: list[dict], budget: int) -> Iterator[list[dict]]:
    page: list[dict] = []
    for item in items:
        if size([item]) > budget:
            raise ContextCapacityError("单项规则/候选超过页预算，不能截断其含义")
        if page and size(page + [item]) > budget:
            yield page
            page = []
        page.append(item)
    if page or not items:
        yield page


def bounded_history(items: list[dict], budget: int, limit: int) -> list[dict]:
    selected: list[dict] = []
    seen = set()
    for item in items:
        if item["memory_id"] in seen:
            continue
        seen.add(item["memory_id"])
        if len(selected) < limit and size(selected + [item]) <= budget:
            selected.append(item)
    return selected


def related_history(history: list[dict], candidate: dict, budget: int) -> list[dict]:
    target = candidate["target_memory_id"]
    required = [m for m in history if m["memory_id"] == target]
    for cited in candidate["evidence"]:
        for memory in history:
            if any(e["evidence_id"] == cited["evidence_id"] and cited["quote"] in e["data"]["user_text"]
                   for e in memory.get("evidence", [])):
                if memory not in required:
                    required.append(memory)
                break
    if size(required) > budget:
        raise ContextCapacityError("候选必需的历史目标或证据超预算，禁止静默删除")
    optional = [m for m in history if m["dimension_id"] == candidate["dimension_id"] and m not in required]
    return bounded_history(required + optional, budget, max(1, len(history)))


def compact_feedback(audit: dict | None) -> dict | None:
    if audit is None:
        return None
    # 保留有界的拒绝理由，否则补提取只能看到Skill名，会重复相同错误。
    return {"followup_skills": audit["followup_skills"],
            "not_assessed": audit["dimension_statuses"]["not_assessed"],
            "rejected": [{"candidate_id": r["candidate_id"], "reason": r["reason"][:240]}
                         for r in audit.get("rejected", [])[-8:]]}
