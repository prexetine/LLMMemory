"""读取已筛选日志与106项维度规则；不修改源数据。"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    """生成稳定内容指纹。"""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def exact_keys(value: Any, keys: set[str]) -> None:
    """拒绝缺失字段和额外动态键。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"字段不符合契约，预期 {sorted(keys)}")


SKILL_NAMES = {
    "profile_relations": "基础信息与人物关系",
    "interests_preferences": "兴趣偏好与能力",
    "events_plans": "事件经历与计划",
    "emotions_companionship": "情绪与陪伴需求",
    "behavior_patterns": "行为规律与跨来源关联",
    "product_interaction": "蛋宝互动与共同经历",
    "memory_control": "记忆纠正与使用边界",
    "exploratory_findings": "探索性发现",
}


def owner(dim: str) -> str:
    """显式按产品含义分配唯一主责，不用关键词猜归属。"""
    group, n = dim[0], int(dim[1:])
    if group == "G":
        if n <= 15 or n == 45:
            return "profile_relations"
        if n in (18, 30):
            return "emotions_companionship"
        if 16 <= n <= 24 or 42 <= n <= 44 or n == 46:
            return "interests_preferences"
        if 25 <= n <= 29 or 31 <= n <= 34:
            return "events_plans"
        return "behavior_patterns"
    if group == "P":
        if n == 3:
            return "memory_control"
        if n >= 8:
            return "behavior_patterns"
        return "product_interaction"
    if n <= 4 or 13 <= n <= 19 or n in (42, 43):
        return "emotions_companionship"
    if 20 <= n <= 29:
        return "memory_control"
    if 5 <= n <= 12 or 30 <= n <= 36:
        return "product_interaction"
    if 37 <= n <= 41:
        return "behavior_patterns"
    return "exploratory_findings"


class Registry:
    """从已有逐维度方案冻结规则，提供轻量目录与按需加载。"""

    def __init__(self, path: Path):
        text = path.read_text(encoding="utf-8")
        matches = list(re.finditer(r"^#### ([GPE]\d{2}) (.+)$", text, re.M))
        self.dimensions: dict[str, dict[str, str]] = {}
        for match in matches:
            end = re.search(r"^#{1,4} ", text[match.end():], re.M)
            body = text[match.end():match.end() + end.start()] if end else text[match.end():]
            dim = match.group(1)
            if dim in self.dimensions:
                raise ValueError("维度重复")
            self.dimensions[dim] = {
                "dimension_id": dim, "name": match.group(2).strip(),
                "owner_skill_id": owner(dim), "rules": body.strip(),
            }
        expected = {f"{p}{n:02d}" for p, count in (("G", 46), ("P", 16), ("E", 44))
                    for n in range(1, count + 1)}
        if set(self.dimensions) != expected:
            raise ValueError("规则必须恰好覆盖106项")
        self.version = digest(self.dimensions)

    def catalog(self) -> list[dict[str, Any]]:
        return [{"skill_id": skill, "name": name,
                 "dimensions": {k: v["name"] for k, v in self.dimensions.items()
                                if v["owner_skill_id"] == skill}}
                for skill, name in SKILL_NAMES.items()]

    def load(self, skill: str) -> list[dict[str, str]]:
        if skill not in SKILL_NAMES:
            raise ValueError("未知Skill")
        return [v for v in self.dimensions.values() if v["owner_skill_id"] == skill]


def load_user_roster(root: Path) -> list[dict]:
    """兼容旧核心名单与独立分批名单；旧名单存在时保留原选择范围。"""
    path = root / "core_50_provisional.csv"
    if not path.is_file():
        path = root / "users.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row.get("device_id", "") for row in rows]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", d) for d in ids):
        raise ValueError("用户名单为空、重复或含非法设备ID")
    return rows


def load_sessions(root: Path, device: str, gap_minutes: int = 30) -> tuple[list[dict], dict]:
    """优先chat_id；缺失时按30分钟间隔切分，并保留逐行原始引用。"""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", device):
        raise ValueError("非法设备ID")
    selected = next((r for r in load_user_roster(root) if r["device_id"] == device), None)
    if selected is None:
        raise ValueError("设备不在当前数据目录的用户名单中")
    path = root / "candidate_data" / device / "dialogue.jsonl.gz"
    records, excluded, seen = [], {}, {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            reason = item["disposition"]
            if reason != "eligible":
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            row = item["data"]
            if row.get("device_id") != device or str(row.get("status")) != "0" or str(row.get("del_flag")) != "0":
                raise ValueError("设备归属或正常状态不符合已确认口径")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("eligible记录无用户文本")
            timestamp = datetime.fromisoformat(item["occurred_at"])
            if timestamp.tzinfo is None:
                raise ValueError("时间缺少时区")
            evidence_id = f"{device}:dialogue:{row.get('dialog_id') or item['row']}"
            record = {
                "evidence_id": evidence_id, "occurred_at": timestamp.isoformat(),
                "user_text": row["text"], "assistant_text": row.get("content") or "",
                "source_file": item["source_file"], "sheet": item["sheet"],
                "source_row": item["row"], "chat_id": row.get("chat_id") or None,
            }
            if evidence_id in seen:
                if seen[evidence_id] != record:
                    raise ValueError("相同证据ID出现冲突内容")
                continue
            seen[evidence_id] = record
            records.append(record)
    records.sort(key=lambda x: (x["occurred_at"], x["evidence_id"]))
    groups: dict[str, list] = {}
    fallback, previous = 0, None
    for record in records:
        now = datetime.fromisoformat(record["occurred_at"])
        if record["chat_id"]:
            sid = "chat-" + str(record["chat_id"])
        else:
            if previous is None or now - previous > timedelta(minutes=gap_minutes):
                fallback += 1
            sid = f"gap-{fallback}"
        previous = now if not record["chat_id"] else None
        groups.setdefault(sid, []).append(record)
    sessions = [{
        "device_id": device, "session_id": sid, "turns": turns,
        "start": turns[0]["occurred_at"], "end": turns[-1]["occurred_at"],
        "input_hash": digest(turns), "session_boundary": "chat_id" if sid.startswith("chat-") else "time_gap",
    } for sid, turns in groups.items()]
    sessions.sort(key=lambda s: (s["start"], s["session_id"]))
    return sessions, {"device_id": device, "selection": selected, "eligible_turns": len(records),
                      "sessions": len(sessions), "excluded": excluded,
                      "source_path": str(path), "media_processed": False}
