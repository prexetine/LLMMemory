"""事实驱动记忆的纯数据规则；不依赖产品维度编号。"""
from __future__ import annotations

import re
from typing import Any

from agent_data import digest


FACT_TYPES = {"attribute", "event", "request", "observation", "relation"}
SOURCE_DECISIONS = {"pass", "pending", "reject"}
RELATIONS = {
    "equivalent", "related_but_independent", "changed_with_direction",
    "contradictory", "insufficient_evidence",
}


def fact_text(record: dict[str, Any]) -> str:
    """返回用于语义召回的最小事实表示，不编码产品维度。"""
    fact = record.get("fact", {})
    return " | ".join(str(value) for value in (
        record.get("subject", ""), fact.get("type", ""), fact.get("predicate", ""),
        fact.get("object", ""), fact.get("value", ""), fact.get("polarity", ""),
        record.get("scope", ""), fact.get("time_text", ""),
    ) if value)


def content_key(record: dict[str, Any]) -> str:
    """事实身份只取可比较的语义字段，绝不取 dimension_id。"""
    fact = record.get("fact", {})
    scoped = fact.get("type") in {"event", "request", "observation"} or record.get("scope") in {
        "session", "本次会话", "当次活动", "bounded_window_of_session"}
    identity = None
    if scoped:
        identity = {"session": record.get("origin_session_id") or record.get("session_id"),
                    "evidence": sorted(e["evidence_id"] for e in record.get("evidence", []))}
    return digest({
        "identity_version": 2, "source_identity": identity,
        "device_id": record.get("device_id"),
        "subject_key": record.get("subject_key"),
        "type": fact.get("type"), "predicate": fact.get("predicate"),
        "object": fact.get("object"), "value": fact.get("value"),
        "polarity": fact.get("polarity"), "scope": record.get("scope"),
        "time_text": fact.get("time_text", ""), "context": record.get("structure", {}).get("context", "real"),
    })[:32]


def facets(record: dict[str, Any]) -> dict[str, str]:
    """从已审核事实生成可解释的受控组合，不把它当入库门禁。"""
    fact = record.get("fact", {})
    predicate = str(fact.get("predicate", "")).strip()
    value = str(fact.get("value", "")).strip()
    mode = record.get("assertion_mode", "self_report")
    if mode == "request" or fact.get("type") == "request":
        category = "user_need"
    elif mode == "observation" or fact.get("type") == "observation":
        category = "observation"
    elif fact.get("type") == "event":
        category = "event_experience"
    elif record.get("subject_role") == "other":
        category = "relationship_social"
    elif any(word in predicate + value for word in ("喜欢", "讨厌", "害怕", "想要", "偏好")):
        category = "preference_attitude"
    elif any(word in predicate + value for word in ("会", "不会", "擅长", "学会")):
        category = "ability_growth"
    else:
        category = "user_fact"
    polarity = str(fact.get("polarity", "positive"))
    temporal = "temporary" if mode == "temporary" or record.get("scope") in {"session", "本次会话", "当次活动"} else "ongoing"
    return {"category": category, "domain": str(fact.get("object", "") or "general"),
            "attribute": predicate or "statement", "polarity": polarity, "temporal": temporal}


def deterministic_relation(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    """只处理完全可证明的关系；其余交给召回后的语义判断。"""
    if left.get("content_key") and left.get("content_key") == right.get("content_key"):
        return "equivalent"
    lf, rf = left.get("fact", {}), right.get("fact", {})
    same_subject = left.get("device_id") == right.get("device_id") and left.get("subject_key") == right.get("subject_key")
    same_property = (lf.get("type"), lf.get("predicate"), lf.get("object"), left.get("scope")) == (
        rf.get("type"), rf.get("predicate"), rf.get("object"), right.get("scope"))
    if (same_subject and same_property and lf.get("time_text") and lf.get("time_text") == rf.get("time_text")
            and lf.get("value") == rf.get("value") and lf.get("polarity") != rf.get("polarity")):
        return "contradictory"
    return None


def projection_state(record: dict[str, Any], support_count: int, distinct_dates: int) -> str:
    """投影状态只在 Session 收口后使用，事实本身不因此被删除。"""
    facet = record.get("facets") or facets(record)
    if record.get("source_decision") != "pass":
        return "provisional"
    if facet["category"] in {"observation", "user_need"}:
        return "episode"
    if support_count >= 2 and distinct_dates >= 2:
        return "stable"
    return "exploring"


def map_legacy_dimensions(record: dict[str, Any]) -> list[str]:
    """只映射明确的结构化谓词，不从正文单字猜测分类。"""
    fact = record.get("fact", {})
    predicate = str(fact.get("predicate", "")).strip()
    if not fact.get("object") or fact.get("type") == "observation":
        return []
    if predicate in {"喜欢", "喜爱", "爱好", "爱"} and fact.get("type") == "attribute":
        return ["G17" if fact.get("polarity") == "negative" else "G16"]
    if predicate in {"不喜欢", "讨厌"} and fact.get("type") == "attribute":
        return ["G17"] if fact.get("polarity") != "negative" else []
    mapping = {"想要": "G19", "希望": "G19", "愿望": "G19",
               "去过": "G25", "去了": "G25", "到访": "G25",
               "会": "G22", "会做": "G22", "能够": "G22", "学会": "G26"}
    return [mapping[predicate]] if predicate in mapping and fact.get("polarity") != "negative" else []
