"""有证据的经历明细与可重建的近期主题汇总；不新增模型或产品维度。"""
from __future__ import annotations

from datetime import datetime, timedelta
from agent_data import digest, exact_keys

TYPES = ("fact", "real_event", "learning_episode", "shared_work", "request", "visual")
CONTEXTS = ("real", "story", "game", "photo")
STEP_KINDS = ("attempt", "hint", "feedback", "contribution", "event", "context")


def validate_structure(c: dict, session: dict) -> dict:
    """明细逐字回指实际话轮；助手仅作题目、提示、反馈或叙事语境。"""
    detail = c.get("structure")
    if detail is None:  # 旧候选及离线兼容，不生成不存在的过程。
        return c
    exact_keys(detail, {"type", "context", "topic", "steps"})
    if detail["type"] not in TYPES or detail["context"] not in CONTEXTS:
        raise ValueError("记忆结构类型/场景非法")
    if not isinstance(detail["topic"], str) or len(detail["topic"]) > 60:
        raise ValueError("主题必须为60字内字符串，不确定留空")
    if not isinstance(detail["steps"], list):
        raise ValueError("经历步骤必须数组")
    turns = {}
    for turn in session["turns"]:
        turns.setdefault(turn["evidence_id"], []).append(turn)
    supported = {e["evidence_id"] for e in c["evidence"]}
    positions = {t["evidence_id"]: i for i, t in enumerate(session["turns"])}
    for step in detail["steps"]:
        exact_keys(step, {"kind", "text", "evidence_id", "speaker"})
        if step["kind"] not in STEP_KINDS or step["speaker"] not in ("user", "assistant"):
            raise ValueError("经历步骤角色/类型非法")
        if step["kind"] == "context" and step["speaker"] != "assistant":
            raise ValueError("context仅用于助手叙事语境")
        fragments = turns.get(step["evidence_id"], [])
        if not fragments or not isinstance(step["text"], str) or not step["text"].strip():
            raise ValueError("经历步骤缺少真实话轮")
        if not any(step["text"] in turn.get("user_text" if step["speaker"] == "user" else "assistant_text", "") for turn in fragments):
            raise ValueError("经历步骤引文与角色原文不一致")
        if step["speaker"] == "user" and step["evidence_id"] not in supported:
            raise ValueError("用户经历步骤未列入候选证据")
        if step["speaker"] == "assistant" and step["kind"] not in ("hint", "feedback", "context"):
            raise ValueError("助手内容只能作为提示、反馈或context语境，不能冒充用户作答/贡献")
    if detail["type"] in ("learning_episode", "shared_work"):
        if not any(s["speaker"] == "user" for s in detail["steps"]):
            raise ValueError("学习/共同作品必须有真实用户参与步骤")
        if c.get("fact", {}).get("type") != "event":
            raise ValueError("学习/共同作品应按事件保存，不是长期属性")
    # fact是通用记录展示，不与事件/当次需求的机器类型重复争夺判定权。
    expected = {"real_event": "event", "request": "request", "visual": "observation"}
    if detail["type"] in expected and c.get("fact", {}).get("type") != expected[detail["type"]]:
        raise ValueError("内容结构与事实类型不一致，需要整理")
    if detail["context"] in ("story", "game") and detail["type"] == "real_event":
        raise ValueError("故事/游戏不得记为现实事件")
    if detail["type"] == "visual" and (detail["context"] != "photo" or session.get("source_kind") != "photo"):
        raise ValueError("视觉结构必须来自照片Session")
    c["structure"] = dict(detail, steps=sorted(detail["steps"], key=lambda s: (positions[s["evidence_id"]], s["speaker"] == "assistant")))
    return c


def collect_step_evidence(c: dict, session: dict) -> dict:
    """从已精确匹配原文的用户步骤补齐引用；助手绝不加入用户证据。"""
    detail = c.get("structure")
    if detail is None:
        return c
    exact_keys(detail, {"type", "context", "topic", "steps"})
    if not isinstance(detail["steps"], list) or not isinstance(c.get("evidence"), list):
        raise ValueError("经历步骤/证据必须数组")
    refs = list(c["evidence"])
    for step in detail["steps"]:
        exact_keys(step, {"kind", "text", "evidence_id", "speaker"})
        if step["speaker"] != "user":
            continue
        if not isinstance(step["text"], str) or not step["text"].strip() or not any(
                t["evidence_id"] == step["evidence_id"] and step["text"] in t["user_text"] for t in session["turns"]):
            raise ValueError("用户经历步骤没有逐字原文依据")
        ref = {"evidence_id": step["evidence_id"], "quote": step["text"]}
        if ref not in refs:
            refs.append(ref)
    return dict(c, evidence=refs)


def rebuild_summaries(state: dict, as_of: str, days: int = 30, min_sessions: int = 2) -> None:
    """从有效来源重建精确主题汇总；不把重复访问提升为稳定喜好。"""
    if type(days) is not int or not 1 <= days <= 365 or type(min_sessions) is not int or min_sessions < 2:
        raise ValueError("汇总窗口须1至365天，独立Session阈值须至少2")
    prior = state.get("summaries", {})
    observations = [o for m in state["memories"].values() for o in m.get("observations", [])]
    end = max([datetime.fromisoformat(as_of)] + [datetime.fromisoformat(o["observed_at"]) for o in observations]
              + ([datetime.fromisoformat(state["summary_watermark"])] if state.get("summary_watermark") else []))
    state["summary_watermark"] = end.isoformat()
    start = end - timedelta(days=days)
    groups = {}
    for m in state["memories"].values():
        detail = m.get("structure", {})
        topic = detail.get("topic", "").strip()
        if m["state"] != "reviewed" or m.get("subject_role") != "current_user" or not topic or detail.get("context") == "photo":
            continue
        recent = [o for o in m.get("observations", []) if start <= datetime.fromisoformat(o["observed_at"]) <= end]
        if recent:
            key = digest([m["device_id"], m["subject_key"], detail["context"], topic])[:24]
            groups.setdefault(key, []).append((m, recent))
    updated = {}
    for key, members in groups.items():
        sessions = sorted({o["session_id"] for _, os in members for o in os})
        if len(sessions) < min_sessions:
            continue
        topic = members[0][0]["structure"]["topic"].strip()
        refs = [{"memory_id": m["memory_id"], "revision": m["revision"],
                 "evidence_ids": sorted({e for o in os for e in o["evidence_ids"]})} for m, os in members]
        row = {"summary_id": key, "status": "active", "topic": topic,
               "subject_key": members[0][0]["subject_key"], "context": members[0][0]["structure"]["context"],
               "claim": f"在观察窗口内，{len(sessions)}个独立会话有与“{topic}”相关的已审核记录。",
               "epistemic_type": "derived_observation", "window_start": start.isoformat(), "window_end": end.isoformat(),
               "session_ids": sessions, "source_refs": sorted(refs, key=lambda r: r["memory_id"]),
               "dimension_ids": sorted({d for m, _ in members for d in m.get("dimension_ids", [m["dimension_id"]])}),
               "boundary": "系统观察性汇总，不是用户明确自述的长期偏好；时间为日志观察时间，不是事实发生时间。"}
        old = prior.get(key)
        unchanged = old and {k: v for k, v in old.items() if k != "revision"} == row
        row["revision"] = old["revision"] if unchanged else (old["revision"] + 1 if old else 1)
        updated[key] = row
    # 失去支持时撤销派生结论，不让旧总结继续有效，也不修改原始记忆。
    for key, old in prior.items():
        if key not in updated:
            updated[key] = dict(old, status="inactive", revision=old["revision"] + (old["status"] != "inactive"))
    state["summaries"] = updated
