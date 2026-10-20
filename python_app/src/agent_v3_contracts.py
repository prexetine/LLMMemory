"""来源审核v3契约；来源可信度与历史维护决定分开。"""
from __future__ import annotations

from datetime import datetime
from agent_data import digest, exact_keys

VERSION = "source-maintenance-v3"
CHECKS = ("evidence", "subject", "dimension", "semantics", "scope", "atomicity", "history", "privacy")
META = {"subject_key", "attribute_key", "valid_from", "valid_to"}
BASE_FIELDS = {"dimension_id", "subject", "claim", "scope", "kind", "keywords", "evidence", "operation", "target_memory_id"}
FORBIDDEN = {"P01", "P05", "E02"} | {f"P{n:02d}" for n in range(8, 17)}


class ModelOutputError(ValueError):
    """模型返回无法通过JSON/候选协议；可隔离当前Session，不代表存储故障。"""


def confirmed_subject(key, device):
    if not isinstance(key, str):
        return False
    return key.strip().lower() not in {"", "user", "self", "device", "unknown", "speaker", "device_user",
        "unknown_subject", "当前用户", "用户", "当前发言者", "本人", "我", "儿童", "孩子", str(device).lower()} and not key.lower().startswith("device:")


def verify_evidence(items: list, session: dict) -> None:
    if not isinstance(items, list) or not items:
        raise ValueError("缺少来源证据")
    texts = {}
    for t in session["turns"]:
        texts.setdefault(t["evidence_id"], []).append(t["user_text"])
    for e in items:
        exact_keys(e, {"evidence_id", "quote"})
        if (not isinstance(e["evidence_id"], str) or not isinstance(e["quote"], str)
                or not e["quote"].strip() or not any(e["quote"] in t for t in texts.get(e["evidence_id"], []))):
            raise ValueError("引文缺失或不是对应用户原话")


def validate_candidate(c: dict, skill: str, registry: object, session: dict, window_id: str) -> dict:
    exact_keys(c, BASE_FIELDS | META)
    for k in sorted(BASE_FIELDS - {"keywords", "evidence"}):
        if not isinstance(c[k], str):
            raise ValueError(f"候选字段类型错误：{k} 必须为string，实际为{type(c[k]).__name__}；空字符串使用\"\"，不能用null")
    if c["dimension_id"] not in registry.dimensions or registry.dimensions[c["dimension_id"]]["owner_skill_id"] != skill:
        raise ValueError("维度不属于当前Skill")
    if c["dimension_id"] in FORBIDDEN:
        raise ValueError("该维度缺少统计/媒体来源")
    if not all(c[k].strip() for k in ("subject", "claim", "scope")) or len(c["claim"]) > 1500:
        raise ValueError("候选正文非法")
    if c["kind"] not in ("self_report", "observation", "hypothesis"):
        raise ValueError("候选类型非法")
    if c["operation"] != "new" or c["target_memory_id"]:
        raise ValueError("提取阶段不得决定历史更新")
    if not isinstance(c["keywords"], list) or not 1 <= len(c["keywords"]) <= 12 or any(not isinstance(w, str) or not w.strip() or len(w) > 60 for w in c["keywords"]):
        raise ValueError("关键词非法")
    for k in META:
        if c[k] is not None and (not isinstance(c[k], str) or not c[k].strip()):
            raise ValueError("归一标识/有效时间必须非空字符串或null")
    for k in ("valid_from", "valid_to"):
        if c[k]:
            datetime.fromisoformat(c[k])
    if c["valid_from"] and c["valid_to"]:
        start, end = datetime.fromisoformat(c["valid_from"]), datetime.fromisoformat(c["valid_to"])
        if (start.tzinfo is None) != (end.tzinfo is None):
            raise ValueError("有效时间基准不一致")
        if end < start:
            raise ValueError("有效时间倒置")
    verify_evidence(c["evidence"], session)
    photos = [t for t in session["turns"] if t.get("source_kind") == "photo"]
    if photos:
        if len(photos) != 1 or c["kind"] != "observation" or c["subject_key"] != photos[0]["evidence_id"]:
            raise ValueError("图片候选只能是以照片证据ID为主体的画面观察，不能推断用户身份或偏好")
        if c["valid_from"] is not None or c["valid_to"] is not None:
            raise ValueError("照片日志时间不能当作画面事实发生时间")
    if not confirmed_subject(c["subject_key"], session["device_id"]):
        c = dict(c, subject_key=None)
    result = dict(c, candidate_id=digest([skill, c])[:20], source_skill_id=skill,
                  skill_version=registry.version, source_windows=[window_id], candidate_version=1)
    return result


def validate_signal(s: dict, session: dict) -> None:
    exact_keys(s, {"kind", "subject_key", "attribute_key", "scope", "evidence"})
    if s["kind"] not in ("correction", "change", "denial", "withdrawal", "subject_switch"):
        raise ValueError("未知控制线索")
    if not isinstance(s["scope"], str):
        raise ValueError("控制线索scope非法")
    for k in ("subject_key", "attribute_key"):
        if s[k] is not None and (not isinstance(s[k], str) or not s[k].strip()):
            raise ValueError("控制线索标识非法")
    verify_evidence(s["evidence"], session)


def source_result(result: dict, candidates: list[dict], visible_ids: set[str]) -> None:
    exact_keys(result, {"results"})
    if not isinstance(result["results"], list):
        raise ValueError("审核结果必须为数组")
    known = {c["candidate_id"]: c for c in candidates}
    ids = []
    for r in result["results"]:
        exact_keys(r, {"candidate_id", "decision", "checks", "reason", "evidence_ids", "needs_context", "context_ids"})
        cid = r["candidate_id"]
        if not isinstance(cid, str) or cid not in known:
            raise ValueError("未知审核候选")
        ids.append(cid)
        if r["decision"] not in ("pass", "pending", "reject"):
            raise ValueError("非法来源结论")
        exact_keys(r["checks"], set(CHECKS))
        if any(v not in ("pass", "fail", "uncertain") for v in r["checks"].values()):
            raise ValueError("非法检查结论")
        if not isinstance(r["reason"], str) or not r["reason"].strip() or type(r["needs_context"]) is not bool:
            raise ValueError("审核理由/补充标志非法")
        for k in ("evidence_ids", "context_ids"):
            if not isinstance(r[k], list) or any(not isinstance(e, str) for e in r[k]) or len(set(r[k])) != len(r[k]):
                raise ValueError("审核引用格式非法")
        if not set(r["evidence_ids"]).issubset(visible_ids):
            raise ValueError("审核引用未发送证据")
        if r["decision"] == "pass":
            required = {e["evidence_id"] for e in known[cid]["evidence"]}
            if r["needs_context"] or set(r["checks"].values()) != {"pass"} or not required.issubset(r["evidence_ids"]):
                raise ValueError("来源通过必须逐项通过且覆盖全部证据")
            if known[cid]["kind"] == "hypothesis":
                raise ValueError("假设不得通过")
    if sorted(ids) != sorted(known):
        raise ValueError("批量审核缺失、重复或额外候选")


def seal(record: dict) -> dict:
    return dict(record, proof_hash=digest(record))


def verify_seal(record: dict) -> None:
    if record.get("proof_hash") != digest({k: v for k, v in record.items() if k != "proof_hash"}):
        raise ValueError("审核或维护凭证内容已改变")


def verify_source(proof: dict, candidate: dict, session: dict) -> None:
    verify_seal(proof)
    if (proof.get("policy") != VERSION or proof.get("candidate_hash") != digest(candidate)
            or proof.get("candidate_version") != candidate["candidate_version"]
            or proof.get("input_hash") != session["input_hash"] or proof.get("device_id") != session["device_id"]
            or proof.get("session_id") != session["session_id"]):
        raise ValueError("来源凭证不属于当前候选/会话")
    pack = proof["evidence_pack"]
    if proof["context_hash"] != digest(pack):
        raise ValueError("来源上下文指纹变化")
    original = {t["evidence_id"]: t for t in session["turns"]}
    for t in pack:
        if t["evidence_id"] not in original or t["user_text"] not in original[t["evidence_id"]]["user_text"]:
            raise ValueError("来源上下文不属于本会话原文")
        if original[t["evidence_id"]].get("source_kind") == "photo":
            if (t.get("source_kind") != "photo" or not t.get("image_sha256")
                    or t["image_sha256"] != original[t["evidence_id"]].get("image_sha256")):
                raise ValueError("图片审核证据哈希变化或来源类型丢失")
    keys = {"candidate_id", "decision", "checks", "reason", "evidence_ids", "needs_context", "context_ids"}
    source_result({"results": [{k: proof[k] for k in keys}]}, [candidate], {t["evidence_id"] for t in pack})
    if proof["decision"] == "pass":
        verify_evidence(candidate["evidence"], {"turns": pack})
        photos = [t for t in pack if t.get("source_kind") == "photo"]
        if photos and (len(photos) != 1 or candidate["kind"] != "observation"
                       or candidate.get("subject_key") != photos[0]["evidence_id"]
                       or candidate.get("valid_from") is not None or candidate.get("valid_to") is not None):
            raise ValueError("图片入库必须限定为无推测时间的照片画面观察")


def validate_comparison(result: dict, candidates: list[dict], history: list[dict], session: dict) -> None:
    exact_keys(result, {"results"})
    if not isinstance(result["results"], list):
        raise ValueError("维护结果必须为数组")
    known = {c["candidate_id"] for c in candidates}
    targets = {m["memory_id"]: m for m in history}
    seen = []
    for r in result["results"]:
        exact_keys(r, {"candidate_id", "action", "targets", "reason", "basis"})
        if r["candidate_id"] not in known:
            raise ValueError("维护未知候选")
        seen.append(r["candidate_id"])
        if r["action"] not in ("new", "reinforce", "coexist", "supersede", "conflict", "pending"):
            raise ValueError("未知维护操作")
        if not isinstance(r["reason"], str) or not r["reason"].strip() or not isinstance(r["targets"], list) or not isinstance(r["basis"], list):
            raise ValueError("维护字段非法")
        tids = []
        for t in r["targets"]:
            exact_keys(t, {"memory_id", "revision"})
            if not isinstance(t["memory_id"], str) or t["memory_id"] not in targets or type(t["revision"]) is not int or t["revision"] != targets[t["memory_id"]]["revision"]:
                raise ValueError("维护目标未发送或版本不匹配")
            tids.append(t["memory_id"])
        if len(tids) != len(set(tids)):
            raise ValueError("重复维护目标")
        if r["action"] in ("reinforce", "supersede", "conflict", "coexist") and not tids:
            raise ValueError("维护操作缺少目标")
        if r["action"] in ("new", "pending") and tids:
            raise ValueError("新增/待定不允许修改目标")
        if r["action"] == "reinforce" and len(tids) != 1:
            raise ValueError("强化必须唯一目标")
        if r["basis"]:
            verify_evidence(r["basis"], session)
        if r["action"] == "supersede" and not r["basis"]:
            raise ValueError("更新必须有明确更正/变化证据")
    if sorted(seen) != sorted(known):
        raise ValueError("历史比对缺失、重复或额外候选")
