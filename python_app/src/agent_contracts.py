"""公共事实与审核契约；存储协议版本保留，执行流程独立版本化。"""
from __future__ import annotations

from datetime import datetime
import re
from agent_data import digest, exact_keys
from agent_fact_policy import normalize_fact, retention
from agent_memory_structure import validate_structure, collect_step_evidence

VERSION = "source-maintenance-v4"
CHECKS = ("evidence", "semantics", "subject")
META = {"subject_key", "attribute_key", "valid_from", "valid_to"}
BASE_FIELDS = {"dimension_id", "subject", "claim", "scope", "kind", "keywords", "evidence", "operation", "target_memory_id", "subject_role", "assertion_mode"}
FORBIDDEN = {"P01", "P05", "E02"} | {f"P{n:02d}" for n in range(8, 17)}


from agent_v3_contracts import ModelOutputError


class NonMemoryControl(ValueError):
    """普通播放/设备指令，不是记忆保存控制。"""


def confirmed_subject(key, device):
    if not isinstance(key, str):
        return False
    return key.strip().lower() not in {"", "user", "self", "device", "unknown", "speaker", "device_user",
        "unknown_subject", "当前用户", "用户", "当前发言者", "本人", "我", "儿童", "孩子", str(device).lower()} and not key.lower().startswith("device:")


def subject_id(device, label=None):
    if label is not None:
        label = {"我爸爸": "爸爸", "我的爸爸": "爸爸", "父亲": "爸爸", "我妈妈": "妈妈", "我的妈妈": "妈妈", "母亲": "妈妈"}.get(label.strip(), label.strip())
    return "account:" + device + (":self" if label is None else ":person:" + digest(label.strip())[:20])


def normalize_signal(s, session):
    s = dict(s)
    label = s["subject_key"]
    if s["kind"] == "subject_switch" and label and not s["target_text"]:
        s["target_text"] = label
    if label in (None, "我", "用户", "当前用户", "self", "user"):
        s["subject_key"] = subject_id(session["device_id"])
    elif not label.startswith("account:" + session["device_id"] + ":"):
        s["subject_key"] = subject_id(session["device_id"], label)
    s.update(device_id=session["device_id"], session_id=session["session_id"])
    return s


def explicit_control(s):
    text = " ".join(e["quote"] for e in s["evidence"])
    return bool(re.search(r"(?:不要|别|不许|不用|不能|禁止).{0,12}(?:记住|记下|记录|保存|记着)|(?:删除|清除|忘掉|忘记).{0,20}(?:记忆|记录|这件事|这段|刚才)|(?:全部|所有|什么都|任何).{0,8}(?:不要记|别记)", text))


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
    exact_keys(c, BASE_FIELDS | META | ({"structure"} if "structure" in c else set()) | ({"fact"} if "fact" in c else set()) | ({"retention", "retention_reason"} if "retention" in c or "retention_reason" in c else set()))
    if "retention" in c and (c["retention"] not in ("memory", "event", "session_record", "discard") or not isinstance(c["retention_reason"], str) or not c["retention_reason"].strip()):
        raise ValueError("保存去向或价值说明非法")
    for k in sorted(BASE_FIELDS - {"keywords", "evidence"}):
        if not isinstance(c[k], str):
            raise ValueError(f"候选字段类型错误：{k} 必须为string，实际为{type(c[k]).__name__}；空字符串使用\"\"，不能用null")
    if c["dimension_id"] != "UNCLASSIFIED" and (c["dimension_id"] not in registry.dimensions or registry.dimensions[c["dimension_id"]]["owner_skill_id"] != skill):
        raise ValueError("维度不属于当前Skill")
    if c["dimension_id"] in FORBIDDEN:
        raise ValueError("该维度缺少统计/媒体来源")
    if not all(c[k].strip() for k in ("subject", "claim", "scope")) or len(c["claim"]) > 1500:
        raise ValueError("候选正文非法")
    if c["kind"] not in ("self_report", "observation", "hypothesis"):
        raise ValueError("候选类型非法")
    if c["subject_role"] not in ("current_user", "other", "unresolved") or c["assertion_mode"] not in ("self_report", "request", "temporary", "observation"):
        raise ValueError("主体角色或表达类型非法")
    c = dict(c)
    c["attribute_key"] = c["attribute_key"] or c["dimension_id"]
    if c["subject_role"] == "current_user":
        c["subject_key"] = subject_id(session["device_id"])
    elif c["subject_role"] == "other":
        if c["subject"].strip() in ("他", "她", "他们", "别人", "另一个人", "我", "用户") or re.search(r"未明确|未确定|未识别|未指明|不明确|不确定|未知|身份不明|不明身份|某个人|某人|不详|无法确定", c["subject"]):
            c.update(subject_role="unresolved", subject_key=None)
        else:
            c["subject_key"] = subject_id(session["device_id"], c["subject"])
    else:
        c["subject_key"] = None
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
    c = collect_step_evidence(c, session)
    verify_evidence(c["evidence"], session)
    photos = [t for t in session["turns"] if t.get("source_kind") == "photo"]
    if photos:
        if len(photos) != 1 or c["kind"] != "observation" or c["assertion_mode"] != "observation" or c["subject_role"] != "other":
            raise ValueError("图片候选只能是以照片证据ID为主体的画面观察，不能推断用户身份或偏好")
        c["subject_key"] = photos[0]["evidence_id"]
        if c["valid_from"] is not None or c["valid_to"] is not None:
            raise ValueError("照片日志时间不能当作画面事实发生时间")
    if not confirmed_subject(c["subject_key"], session["device_id"]):
        c = dict(c, subject_key=None)
    c = normalize_fact(c, session)
    c = validate_structure(c, session)
    c["retention"], c["retention_reason"] = retention(c)
    result = dict(c, candidate_id=digest([skill, c])[:20], source_skill_id=skill,
                  skill_version=registry.version, source_windows=[window_id], candidate_version=1,
                  device_id=session["device_id"], origin_session_id=session["session_id"])
    return result


def validate_signal(s: dict, session: dict) -> None:
    expected = {"kind", "subject_key", "attribute_key", "scope", "evidence", "target_type", "target_text"}
    if "switch_basis" in s or "switch_evidence" in s:
        expected |= {"switch_basis", "switch_evidence"}
        if s["switch_basis"] not in ("none", "handoff", "self_identification"):
            raise ValueError("换人依据类型非法")
        if s["switch_evidence"] is not None:
            verify_evidence([s["switch_evidence"]],session)
    if "device_id" in s or "session_id" in s:
        expected |= {"device_id", "session_id"}
        if s.get("device_id") != session["device_id"] or s.get("session_id") != session["session_id"]:
            raise ValueError("控制线索越过用户/会话")
    exact_keys(s, expected)
    if s["kind"] not in ("correction", "change", "denial", "withdrawal", "subject_switch"):
        raise ValueError("未知控制线索")
    if not isinstance(s["scope"], str):
        raise ValueError("控制线索scope非法")
    for k in ("subject_key", "attribute_key"):
        if s[k] is not None and (not isinstance(s[k], str) or not s[k].strip()):
            raise ValueError("控制线索标识非法")
    verify_evidence(s["evidence"], session)
    if s["target_type"] not in ("item", "session", "all", "unresolved") or not isinstance(s["target_text"], str):
        raise ValueError("控制目标格式错误")
    if s["kind"] == "withdrawal":
        if not explicit_control(s):
            raise NonMemoryControl("不是明确记忆保存指令：暂停、关机、语言要求、想不起来不属于撤回")
        quote = " ".join(e["quote"] for e in s["evidence"])
        if s["target_type"] == "all" and not re.search("全部|所有|任何|什么都", quote):
            raise ValueError("全部撤回缺少明确全部依据")
        if s["target_type"] == "session" and not re.search("这次|这段|本次|刚才|今天.*(?:对话|聊天)", quote):
            raise ValueError("当前会话撤回缺少会话范围依据")
        if s["target_type"] == "item" and (not s["target_text"].strip() or s["target_text"] not in quote or s["target_text"] in ("这件事", "这个", "刚才", "内容", "记录", "记忆")):
            raise ValueError("具体撤回目标必须有逐字原话依据；无法确定使用unresolved")


def explicit_subject_switch(s: dict) -> bool:
    """保守验证显式交接，不把提及第三人或单纯职业/姓名自述当成换人。"""
    basis, evidence = s.get("switch_basis"), s.get("switch_evidence")
    if basis not in ("handoff", "self_identification") or not evidence:
        return False
    text = evidence["quote"]
    if basis == "handoff":
        return bool(re.search(r"(?:换|轮到|现在由|接下来由|接着由|让.{0,12}来)(?:.{0,16})(?:说|聊|讲|我|他|她|爸爸|妈妈|哥哥|姐姐|弟弟|妹妹)",text))
    label = s.get("target_text") or s.get("subject_key")
    return bool(label and label in text and re.search(r"(?:现在|换|刚才.{0,12}不是|不是.{0,12}[，,。]).{0,16}(?:我是|我叫|是我)",text))


def source_result(result: dict, candidates: list[dict], visible_ids: set[str], *, normalize: bool = False) -> None:
    exact_keys(result, {"results"})
    if not isinstance(result["results"], list):
        raise ValueError("审核结果必须为数组")
    known = {c["candidate_id"]: c for c in candidates}
    ids = []
    for r in result["results"]:
        exact_keys(r, {"candidate_id", "decision", "checks", "reason", "evidence_ids", "needs_context", "context_ids", "repair"})
        exact_keys(r["repair"], {"kind", "reason"})
        if r["repair"]["kind"] not in ("none", "split", "reclassify", "narrow") or not isinstance(r["repair"]["reason"], str):
            raise ValueError("修订建议非法")
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
            problems = []
            if known[cid]["subject_role"] == "unresolved":
                problems.append("主体指代未解决")
            if r["needs_context"] or set(r["checks"].values()) != {"pass"} or not required.issubset(r["evidence_ids"]):
                problems.append("来源检查/证据覆盖/上下文不足")
            if known[cid]["kind"] == "hypothesis":
                problems.append("假设不能成为确定事实")
            if problems:
                if not normalize:
                    raise ValueError(f"candidate={cid}：" + "；".join(problems))
                r.update(decision="pending", reason=r["reason"] + "；程序逐条保守处理：" + "；".join(problems))
                if known[cid]["subject_role"] == "unresolved":
                    r["checks"]["subject"] = "uncertain"
                if not required.issubset(r["evidence_ids"]):
                    r["checks"]["evidence"] = "uncertain"
                if r["needs_context"] or known[cid]["kind"] == "hypothesis":
                    r["checks"]["semantics"] = "uncertain"
    if sorted(ids) != sorted(known):
        raise ValueError("批量审核缺失、重复或额外候选")


def seal(record: dict) -> dict:
    return dict(record, proof_hash=digest(record))


def verify_seal(record: dict) -> None:
    if record.get("proof_hash") != digest({k: v for k, v in record.items() if k != "proof_hash"}):
        raise ValueError("审核或维护凭证内容已改变")


def source_matches(proof: dict, candidate: dict) -> bool:
    """新来源凭证只绑定事实；旧凭证仍按原完整候选校验。"""
    if candidate.get("flow_version") in ("generic-flow-v2", "generic-flow-v3-relations"):
        from agent_flow_data import content_hash
        return proof.get("source_content_hash") == content_hash(candidate) and proof.get("candidate_id") == candidate.get("candidate_id")
    return proof.get("candidate_hash") == digest(candidate)


def verify_source(proof: dict, candidate: dict, session: dict) -> None:
    verify_seal(proof)
    if proof.get("source_content_hash"):
        from agent_flow_data import content_hash
        if proof["source_content_hash"] != content_hash(candidate):
            raise ValueError("分类之外的事实内容已改变，必须重新审核")
    if (proof.get("policy") != VERSION or not source_matches(proof,candidate)
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
    keys = {"candidate_id", "decision", "checks", "reason", "evidence_ids", "needs_context", "context_ids", "repair"}
    source_result({"results": [{k: proof[k] for k in keys}]}, [candidate], {t["evidence_id"] for t in pack})
    if proof["decision"] == "pass":
        verify_evidence(candidate["evidence"], {"turns": pack})
        # 经历明细属于审核内容，不能在提交时绕开角色/引文校验。
        validate_structure(dict(candidate), session)
        validate_structure(dict(candidate), dict(session, turns=pack))
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
        exact_keys(r, {"candidate_id", "action", "targets", "reason", "basis"} | ({"related_memory_ids"} if "related_memory_ids" in r else set()))
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
        related = r.get("related_memory_ids", [])
        if not isinstance(related, list) or any(not isinstance(i, str) or i not in targets for i in related) or len(set(related)) != len(related):
            raise ValueError("参考历史ID非法或未发送")
        if r["action"] in ("reinforce", "supersede", "conflict", "coexist") and not tids:
            raise ValueError("维护操作缺少目标")
        if r["action"] in ("new", "pending") and tids:
            raise ValueError("新增/待定不允许修改目标")
        # 多目标reinforce表示候选及这些目标彼此同义；事务层归并证据和版本。
        if r["basis"]:
            verify_evidence(r["basis"], session)
        if r["action"] == "supersede" and not r["basis"]:
            raise ValueError("更新必须有明确更正/变化证据")
    if sorted(seen) != sorted(known):
        raise ValueError("历史比对缺失、重复或额外候选")
