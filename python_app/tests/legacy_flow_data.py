"""通用流程的数据投影：证据定位、事实比较和分类标签互不替代。"""
from __future__ import annotations
import copy
from agent_data import digest, exact_keys

FLOW = "generic-flow-v1"


def validate_flow(config: dict, previous: dict | None = None) -> str:
    """统一运行边界；旧配置可读可跑，但不能混用新引擎或续写版本。"""
    flow = config.get("flow_version")
    if flow not in (None, FLOW):
        raise ValueError(f"未知flow_version={flow!r}，禁止静默回退旧流程")
    if flow == FLOW and config.get("review_strategy") != "source_v4":
        raise ValueError("generic-flow-v1必须使用review_strategy=source_v4，禁止误走旧引擎")
    resolved = flow or "legacy"
    if previous is not None and previous.get("flow_version", "legacy") != resolved:
        raise ValueError("运行流程版本不一致，禁止跨流程续写；旧结果仅保留读取，请使用新输出目录")
    return resolved


class EvidenceReferenceError(ValueError):
    """可追溯的证据定位错误；不通过模糊匹配掩盖来源问题。"""

    def __init__(self, code: str, evidence_id: str, message: str):
        self.code = code
        self.evidence_id = evidence_id
        super().__init__(f"[{code}] evidence_id={evidence_id}: {message}")


def semantic_config(config: dict) -> dict:
    """容量与重试调度可调，模型、提示策略、输入选择仍锁定。"""
    scheduling={"timeout_seconds","network_retries","model_output_attempts","history_context_tokens",
                "source_review_batch_size","context_window_tokens","context_safety_tokens","parallel_users"}
    return {k:v for k,v in config.items() if k not in scheduling}


def materialize(raw: dict, session: dict) -> dict:
    """将ID/角色/字符位置转换为原文；绝不模糊修正引文。"""
    c = copy.deepcopy(raw)
    by_id = {}
    for turn in session["turns"]:
        by_id.setdefault(turn["evidence_id"], []).append(turn)
    def resolve(ref: dict, user_only: bool = False) -> tuple[str, str]:
        exact_keys(ref, {"evidence_id", "speaker", "start", "end"})
        eid = ref["evidence_id"]
        def fail(code: str, message: str) -> None:
            raise EvidenceReferenceError(code, str(eid), message)
        if not isinstance(eid, str) or not eid:
            fail("invalid_id", "证据ID必须是非空字符串")
        if ref["speaker"] not in ("user", "assistant") or user_only and ref["speaker"] != "user":
            fail("wrong_speaker", "事实证据必须引用用户；助手仅作上下文")
        turns = by_id.get(eid, [])
        if not turns:
            fail("id_not_in_context", "当前证据包没有此ID；不代表原始数据库不存在")
        field = ref["speaker"] + "_text"
        start, end = ref["start"], ref["end"]
        if (start is None) != (end is None) or start is not None and (type(start) is not int or type(end) is not int or start < 0 or end <= start):
            fail("invalid_span", "引用位置必须同时为空或有效半开字符区间")
        if not any(t.get(field, "").strip() for t in turns):
            fail("empty_speaker_text", "指定角色没有可引用原文")
        found = []
        for t in turns:
            text = t.get(field, "")
            left, right = t.get("source_spans", {}).get(field, [0, len(text)])
            if start is None:
                if text.strip(): found.append(text)
            elif left <= start < end <= right:
                found.append(text[start-left:end-left])
        if not found:
            fail("span_not_visible", "引用超出当前可见片段；不能跨片段拼接")
        if len(set(found)) != 1:
            fail("ambiguous_fragment", "同一ID有多个不同片段，请指定可见字符范围")
        if not found[0].strip():
            fail("empty_quote", "定位到的引文为空白")
        return ref["evidence_id"], found[0]
    c["evidence"] = [dict(zip(("evidence_id", "quote"), resolve(e, True))) for e in c["evidence"]]
    for step in c.get("structure", {}).get("steps", []):
        exact_keys(step, {"kind", "reference"})
        ref = step["reference"]
        eid, text = resolve(ref)
        kind = step["kind"]
        if ref["speaker"] == "assistant" and kind not in ("hint", "feedback"):
            raise ValueError("助手步骤只能提供题目、提示或反馈")
        step.clear(); step.update(kind=kind, evidence_id=eid, text=text, speaker=ref["speaker"])
    return c


def compact_fact(c: dict) -> dict:
    """比较不复制原文/步骤/审核记录，不截断影响语义的事实字段。"""
    keys = ("candidate_id", "memory_id", "revision", "device_id", "subject_key", "claim", "fact",
            "scope", "valid_from", "valid_to", "last_seen", "state", "origin_session_id",
            "assertion_mode", "source_order", "attribute_key", "retention")
    result = {k: c[k] for k in keys if k in c}
    result["context"] = c.get("structure", {}).get("context", "real")
    result["evidence_ids"] = sorted({e["evidence_id"] for e in c.get("evidence", [])})
    return result


def content_hash(c: dict) -> str:
    """内容指纹不包含可独立更新的产品标签。"""
    return digest({k: v for k, v in c.items() if k not in
                   ("dimension_id", "dimension_ids", "classification", "candidate_id", "source_skill_id", "source_skill_ids", "source_windows")})


def source_pairs(candidates: list[dict]) -> list[tuple[dict, dict]]:
    """同源候选召回不依赖维度或关键词特例；是否等价由语义裁决。"""
    pairs = []
    for i, a in enumerate(candidates):
        for b in candidates[i+1:]:
            if (a.get("subject_key") and a["subject_key"] == b.get("subject_key")
                    and a.get("device_id") == b.get("device_id")
                    and a.get("origin_session_id") == b.get("origin_session_id")
                    and {e["evidence_id"] for e in a["evidence"]} & {e["evidence_id"] for e in b["evidence"]}):
                pairs.append((a,b))
    return pairs


def verify_consistency(candidates: list[dict], proofs: list[dict], audit: list[dict]) -> None:
    """事务入口验证关系裁决完整且未绕过来源分歧；不信任展示状态。"""
    pairs={digest([a["candidate_id"],b["candidate_id"]])[:24]:(a,b) for a,b in source_pairs(candidates)
           if a.get("retention")!="discard" and b.get("retention")!="discard"}
    if len(audit)!=len(pairs) or {r["pair_id"] for r in audit}!=set(pairs):
        raise ValueError("同源关系裁决遗漏或重复")
    byid={p["candidate_id"]:p for p in proofs}
    for row in audit:
        expected=[c["candidate_id"] for c in pairs[row["pair_id"]]]
        if row["candidate_ids"]!=expected or row["relation"] not in ("equivalent","distinct","incompatible","uncertain"):
            raise ValueError("同源关系裁决目标错误")
        ps=[byid[i] for i in expected]
        if row["relation"] in ("incompatible","uncertain") and any(p["decision"]=="pass" for p in ps):
            raise ValueError("未解决同源分歧不得来源通过")
        if row["relation"]=="equivalent" and any(p["decision"]!="pass" and any(p["checks"][k]!="pass" for k in ("evidence","semantics","subject")) for p in ps) and any(p["decision"]=="pass" for p in ps):
            raise ValueError("等价结论不能绕过实质审核分歧")
