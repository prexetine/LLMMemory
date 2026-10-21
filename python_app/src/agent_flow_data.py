"""通用流程的数据投影：证据定位、事实比较和分类标签互不替代。"""
from __future__ import annotations
import copy
from agent_data import digest, exact_keys

FLOW = "generic-flow-v2"
PROCESSING_REVISION = "contract-fix-2026-10-13-1"


def validate_flow(config: dict, previous: dict | None = None) -> str:
    """生产只允许当前执行版本；旧配置不自动升级。"""
    flow = config.get("flow_version")
    if config.get("dimension_mapping") not in (None, "fact-dimensions-106-v1"):
        raise ValueError("未知106维映射版本")
    if type(config.get("dimension_mapping_batch_size", 8)) is not int or config.get("dimension_mapping_batch_size", 8) < 1:
        raise ValueError("dimension_mapping_batch_size必须为正整数")
    if flow == "fact-driven-v1" and config.get("session_protocol") != "session-reliable-v2":
        raise ValueError("旧事实配置停止执行；请使用 memory_agent_reliable_v2_* 配置及独立输出目录")
    if flow == "fact-driven-v1" and previous and previous.get("session_protocol") != "session-reliable-v2":
        raise ValueError("旧事实报告禁止跨协议续跑，请使用独立输出目录")
    if config.get('token_estimator','utf8_bytes_legacy') not in ('utf8_bytes_legacy','mixed_chars_v1'):
        raise ValueError('未知token_estimator')
    if config.get('extraction_execution') not in (None,'isolated-skill-batch-repair-v1'):
        raise ValueError('未知extraction_execution')
    if type(config.get('extraction_repair_batch_size',8)) is not int or config.get('extraction_repair_batch_size',8)<1:
        raise ValueError('extraction_repair_batch_size必须是正整数')
    if "session_chunk_chars" in config and (type(config["session_chunk_chars"]) is not int or config["session_chunk_chars"] < 1):
        raise ValueError("session_chunk_chars必须为正整数（对话正文字符数）")
    if config.get('relation_policy') not in (None,'sparse-relations-v2'):
        raise ValueError('未知relation_policy')
    for key,default in (('relation_warn_tasks',2000),('relation_warn_batches',200),('relation_max_batches',600)):
        if type(config.get(key,default)) is not int or config.get(key,default)<1:
            raise ValueError(key+'必须为正整数')
    if type(config.get('retrieval',{}).get('batch_top_k',8)) is not int or config.get('retrieval',{}).get('batch_top_k',8)<1:
        raise ValueError('batch_top_k必须为正整数')
    if 'extraction_contract' in config:
        raise ValueError('联合提取配置extraction_contract已停用；请使用原独立Skill配置，不得续写联合提取结果')
    if 'history_page_bytes' in config and (type(config['history_page_bytes']) is not int or config['history_page_bytes'] < 1024):
        raise ValueError('history_page_bytes必须为至少1024的整数')
    if flow not in (FLOW, 'generic-flow-v3-relations', 'fact-driven-v1'):
        raise ValueError(f"旧版或未知flow_version={flow!r}已停止执行；请使用memory_agent_generic_flow.json和独立输出目录")
    obsolete = {"review_strategy", "max_agent_steps", "max_audit_rounds", "history_limit", "network_retries", "model_output_attempts"} & config.keys()
    if obsolete:
        raise ValueError("新流程不接受旧控制参数：" + ",".join(sorted(obsolete)))
    attempts = config.get("stage_max_attempts", 3)
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError("stage_max_attempts必须为1至3的整数")
    if flow in ('generic-flow-v3-relations', 'fact-driven-v1'):
        retrieval = config.get('retrieval', {})
        if not isinstance(retrieval, dict) or not retrieval.get('service_config'):
            raise ValueError('新关系流程需要本地retrieval.service_config')
        for key, default in (('top_k',20), ('port',8766), ('timeout_seconds',120)):
            if type(retrieval.get(key,default)) is not int or retrieval.get(key,default) < 1:
                raise ValueError('非法召回配置：'+key)
        for key, default in (('relation_batch_size',16),('relation_batch_bytes',24000)):
            if type(config.get(key,default)) is not int or config.get(key,default)<1:
                raise ValueError('非法关系批次配置：'+key)
    resolved = flow
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
    scheduling={"timeout_seconds","stage_max_attempts","history_context_tokens","rule_page_tokens",
                "source_review_batch_size","context_window_tokens","context_safety_tokens","parallel_users","rules_file","history_page_bytes"}
    return {k:v for k,v in config.items() if k not in scheduling}


def reference_resolver(session: dict):
    """同一证据包只建一次索引，定位严格限制于实际提供的话轮。"""
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
    return resolve


def materialize_references(refs: list, session: dict) -> list[dict]:
    if not isinstance(refs, list):
        raise ValueError("证据引用必须为数组")
    resolve = reference_resolver(session)
    return [dict(zip(("evidence_id", "quote"), resolve(ref, True))) for ref in refs]


def materialize(raw: dict, session: dict) -> dict:
    """将ID/角色/字符位置转换为原文；绝不模糊修正引文。"""
    c = copy.deepcopy(raw)
    if c.get("valid_from") is not None or c.get("valid_to") is not None:
        raise ValueError("模型不得自由生成有效日期；使用time_reference定位原话")
    resolve = reference_resolver(session)
    c["evidence"] = materialize_references(c["evidence"], session)
    if "fact" in c:
        fact = c["fact"]
        if "time_text" in fact or "time_value" in fact:
            raise ValueError("模型时间必须使用time_reference，不得生成time_text/time_value")
        time_ref = fact.pop("time_reference")
        fact.update(time_text="", time_value=None)
        if time_ref is not None:
            if session.get("source_kind") == "photo":
                raise ValueError("图片定位标记不是事实发生时间")
            eid, quote = resolve(time_ref, True)
            if time_ref["start"] is None:
                raise EvidenceReferenceError("time_span_required",eid,"时间表达须定位确切字符区间，不能用整句话替代时间")
            fact["time_text"] = quote
            ref = dict(evidence_id=eid,quote=quote)
            if ref not in c["evidence"]:
                c["evidence"].append(ref)
    for step in c.get("structure", {}).get("steps", []):
        exact_keys(step, {"kind", "reference"})
        ref = step["reference"]
        eid, text = resolve(ref)
        kind = step["kind"]
        if kind == "context" and ref["speaker"] != "assistant":
            raise EvidenceReferenceError("step_role_mismatch", eid, "context仅用于助手叙事语境；用户内容应使用contribution/attempt/feedback，不改原文角色")
        if ref["speaker"] == "assistant" and kind not in ("hint", "feedback", "context"):
            raise EvidenceReferenceError("step_role_mismatch", eid, "助手步骤只能使用hint/feedback/context；助手猜测用context，不能改为用户证据")
        step.clear(); step.update(kind=kind, evidence_id=eid, text=text, speaker=ref["speaker"])
    return c


def compact_fact(c: dict) -> dict:
    """比较不复制原文/步骤/审核记录，不截断影响语义的事实字段。"""
    keys = ("candidate_id", "memory_id", "revision", "device_id", "subject_key", "claim", "fact",
            "scope", "valid_from", "valid_to", "last_seen", "state", "origin_session_id",
            "assertion_mode", "source_order", "attribute_key", "retention")
    result = {k: c[k] for k in keys if k in c}
    result["context"] = c.get("context", c.get("structure", {}).get("context", "real"))
    result["evidence_ids"] = sorted({e["evidence_id"] for e in c.get("evidence", [])} | set(c.get("evidence_ids", [])))
    return result


def content_hash(c: dict) -> str:
    """内容指纹不包含产品标签和可重算的事实标识；派生标识另行校验。"""
    return digest({k: v for k, v in c.items() if k not in
                   ("dimension_id", "dimension_ids", "classification", "candidate_id", "source_skill_id", "source_skill_ids", "source_windows", "content_key")})


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
