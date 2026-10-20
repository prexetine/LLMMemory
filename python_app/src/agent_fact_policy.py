"""事实粒度、事件顺序及历史写入保护；不以维度编号证明互斥。"""
from __future__ import annotations

import re
from agent_data import digest, exact_keys
from agent_source_consistency import same_source, base_preference_support

REVISION = "2026-10-08-source-isolation-1"
FACT_FIELDS = {"type", "predicate", "object", "value", "polarity", "cardinality", "time_text", "time_value"}


def compatible_scope(a: dict, b: dict) -> bool:
    if scope_key(a) == scope_key(b):
        return True
    generic = {"all_time", "session:" + a.get("origin_session_id", "")}
    return (same_source(a, b) and a.get("fact", {}).get("type") == b.get("fact", {}).get("type") == "attribute"
            and scope_key(a) in generic and scope_key(b) in generic
            and not any(x.get(k) for x in (a, b) for k in ("valid_from", "valid_to")))


def scope_key(c: dict) -> str:
    """规范提取窗口标记，不抹掉真实时间、地点或活动限定。"""
    scope = c.get("scope", "").strip()
    if scope in {"当前会话", "本次会话", "当前会话可见对话段", "本次会话窗口内",
                 "当前会话 bounded_window_of_session", "bounded_window_of_session", "session"}:
        return "session:" + c.get("origin_session_id", "")
    return scope


def canonical_fact(fact: dict) -> dict:
    """只规范可证明等价的简单用语；不拆比较、共同活动或否定。"""
    f = dict(fact)
    for key in ("predicate", "object", "value"):
        f[key] = f[key].strip()
    if f["type"] == "request":
        aliases = {"想要", "想要获得", "想要获取", "请求获取", "索要", "想要获得物品"}
        if f["predicate"] in aliases:
            f["predicate"] = "请求获得"
            if f["object"] and f["value"] in {f["object"], *(p + f["object"] for p in ("要", "想要", "想要获得", "想要获取"))}:
                f["value"] = f["object"]
    return f


def comparable(a: dict, b: dict) -> bool:
    """扩大同义召回，但始终隔离用户、主体和临时事件的Session。"""
    if a.get('flow_version') == 'generic-flow-v3-relations':
        from agent_retrieval import compatible
        return compatible(a,b)
    if not a.get("subject_key") or a.get("subject_key") != b.get("subject_key") or a.get("device_id") != b.get("device_id"):
        return False
    if (a.get("structure") or {}).get("context", "real") != (b.get("structure") or {}).get("context", "real"):
        return False
    shared = {e["evidence_id"] for e in a.get("evidence", [])} & {e["evidence_id"] for e in b.get("evidence", [])}
    if a.get("assertion_mode") != b.get("assertion_mode"):
        # 同源不同分类必须比较，但不意味着允许自动合并或升级为长期偏好。
        return bool(shared) and a.get("origin_session_id") == b.get("origin_session_id")
    fa, fb = a.get("fact", {}), b.get("fact", {})
    if a.get("assertion_mode") in ("request", "temporary") or fa.get("type") in ("event", "request", "observation") or fb.get("type") in ("event", "request", "observation"):
        if a.get("origin_session_id") != b.get("origin_session_id"):
            return False
    shared = {e["evidence_id"] for e in a.get("evidence", [])} & {e["evidence_id"] for e in b.get("evidence", [])}
    return bool(shared or a.get("dimension_id") == b.get("dimension_id") or a.get("attribute_key") == b.get("attribute_key")
                or fa.get("object") and fa.get("object") == fb.get("object"))


def retention(c: dict) -> tuple[str, str]:
    """保存去向独立于来源可信度；兼容旧候选，保留上下文确认的具体事实。"""
    if c.get("retention") == "discard":
        return "discard", c.get("retention_reason", "没有可独立理解的新增事实")
    claim = c.get("claim", "")
    if re.search(r"(?:使用.{0,12}(?:回应|应答)|进行了?(?:回应|应答)|说出英文语句|说出英文片段|进行问候|打招呼)", claim) and not re.search(r"喜欢|计划|约定|名字|称呼|生日", claim):
        return "discard", "只描述应答动作或不可解释的语言碎片，没有具体用户事实"
    kind = c.get("fact", {}).get("type")
    predicate = c.get('fact', {}).get('predicate', '').strip()
    if c.get("structure", {}).get("type") in ("learning_episode", "shared_work"):
        return "event", "完整互动经历，不将一次过程升级为长期特征"
    if (c.get("assertion_mode") == "request" or kind == "request"
            or c.get('structure', {}).get('type') == 'request'
            or predicate in {'有需求', '当次需求', '索要', '请求获得', '请求获取'}):
        return "session_record", "当次需求保留会话记录，不自动升级为长期偏好"
    if kind == "event" or c.get("assertion_mode") == "temporary" or c.get('structure', {}).get('type') == 'real_event':
        return "event", "具体经历或状态变化作为事件保留"
    if c.get("assertion_mode") == "observation":
        return "event", "仅记录该次画面观察，不推断用户特征"
    if c.get('retention', 'memory') == 'memory' and (scope_key(c).startswith('session:') or c.get('scope') in {'本次活动', '当次活动'}):
        return 'event', '仅当前会话或活动范围成立的事实，不提升为长期属性'
    return c.get("retention", "memory"), c.get("retention_reason", "具有具体内容的用户事实")


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    """来源审核前合并精确事实，保留全部引文、维度及来源窗口。"""
    result = []
    for c in candidates:
        old = next((x for x in result if same_fact(c, x) and retention(c)[0] == retention(x)[0]
                    and all(c.get(k) == x.get(k) for k in ("kind", "subject_role", "assertion_mode", "valid_from", "valid_to"))), None)
        if old is None:
            result.append(dict(c))
            continue
        for key, fallback in (("dimension_ids", "dimension_id"), ("source_skill_ids", "source_skill_id")):
            old[key] = sorted(set(old.get(key, [old[fallback]])) | set(c.get(key, [c[fallback]])))
        old["evidence"] = list({(e["evidence_id"], e["quote"]): e for e in old["evidence"] + c["evidence"]}.values())
        old["source_windows"] = sorted(set(old["source_windows"] + c["source_windows"]))
        old["keywords"] = sorted(set(old["keywords"] + c["keywords"]))
        if old.get("structure") and c.get("structure"):
            old["structure"] = dict(old["structure"], steps=list({digest(s): s for s in old["structure"]["steps"] + c["structure"]["steps"]}.values()))
        old["merged_candidate_ids"] = sorted(set(old.get("merged_candidate_ids", [old["candidate_id"]])) | {c["candidate_id"]})
    return result


def normalize_fact(candidate: dict, session: dict) -> dict:
    """旧候选可读；新事实字段必须完整且拒绝动态键。时间只采用明确日期原话。"""
    c = candidate
    fact = c.get("fact")
    if fact is None:
        return c
    exact_keys(fact, FACT_FIELDS)
    if any(not isinstance(fact[k], str) for k in FACT_FIELDS - {"time_value"}):
        raise ValueError("fact字符串字段非法")
    if fact["type"] not in ("attribute", "event", "request", "observation") or fact["polarity"] not in ("positive", "negative") or fact["cardinality"] not in ("single", "multiple"):
        raise ValueError("fact类型/极性/基数非法")
    if not fact["predicate"].strip() or not fact["value"].strip():
        raise ValueError("fact属性和事实值不能为空")
    if fact["time_value"] is not None and not isinstance(fact["time_value"], str):
        raise ValueError("fact时间必须字符串或null")
    fact = canonical_fact(fact)
    quotes = " ".join(e["quote"] for e in c["evidence"])
    if fact["time_text"] and fact["time_text"] not in quotes:
        raise ValueError("事件时间描述不是用户原话")
    # 不用日志时间补造日期。相对日期原样保留，绝对日期由程序解析而非信任生成值。
    date = re.search(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})(?:日|号)?", fact["time_text"])
    if date:
        from datetime import date as calendar_date
        fact["time_value"] = calendar_date(*map(int, date.groups())).isoformat()
    else:
        fact["time_value"] = None
    detail_type = c.get("structure", {}).get("type")
    if detail_type in ("learning_episode", "shared_work", "real_event"):
        fact["type"] = "event"
    elif detail_type == "request":
        fact["type"] = "request"
    elif c.get("assertion_mode") == "temporary":
        fact["type"] = "event"
    elif c.get("assertion_mode") == "request":
        fact["type"] = "request"
    elif c.get("assertion_mode") == "observation" and session.get("source_kind") == "photo":
        fact["type"] = "observation"
    c["fact"] = fact
    c["scope"] = scope_key(dict(c, origin_session_id=session["session_id"]))
    c["attribute_key"] = "fact:" + digest([fact["predicate"], fact["object"]])[:20]
    positions = {t["evidence_id"]: i for i, t in enumerate(session["turns"])}
    c["source_order"] = min(positions[e["evidence_id"]] for e in c["evidence"])
    if fact["type"] in ("event", "request"):
        c["scope"] = "session:" + session["session_id"]
        c["valid_from"], c["valid_to"] = fact["time_value"], None
        c["event"] = {"event_id": digest([session["device_id"], session["session_id"], c["subject_key"], fact["predicate"], fact["object"], fact["time_value"], sorted(e["evidence_id"] for e in c["evidence"])])[:24],
                      "actor": c["subject"], "time": fact["time_value"], "time_text": fact["time_text"],
                      "description": c["claim"], "order": c["source_order"]}
    return c


def same_fact(a: dict, b: dict) -> bool:
    fa, fb = a.get("fact"), b.get("fact")
    if not fa or not fb or not a.get("subject_key") or a.get("subject_key") != b.get("subject_key"):
        return False
    if a.get("device_id") != b.get("device_id") or (a.get("structure") or {}).get("context", "real") != (b.get("structure") or {}).get("context", "real"):
        return False
    fa, fb = canonical_fact(fa), canonical_fact(fb)
    if any(fa[k] != fb[k] for k in ("type", "predicate", "object", "value", "polarity", "time_value")):
        return False
    if fa["type"] in ("event", "request", "observation"):
        # 同一事件内独立说法可以合并；不同会话/观察不能凭同字面吞并。
        shared = {e["evidence_id"] for e in a.get("evidence", [])} & {e["evidence_id"] for e in b.get("evidence", [])}
        return bool(shared) and a.get("origin_session_id") == b.get("origin_session_id") and scope_key(a) == scope_key(b)
    return scope_key(a) == scope_key(b) and a.get("assertion_mode") == b.get("assertion_mode")


def relation_guard(candidate: dict, decision: dict, history: list[dict]) -> dict:
    """危险动作有硬门禁；信息不足待处理，不凭同维度建立冲突或覆盖。"""
    d = dict(decision)
    if candidate.get('flow_version') == 'generic-flow-v3-relations':
        targets={m['memory_id']:m for m in history}
        if d['action'] not in ('pending','new'):
            if not d.get('semantic_verified') or any(t['memory_id'] not in targets or not comparable(candidate,targets[t['memory_id']]) for t in d['targets']):
                return dict(d,action='pending',targets=[],basis=[],reason='缺少已校验关系或目标主体范围不匹配')
            if d['action']=='reinforce' and any(candidate.get('fact',{}).get('polarity')!=targets[t['memory_id']].get('fact',{}).get('polarity') for t in d['targets']):
                return dict(d,action='pending',targets=[],basis=[],reason='肯否相反不能强化')
        return d
    indexed = {m["memory_id"]: m for m in history}
    shared_mismatch = [m for m in history if comparable(candidate, m)
                       and candidate.get("assertion_mode") != m.get("assertion_mode")
                       and candidate.get("fact", {}).get("object") == m.get("fact", {}).get("object")]
    if shared_mismatch and d["action"] in ("new", "coexist", "reinforce", "supersede", "conflict"):
        return dict(d, action="pending", targets=[], basis=[], reason="同源同对象的表达类型不一致，需要整理为需求或事实，不能重复新增或直接合并")
    exact = [m for m in history if same_fact(candidate, m) and comparable(candidate, m)]
    if exact and d["action"] in ("new", "coexist"):
        if len(exact) > 1 and any(m.get("state") != "reviewed" for m in exact):
            return dict(d, action="pending", targets=[], basis=[], reason="存在多个相同历史且涉及冲突，不能自动归并或重复新增")
        d = dict(d, action="reinforce", targets=[{"memory_id": m["memory_id"], "revision": m["revision"]} for m in exact],
                 semantic_verified=True, basis=[], reason="程序检测到完全相同事实，强化已有记录，不重复新增")
    if d["action"] not in ("conflict", "supersede", "reinforce"):
        return d
    targets = [indexed[t["memory_id"]] for t in d["targets"] if t["memory_id"] in indexed]
    if d.get("support_kind") == "base_only":
        if d["action"] == "reinforce" and len(targets) == 1 and comparable(candidate, targets[0]) and compatible_scope(candidate, targets[0]) and base_preference_support(candidate, targets[0]):
            return d
        return dict(d, action="pending", targets=[], basis=[], reason="基础喜好支持的主体、范围或程度方向不成立")
    # 已通过严格协议和来源校验的同义结论，不再被自由措辞差异改写为并存。
    if d["action"] == "reinforce" and d.get("semantic_verified") and targets:
        valid = all(comparable(candidate, m) and compatible_scope(candidate, m)
                    and candidate.get("fact", {}).get("type") == m.get("fact", {}).get("type")
                    and candidate.get("fact", {}).get("polarity") == m.get("fact", {}).get("polarity")
                    and candidate.get("valid_from") == m.get("valid_from")
                    and candidate.get("valid_to") == m.get("valid_to") for m in targets)
        if valid and (len(targets) == 1 or all(m.get("state") == "reviewed" for m in targets)):
            return d
        return dict(d, action="pending", targets=[], basis=[], reason="同义目标的主体、范围、时间或极性不一致，或涉及未解冲突，不能归并")
    if d["action"] == "conflict" and d.get("semantic_verified") and any(
            candidate.get("fact", {}).get("object") == m.get("fact", {}).get("object")
            and candidate.get("fact", {}).get("predicate") != m.get("fact", {}).get("predicate") for m in targets):
        return dict(d, action="pending", targets=[], basis=[], reason="疑似矛盾的属性措辞未归一，不能仅凭谓词字面不同判为并存")
    a = candidate.get("fact")
    if not a or any(not m.get("fact") for m in targets):
        keys = ("subject_key", "dimension_id", "claim", "scope", "kind", "valid_from", "valid_to")
        if d["action"] == "reinforce" and len(targets) == 1 and all(candidate.get(k) == targets[0].get(k) for k in keys):
            return d
        return dict(d, action="pending", targets=[], basis=[], reason="缺少具体事实属性，不能仅按维度强化、覆盖或判冲突")
    relations = []
    for m in targets:
        b = m["fact"]
        if same_fact(candidate, m):
            relations.append("same")
        elif (a["predicate"], a["object"]) != (b["predicate"], b["object"]):
            relations.append("coexist")
        elif a["type"] in ("event", "request", "observation") or b["type"] in ("event", "request", "observation"):
            relations.append("event")
        elif scope_key(candidate) != scope_key(m) or a["time_value"] != b["time_value"]:
            relations.append("unknown")
        elif a["value"] == b["value"] and a["polarity"] != b["polarity"]:
            relations.append("opposite")
        elif a["predicate"] in ("出生日期", "出生年月日") and a["cardinality"] == b["cardinality"] == "single" and a["value"] != b["value"]:
            relations.append("opposite")
        elif a["cardinality"] == b["cardinality"] == "multiple":
            relations.append("coexist")
        else:
            # 名称、地域层级等不能只因值不同而证明互斥。
            relations.append("unknown")
    if d["action"] == "reinforce" and relations == ["same"]:
        return d
    if d["action"] == "conflict" and relations and all(x == "opposite" for x in relations):
        return d
    if d["action"] == "supersede" and relations and all(x in ("same", "opposite", "unknown", "event") for x in relations) and d.get("basis"):
        return d  # 事务层另核对明确更正所指对象及来源。
    if relations and all(x in ("coexist", "event") for x in relations):
        return dict(d, action="coexist", basis=[], reason="不同事实对象、多值共存或事件顺序记录，不以同维度/同会话认定互斥")
    return dict(d, action="pending", targets=[], basis=[], reason="维护关系缺少足够互斥/同义依据，不修改历史；原判断：" + d["reason"])


def latest_candidates(candidates: list[dict]) -> list[dict]:
    """业务展示仅保留叶子版本；没有成功产生子版本的父候选仍保留。"""
    parents = {c.get("parent_candidate_id") for c in candidates}
    return [c for c in candidates if c.get("candidate_id") not in parents]
