"""v3 模型协议。所有对象拒绝动态键；运行时另校验语义和引用。"""
from agent_schema import obj, arr
from agent_data import SKILL_NAMES
from agent_v3_contracts import CHECKS


def schema(role, data):
    s = {"type": "string"}
    nullable = {"type": ["string", "null"]}
    ev = obj({"evidence_id": s, "quote": s})
    if role == "source_router":
        signal = obj({"kind": {"type": "string", "enum": ["correction", "change", "denial", "withdrawal", "subject_switch"]},
                      "subject_key": nullable, "attribute_key": nullable, "scope": s, "evidence": arr(ev)})
        return obj({"skills": arr({"type": "string", "enum": list(SKILL_NAMES)}), "signals": arr(signal), "reason": s})
    if role.startswith("source_extractor:"):
        candidate = obj({"dimension_id": {"type": "string", "enum": [d["dimension_id"] for d in data["skill_rules"]]},
                         "subject": s, "claim": s, "scope": s,
                         "kind": {"type": "string", "enum": ["self_report", "observation", "hypothesis"]},
                         "keywords": arr(s), "evidence": arr(ev),
                         "operation": {"type": "string", "enum": ["new"]}, "target_memory_id": s,
                         "subject_key": nullable, "attribute_key": nullable, "valid_from": nullable, "valid_to": nullable})
        return obj({"candidates": arr(candidate), "notes": s})
    if role == "source_reviewer":
        return obj({"results": arr(obj({"candidate_id": s,
            "decision": {"type": "string", "enum": ["pass", "pending", "reject"]},
            "checks": obj({k: {"type": "string", "enum": ["pass", "fail", "uncertain"]} for k in CHECKS}),
            "reason": s, "evidence_ids": arr(s), "needs_context": {"type": "boolean"}, "context_ids": arr(s)}))})
    if role == "history_compare":
        return obj({"results": arr(obj({"candidate_id": s,
            "action": {"type": "string", "enum": ["new", "reinforce", "coexist", "supersede", "conflict", "pending"]},
            "targets": arr(obj({"memory_id": s, "revision": {"type": "integer"}})), "reason": s, "basis": arr(ev)}))})
    return None
