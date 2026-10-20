"""v4严格模型协议，复用基础结构但不复用v3审核结论。"""
from agent_schema import obj, arr
from agent_v3_schema import schema as old_schema
from agent_contracts import CHECKS, FORBIDDEN
from agent_memory_structure import TYPES, CONTEXTS, STEP_KINDS


def schema(role, data):
    s = {"type": "string"}
    repair = obj({"kind": {"type": "string", "enum": ["none", "split", "reclassify", "narrow"]}, "reason": s})
    if role == "source_repair":
        candidate = schema("source_extractor:repair", data)["properties"]["candidates"]["items"]
        return obj({"results": arr(obj({"parent_candidate_id": s, "candidates": arr(candidate), "reason": s}))})
    result = old_schema(role, data)
    if role == "source_router":
        signal = result["properties"]["signals"]["items"]
        signal["properties"].update(target_type={"type": "string", "enum": ["item", "session", "all", "unresolved"]}, target_text=s)
        signal["required"] = list(signal["properties"])
    elif role.startswith("source_extractor:"):
        candidate = result["properties"]["candidates"]["items"]
        p = candidate["properties"]
        p["dimension_id"]["enum"] = [d for d in p["dimension_id"]["enum"] if d not in FORBIDDEN]
        p["dimension_id"]["enum"].append("UNCLASSIFIED")
        p.update(subject_role={"type": "string", "enum": ["current_user", "other", "unresolved"]},
                 assertion_mode={"type": "string", "enum": ["self_report", "request", "temporary", "observation"]})
        p.update(retention={"type": "string", "enum": ["memory", "event", "session_record", "discard"]}, retention_reason=s)
        p["structure"] = obj({"type": {"type": "string", "enum": list(TYPES)},
                              "context": {"type": "string", "enum": list(CONTEXTS)}, "topic": s,
                              "steps": arr(obj({"kind": {"type": "string", "enum": list(STEP_KINDS)},
                                                "text": s, "evidence_id": s,
                                                "speaker": {"type": "string", "enum": ["user", "assistant"]}}))})
        p["fact"] = obj({"type": {"type": "string", "enum": ["attribute", "event", "request", "observation"]},
                         "predicate": s, "object": s, "value": s,
                         "polarity": {"type": "string", "enum": ["positive", "negative"]},
                         "cardinality": {"type": "string", "enum": ["single", "multiple"]},
                         "time_text": s, "time_value": {"type": ["string", "null"]}})
        candidate["required"] = list(p)
    elif role == "history_compare":
        row = result["properties"]["results"]["items"]
        row["properties"]["related_memory_ids"] = arr(s)
        row["required"] = list(row["properties"])
    elif role == "source_reviewer":
        row = result["properties"]["results"]["items"]
        row["properties"]["checks"] = obj({k: {"type": "string", "enum": ["pass", "fail", "uncertain"]} for k in CHECKS})
        row["properties"]["repair"] = repair
        row["required"] = list(row["properties"])
    return result
