"""同源结论一致性与有限同义比较；不按多数投票制造来源通过。"""
from __future__ import annotations

import re
from agent_data import digest


def object_key(value: str) -> str:
    """仅统一明确的聊天动作措辞；不猜人名、否定或ASR。"""
    text = value.replace("智能助手", "助手").replace("对话智能体", "助手")
    for old, new in (("对话", "聊天"), ("与", "和"), ("一起", ""), ("呆着", ""), ("待着", ""), ("进行", "")):
        text = text.replace(old, new)
    return re.sub(r"[\s，。！？、]", "", text)


def same_source(a: dict, b: dict) -> bool:
    return (bool(a.get("evidence")) and a.get("subject_key") is not None and a.get("subject_key") == b.get("subject_key")
            and a.get("device_id") == b.get("device_id") and a.get("origin_session_id") == b.get("origin_session_id")
            and {e["evidence_id"] for e in a.get("evidence", [])} == {e["evidence_id"] for e in b.get("evidence", [])})


def related_conclusion(a: dict, b: dict) -> bool:
    if not same_source(a, b) or a.get("dimension_id") != b.get("dimension_id"):
        return False
    fa, fb = a.get("fact", {}), b.get("fact", {})
    if any(fa.get(k) != fb.get(k) for k in ("type", "polarity", "predicate")) or any(a.get(k) != b.get(k) for k in ("valid_from", "valid_to")):
        return False
    x, y = object_key(fa.get("object", "")), object_key(fb.get("object", ""))
    if a.get("dimension_id") == "G19":
        for word in ("每天", "日常", "的"):
            x, y = x.replace(word, ""), y.replace(word, "")
    return bool(x) and x == y


def align_source_proofs(candidates: list[dict], proofs: list[dict]) -> list[dict]:
    """相同结论出现实质审核分歧时，阻止通过方绕过歧义入库。"""
    byid = {p["candidate_id"]: p for p in proofs}
    result = []
    for c in candidates:
        p = byid[c["candidate_id"]]
        if c.get("flow_version") in ("generic-flow-v1", "generic-flow-v2"):
            result.append(p)
            continue
        peers = [byid[x["candidate_id"]] for x in candidates if x["candidate_id"] != c["candidate_id"]
                 and related_conclusion(c, x) and byid[x["candidate_id"]]["decision"] != "pass"
                 and any(byid[x["candidate_id"]]["checks"].get(k) != "pass" for k in ("evidence", "semantics", "subject"))]
        if p["decision"] == "pass" and peers:
            p = dict(p, decision="pending", checks=dict(p["checks"], semantics="uncertain"),
                     reason="同源同结论存在实质审核分歧，不按通过票数入库：" + "；".join(x["reason"] for x in peers),
                     consistency_peer_ids=[x["candidate_id"] for x in peers])
            p["proof_hash"] = digest({k: v for k, v in p.items() if k != "proof_hash"})
        result.append(p)
    return result


def base_preference_support(candidate: dict, old: dict) -> bool:
    """弱表达只支持已有更强表达的基础喜好，不支持程度。"""
    a, b = candidate.get("fact", {}), old.get("fact", {})
    return (all(candidate.get(k) == old.get(k) for k in ("valid_from", "valid_to"))
            and a.get("type") == b.get("type") == "attribute" and a.get("polarity") == b.get("polarity") == "positive"
            and a.get("predicate") in ("喜欢", "喜好") and b.get("predicate") in ("喜欢", "喜好", "最喜欢")
            and a.get("value") == "喜欢" and b.get("value") == "最喜欢"
            and bool(a.get("object")) and object_key(a["object"]) == object_key(b.get("object", "")))
