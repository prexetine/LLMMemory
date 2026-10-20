"""v3 权威状态、纯函数维护和 Session 原子提交。无外部向量服务。"""
from __future__ import annotations

import copy
import json
from datetime import datetime

from agent_data import digest
from agent_v3_contracts import VERSION, verify_evidence, verify_seal, verify_source, validate_signal, confirmed_subject


class StaleHistory(RuntimeError):
    """来源凭证仍有效，仅历史计划需要重算。"""


def empty_state():
    return {"memories": {}, "versions": [], "groups": {}}


def relevant(state, c):
    # 同维度兜底，不因模型属性标签不一致就漏掉矛盾；不设置 top-N。
    return [copy.deepcopy(m) for m in state["memories"].values()
            if m["state"] in ("reviewed", "conflicted") and m.get("subject_key") == c.get("subject_key")
            and (m["dimension_id"] == c["dimension_id"] or m.get("attribute_key") == c.get("attribute_key"))]


def archive(state, m):
    state["versions"].append(copy.deepcopy(m))


def touch(state, m):
    archive(state, m)
    old_revision = m["revision"]
    m["revision"] += 1
    for g in state["groups"].values():
        if g["status"] == "open":
            for member in g["members"]:
                if member["memory_id"] == m["memory_id"] and member["revision"] == old_revision:
                    g.setdefault("member_history", []).append(copy.deepcopy(member))
                    member["revision"] = m["revision"]


def supports(m, c, session):
    known = set(m.get("support_evidence_ids", []))
    incoming = {e["evidence_id"] for e in c["evidence"]}
    sessions = set(m.get("support_session_ids", []))
    if incoming - known:
        sessions.add(session["session_id"])
    m["support_evidence_ids"] = sorted(known | incoming)
    m["support_session_ids"] = sorted(sessions)
    m["support_count"] = len(sessions)
    m["evidence"] = list({(e["evidence_id"], e["quote"]): e for e in m.get("evidence", []) + c["evidence"]}.values())
    m["keywords"] = sorted(set(m["keywords"]) | set(c["keywords"]))
    m["last_seen"] = max(m["last_seen"], session["end"])


def applies(signal, c):
    return ((not signal["subject_key"] or signal["subject_key"] == c.get("subject_key"))
            and (not signal["attribute_key"] or signal["attribute_key"] == c.get("attribute_key")))


def definitely_later(new_time, old_time):
    if not new_time or not old_time:
        return False
    try:
        return datetime.fromisoformat(new_time) > datetime.fromisoformat(old_time)
    except (TypeError, ValueError):
        return False


def apply_decision(state, c, decision, proof, session, signals):
    """同一函数用于预演与事务重放；每项变更都有来源证明和维护决定。"""
    verify_source(proof, c, session)
    if (proof["candidate_hash"] != digest(c) or proof["input_hash"] != session["input_hash"]
            or proof["device_id"] != session["device_id"] or proof["session_id"] != session["session_id"]):
        raise ValueError("来源凭证不属于当前候选版本/会话")
    action = decision["action"]
    if action == "pending":
        return None
    if proof["decision"] != "pass" or not confirmed_subject(c.get("subject_key"), session["device_id"]) or not c.get("attribute_key"):
        raise ValueError("缺少来源通过凭证或可确认主体/属性")
    verify_evidence(c["evidence"], session)
    if any(s["kind"] == "withdrawal" and applies(s, c) for s in signals):
        raise ValueError("禁止保存线索阻止有效入库")
    targets = []
    for t in decision["targets"]:
        m = state["memories"].get(t["memory_id"])
        if not m or m["revision"] != t["revision"] or m["state"] not in ("reviewed", "conflicted"):
            raise StaleHistory("历史目标版本/状态变化")
        if m["device_id"] != session["device_id"] or m.get("subject_key") != c["subject_key"]:
            raise ValueError("禁止跨设备/未确认主体合并")
        if m["dimension_id"] != c["dimension_id"] and m.get("attribute_key") != c["attribute_key"]:
            raise ValueError("历史属性不可比较")
        targets.append(m)
    if action not in ("new", "reinforce", "coexist", "supersede", "conflict"):
        raise ValueError("非法维护动作")
    if action != "new" and not targets or action == "new" and targets:
        raise ValueError("维护目标数量不正确")
    if action == "supersede":
        verify_evidence(decision["basis"], session)
        basis = {(e["evidence_id"], e["quote"]) for e in decision["basis"]}
        explicit = {(e["evidence_id"], e["quote"]) for s in signals
                    if s["kind"] in ("correction", "change") and applies(s, c) for e in s["evidence"]}
        if not basis & explicit:
            raise ValueError("缺少已识别明确更正/状态变化线索")
        # 迟到日志不是新状态；只允许明确可比较的有效时间推翻该保护。
        for m in targets:
            if session["end"] < m["last_seen"] and not definitely_later(c.get("valid_from"), m.get("valid_from")):
                raise ValueError("迟到日志不能覆盖较新状态")
    mid = targets[0]["memory_id"] if action == "reinforce" else digest([session["device_id"], session["session_id"], c["candidate_id"]])[:24]
    if action == "reinforce":
        if len(targets) != 1:
            raise ValueError("强化目标必须唯一")
        m = targets[0]
        touch(state, m)
    else:
        if mid in state["memories"]:
            raise ValueError("重复创建未经过幂等批次门禁")
        m = dict(copy.deepcopy(c), memory_id=mid, device_id=session["device_id"], revision=1,
                 state="reviewed", created_at=session["end"], last_seen=session["end"], evidence=[],
                 source_proof=proof["proof_hash"], supersedes=[])
        state["memories"][mid] = m
    supports(m, c, session)
    if action == "supersede":
        for old in targets:
            m["supersedes"].append({"memory_id": old["memory_id"], "revision": old["revision"]})
            touch(state, old)
            old["state"] = "superseded"
        target_ids = {x["memory_id"] for x in targets}
        for g in state["groups"].values():
            if g["status"] != "open" or not target_ids.intersection(x["memory_id"] for x in g["members"]):
                continue
            # 没有明确覆盖全部矛盾成员，不设置获胜方，保留未解组。
            if all(x["memory_id"] in target_ids for x in g["members"]):
                g.update(status="resolved", resolution={"evidence": decision["basis"],
                         "replacement": {"memory_id": mid, "revision": m["revision"]}, "at": session["end"]})
            else:
                m["state"] = "conflicted"
                g["members"].append({"memory_id": mid, "revision": m["revision"]})
                g["reason"] += "；部分成员更正，其他矛盾尚未解决"
    if action == "conflict":
        ids = {mid} | {x["memory_id"] for x in targets}
        groups = [g for g in state["groups"].values() if g["status"] == "open"
                  and ids.intersection(x["memory_id"] for x in g["members"])]
        for g in groups:
            ids.update(x["memory_id"] for x in g["members"])
        gid = groups[0]["conflict_group_id"] if groups else digest([session["device_id"], sorted(ids)])[:24]
        prior_members = [copy.deepcopy(g["members"]) for g in groups]
        for other in groups[1:]:
            other.update(status="resolved", resolution={"merged_into": gid, "at": session["end"], "kind": "group_merge_not_fact_resolution"})
        for member_id in ids:
            member = state["memories"][member_id]
            if member["state"] == "reviewed":
                if member_id != mid:
                    touch(state, member)
                member["state"] = "conflicted"
        state["groups"][gid] = {"conflict_group_id": gid, "device_id": session["device_id"],
            "subject_key": c["subject_key"], "attribute_key": c["attribute_key"], "scope": c["scope"],
            "members": [{"memory_id": i, "revision": state["memories"][i]["revision"]} for i in sorted(ids)],
            "prior_members": prior_members, "reason": decision["reason"],
            "evidence_ids": sorted({e["evidence_id"] for i in ids for e in state["memories"][i]["evidence"]}),
            "status": "open", "resolution": None}
    return {"candidate_id": c["candidate_id"], "memory_id": mid, "revision": m["revision"], "operation": action}


def suspend(state, signals):
    for m in state["memories"].values():
        if m["state"] in ("reviewed", "conflicted") and any(s["kind"] == "withdrawal" and applies(s, m) for s in signals):
            touch(state, m)
            m["state"] = "pending_control"


class V3Store:
    def __init__(self, base):
        self.base, self.db = base, base.db
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS v3_state(device_id TEXT PRIMARY KEY, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS v3_receipts(batch_id TEXT PRIMARY KEY, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS v3_source_reviews(proof_hash TEXT PRIMARY KEY, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS v3_plans(batch_id TEXT PRIMARY KEY, data TEXT NOT NULL);
        """)
        self.db.commit()
        if self.db.execute("SELECT 1 FROM sessions LIMIT 1").fetchone() and not self.db.execute("SELECT 1 FROM v3_receipts LIMIT 1").fetchone():
            raise ValueError("旧数据库不允许自动迁移，请使用独立 v3 输出目录")

    def snapshot(self, device):
        r = self.db.execute("SELECT data FROM v3_state WHERE device_id=?", (device,)).fetchone()
        return json.loads(r[0]) if r else empty_state()

    def receipt(self, session):
        bid = digest([VERSION, session["device_id"], session["session_id"], session["input_hash"]])
        r = self.db.execute("SELECT data FROM v3_receipts WHERE batch_id=?", (bid,)).fetchone()
        return json.loads(r[0]) if r else None

    def commit(self, session, candidates, proofs, plan, signals, coverage):
        verify_seal(plan)
        for signal in signals:
            validate_signal(signal, session)
        if plan["policy"] != VERSION or plan["input_hash"] != session["input_hash"] or plan["signals_hash"] != digest(signals):
            raise ValueError("维护计划不匹配")
        existing = self.receipt(session)
        if existing:
            return existing
        known = {c["candidate_id"]: c for c in candidates}
        source = {p["candidate_id"]: p for p in proofs}
        if set(source) != set(known) or len(proofs) != len(known):
            raise ValueError("逐条来源凭证不完整")
        if sorted(d["candidate_id"] for d in plan["decisions"]) != sorted(known):
            raise ValueError("逐条维护决定不完整")
        bid = digest([VERSION, session["device_id"], session["session_id"], session["input_hash"]])
        try:
            self.db.execute("BEGIN IMMEDIATE")
            state = self.snapshot(session["device_id"])
            if digest(state) != plan["base_hash"]:
                raise StaleHistory("历史快照已变化，仅重新比对")
            if self.db.execute("SELECT 1 FROM sessions WHERE device_id=? AND session_id=?", (session["device_id"], session["session_id"])).fetchone():
                raise ValueError("同Session输入修订需独立运行目录")
            mapping = []
            for d in plan["decisions"]:
                c, p = known[d["candidate_id"]], source[d["candidate_id"]]
                if d["source_proof"] != p["proof_hash"]:
                    raise ValueError("来源凭证被替换")
                if d["history_hash"] != digest(relevant(state, c)):
                    raise StaleHistory("相关历史未完整比对或版本变化")
                item = apply_decision(state, c, d, p, session, signals)
                if item:
                    mapping.append(item)
            suspend(state, signals)
            if digest(state) != plan["result_hash"]:
                raise ValueError("预演与实际维护结果不同")
            receipt = {"status": "committed", "batch_id": bid, "mapping": mapping,
                       "counts": {a: sum(d["action"] == a for d in plan["decisions"]) for a in ("new", "reinforce", "coexist", "supersede", "conflict", "pending")},
                       "vector": "disabled", "policy": VERSION}
            dump = lambda x: json.dumps(x, ensure_ascii=False)
            self.db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?)", (bid, session["device_id"], session["session_id"], session["input_hash"], session["start"], session["end"], "completed", dump(coverage)))
            for t in session["turns"]:
                prior = self.db.execute("SELECT data FROM evidence WHERE evidence_id=?", (t["evidence_id"],)).fetchone()
                if prior and json.loads(prior[0])["user_text"] != t["user_text"]:
                    raise ValueError("同一证据ID原文变化")
                self.db.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?,?)", (t["evidence_id"], session["device_id"], session["session_id"], dump(t)))
            for p in proofs:
                verify_seal(p)
                self.db.execute("INSERT OR IGNORE INTO v3_source_reviews VALUES(?,?)", (p["proof_hash"], dump(p)))
                legacy = dict(p, review_id=p["proof_hash"], decision={"pass": "approve", "pending": "defer", "reject": "reject"}[p["decision"]], candidate=known[p["candidate_id"]], reviewed_at=session["end"])
                self.db.execute("INSERT OR IGNORE INTO reviews VALUES(?,?,?,?,?)", (p["proof_hash"], bid, p["candidate_id"], legacy["decision"], dump(legacy)))
            for m in state["memories"].values():
                self.db.execute("INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(memory_id) DO UPDATE SET state=excluded.state,last_seen=excluded.last_seen,revision=excluded.revision,data=excluded.data", (m["memory_id"], m["device_id"], m["dimension_id"], m["subject"], m["claim"], m["scope"], m["kind"], m["state"], m["created_at"], m["last_seen"], m["revision"], dump(m)))
                for e in m["evidence"]:
                    self.db.execute("INSERT OR IGNORE INTO memory_evidence VALUES(?,?,?)", (m["memory_id"], e["evidence_id"], e["quote"]))
                for word in m["keywords"]:
                    self.db.execute("INSERT OR IGNORE INTO keywords VALUES(?,?)", (word, m["memory_id"]))
                self.db.execute("INSERT OR REPLACE INTO memory_reviews VALUES(?,?,?)", (m["memory_id"], m["revision"], m["source_proof"]))
                self.db.execute("INSERT OR REPLACE INTO vector_queue VALUES(?,?,?,?)", (m["memory_id"], m["revision"], digest(m["claim"]), "disabled" if m["state"] == "reviewed" else "invalidated"))
            for item in mapping:
                self.db.execute("INSERT INTO changes VALUES(?,?,?,?,?)", (bid, item["candidate_id"], item["memory_id"], item["operation"], ""))
            self.db.execute("INSERT OR REPLACE INTO v3_state VALUES(?,?)", (session["device_id"], dump(state)))
            self.db.execute("INSERT INTO v3_plans VALUES(?,?)", (bid, dump(plan)))
            self.db.execute("INSERT INTO v3_receipts VALUES(?,?)", (bid, dump(receipt)))
            self.db.commit()
            return receipt
        except BaseException:
            self.db.rollback()
            raise
