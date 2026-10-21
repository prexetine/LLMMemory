"""事实驱动记忆的独立 SQLite 真源；旧 memories 表不参与写入。"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any

from agent_data import digest
from agent_fact_model import content_key, facets, map_legacy_dimensions, projection_state


class FactStore:
    """管理证据、事实、关系、投影和分析观察，所有写入可按事实幂等恢复。"""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS fact_sessions(
          run_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, session_id TEXT NOT NULL,
          input_hash TEXT NOT NULL, status TEXT NOT NULL, closed_at TEXT, data TEXT NOT NULL,
          UNIQUE(device_id, session_id, input_hash)
        );
        CREATE TABLE IF NOT EXISTS fact_evidence(
          evidence_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, session_id TEXT NOT NULL, data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fact_assertions(
          fact_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, subject_key TEXT, content_key TEXT NOT NULL,
          fact_type TEXT NOT NULL, predicate TEXT NOT NULL, object_text TEXT, value_text TEXT,
          polarity TEXT NOT NULL, scope TEXT NOT NULL, source_decision TEXT NOT NULL, state TEXT NOT NULL,
          created_at TEXT NOT NULL, last_seen TEXT NOT NULL, data TEXT NOT NULL,
          UNIQUE(device_id, content_key)
        );
        CREATE TABLE IF NOT EXISTS fact_evidence_links(
          fact_id TEXT NOT NULL REFERENCES fact_assertions(fact_id), evidence_id TEXT NOT NULL REFERENCES fact_evidence(evidence_id),
          quote TEXT NOT NULL, PRIMARY KEY(fact_id, evidence_id, quote)
        );
        CREATE TABLE IF NOT EXISTS fact_relations(
          relation_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, left_fact_id TEXT NOT NULL,
          right_fact_id TEXT NOT NULL, relation TEXT NOT NULL, state TEXT NOT NULL, reason TEXT NOT NULL,
          data TEXT NOT NULL, UNIQUE(left_fact_id, right_fact_id, relation)
        );
        CREATE TABLE IF NOT EXISTS fact_projections(
          projection_id TEXT PRIMARY KEY, fact_id TEXT NOT NULL REFERENCES fact_assertions(fact_id),
          projection_type TEXT NOT NULL, state TEXT NOT NULL, data TEXT NOT NULL,
          UNIQUE(fact_id, projection_type)
        );
        CREATE TABLE IF NOT EXISTS fact_dimension_maps(
          fact_id TEXT NOT NULL REFERENCES fact_assertions(fact_id), dimension_id TEXT NOT NULL,
          mapping_version TEXT NOT NULL, reason TEXT NOT NULL, PRIMARY KEY(fact_id, dimension_id, mapping_version)
        );
        CREATE TABLE IF NOT EXISTS fact_taxonomy_gaps(
          fact_id TEXT PRIMARY KEY REFERENCES fact_assertions(fact_id), state TEXT NOT NULL, reason TEXT NOT NULL, data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fact_candidate_audits(
          candidate_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, session_id TEXT NOT NULL,
          fact_id TEXT, source_decision TEXT NOT NULL, maintenance_action TEXT NOT NULL, data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fact_tasks(
          task_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, session_id TEXT NOT NULL, stage TEXT NOT NULL,
          state TEXT NOT NULL, reason TEXT NOT NULL, data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analysis_observations(
          observation_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, observation_type TEXT NOT NULL,
          state TEXT NOT NULL, period_start TEXT, period_end TEXT, data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fact_aliases(alias_id TEXT PRIMARY KEY, canonical_id TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS fact_support(
          fact_id TEXT NOT NULL, session_id TEXT NOT NULL, evidence_id TEXT NOT NULL,
          PRIMARY KEY(fact_id,session_id,evidence_id), UNIQUE(fact_id,evidence_id));
        CREATE TABLE IF NOT EXISTS fact_group_members(
          device_id TEXT, session_id TEXT, input_hash TEXT, candidate_id TEXT,
          PRIMARY KEY(device_id,session_id,input_hash,candidate_id));
        CREATE TABLE IF NOT EXISTS fact_changes(seq INTEGER PRIMARY KEY AUTOINCREMENT, fact_id TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS fact_insert_change AFTER INSERT ON fact_assertions
          BEGIN INSERT INTO fact_changes(fact_id) VALUES(NEW.fact_id); END;
        CREATE TRIGGER IF NOT EXISTS fact_update_change AFTER UPDATE ON fact_assertions
          BEGIN INSERT INTO fact_changes(fact_id) VALUES(NEW.fact_id); END;
        """)
        self.db.commit()

    def ensure_protocol(self):
        marker = self.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_protocol'").fetchone()
        if marker is None:
            if self.db.execute("SELECT 1 FROM fact_sessions LIMIT 1").fetchone():
                raise ValueError("旧事实结果禁止跨协议续跑，请使用独立目录")
            with self.db:
                self.db.execute("CREATE TABLE session_protocol(version TEXT NOT NULL)")
                self.db.execute("INSERT INTO session_protocol VALUES('session-reliable-v2')")
        elif self.db.execute("SELECT version FROM session_protocol").fetchone()[0] != "session-reliable-v2":
            raise ValueError("Session协议版本不一致")

    def save_receipt(self, session: dict[str, Any]) -> None:
        """模型调用前保存原始证据；失败时由数据库异常明确上报。"""
        run_id = digest(["fact-driven-v1", session["device_id"], session["session_id"], session["input_hash"]])
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO fact_sessions VALUES(?,?,?,?,?,?,?)",
                            (run_id, session["device_id"], session["session_id"], session["input_hash"],
                             "evidence_only", None, json.dumps({"evidence_count": len(session["turns"])}, ensure_ascii=False)))
            for turn in session["turns"]:
                self.db.execute("INSERT OR IGNORE INTO fact_evidence VALUES(?,?,?,?)",
                                (turn["evidence_id"], session["device_id"], session["session_id"],
                                 json.dumps(turn, ensure_ascii=False)))

    def save_candidates(self, session, candidates):
        """审核前保留候选，后续审核记录用相同ID更新。"""
        with self.db:
            for candidate in candidates:
                self.db.execute("INSERT OR IGNORE INTO fact_candidate_audits VALUES(?,?,?,?,?,?,?)",
                                (candidate["candidate_id"], session["device_id"], session["session_id"], None,
                                 "pending", "provisional", json.dumps({"candidate": candidate, "reason": "审核尚未完成"}, ensure_ascii=False)))

    def reconcile_candidates(self, session, candidates):
        """未进入最终候选集的原始提取仍可追溯，但不伪装成待审核漏项。"""
        keep = {c["candidate_id"] for c in candidates} | self.completed_candidates(session)
        with self.db:
            for row in self.db.execute("SELECT candidate_id,data FROM fact_candidate_audits WHERE device_id=? AND session_id=? AND fact_id IS NULL",
                                       (session["device_id"],session["session_id"])).fetchall():
                if row["candidate_id"] not in keep:
                    data=json.loads(row["data"])
                    data["decision"]={"action":"provisional","discard":True,"reason":"原始候选已精确去重或被修订版本替代"}
                    self.db.execute("UPDATE fact_candidate_audits SET data=? WHERE candidate_id=?",
                                    (json.dumps(data,ensure_ascii=False),row["candidate_id"]))

    def session_completed(self, session: dict[str, Any]) -> bool:
        row = self.db.execute("SELECT status FROM fact_sessions WHERE device_id=? AND session_id=? AND input_hash=?",
                              (session["device_id"], session["session_id"], session["input_hash"])).fetchone()
        if not row or row["status"] != "closed":
            return False
        pending = self.db.execute("SELECT 1 FROM fact_tasks WHERE device_id=? AND session_id=? AND state='technical_retry' LIMIT 1",
                                  (session["device_id"], session["session_id"])).fetchone()
        return pending is None

    def session_result(self, session):
        """从数据库事实与审核收据恢复报告，不调用模型。"""
        rows = self.db.execute("SELECT * FROM fact_candidate_audits WHERE device_id=? AND session_id=?",
                               (session["device_id"], session["session_id"])).fetchall()
        tasks = [json.loads(row[0]) for row in self.db.execute(
            "SELECT data FROM fact_tasks WHERE device_id=? AND session_id=? AND state='technical_retry'",
            (session["device_id"], session["session_id"]))]
        receipt = self.db.execute("SELECT status FROM fact_sessions WHERE device_id=? AND session_id=? AND input_hash=?",
            (session["device_id"],session["session_id"],session["input_hash"])).fetchone()
        if (not receipt or receipt[0] not in ("closed", "closed_with_tasks")) and not tasks:
            tasks.append(dict(task_id=digest(["incomplete",session["input_hash"]]), stage="recovery",
                              state="technical_retry", candidate_ids=[], reason="保存已完成，流程尚未收口"))
        candidates, reviews, maintenance, approved = [], [], [], []
        summary = dict(evidence_count=len(session["turns"]), candidate_count=len(rows), effective_count=0,
                       rejected_count=0, semantic_pending_count=0, technical_task_count=len(tasks))
        for row in rows:
            data = json.loads(row["data"])
            candidate = data["candidate"]
            candidates.append(candidate)
            decision = data.get("decision", {"action":"provisional", "reason":"审核未完成", "technical_pending":True})
            # 兼容旧事实收据中的字符串；恢复和导出统一使用带原因的控制对象。
            if isinstance(decision.get("storage_control"), str):
                decision = dict(decision, storage_control={"status": decision["storage_control"],
                                                           "reason": decision.get("reason", "")})
            maintenance.append(dict(decision, candidate_id=row["candidate_id"]))
            if data.get("proof"):
                reviews.append(dict(data["proof"], candidate=candidate))
            if row["source_decision"] == "reject":
                summary["rejected_count"] += 1
            elif decision.get("action") == "provisional" and not decision.get("technical_pending") and not decision.get("discard") and decision.get("storage_control", {}).get("status") != "blocked":
                summary["semantic_pending_count"] += 1
            if row["fact_id"] and decision.get("action") != "provisional":
                stored = self.db.execute("SELECT state FROM fact_assertions WHERE fact_id=?",
                                         (self.canonical(row["fact_id"]),)).fetchone()
                if stored and stored[0] in ("accepted", "conflicted"):
                    approved.append(dict(candidate, operation=decision["action"]))
        summary["effective_count"] = len(approved)
        conclusion = ("partial" if candidates else "evidence_only") if tasks else (
            "semantic_pending" if summary["semantic_pending_count"] else "complete_with_facts" if approved else "complete_empty")
        return dict(session_id=session["session_id"], turns=len(session["turns"]), status="completed",
                    candidates=candidates, reviews=reviews, maintenance=maintenance, approved=approved,
                    retry_tasks=tasks, needs_attention=bool(tasks), conclusion=conclusion, session_summary=summary,
                    commit={"status":"already_committed" if receipt and receipt[0] == "closed" else "partial"},
                    execution_status="awaiting_retry" if tasks else "completed",
                    commit_status="partial" if tasks else "committed",
                    flow_version="fact-driven-v1", batch_metrics={})

    def history(self, device_id: str, subject_key: str | None) -> list[dict[str, Any]]:
        args: list[Any] = [device_id]
        sql = "SELECT * FROM fact_assertions WHERE device_id=? AND state IN ('accepted','conflicted')"
        if subject_key:
            sql += " AND subject_key=?"; args.append(subject_key)
        rows = self.db.execute(sql + " ORDER BY last_seen,fact_id", args).fetchall()
        result = []
        for row in rows:
            item = json.loads(row["data"])
            item.update(fact_id=row["fact_id"], content_key=row["content_key"], source_decision=row["source_decision"], state=row["state"])
            item["target_version"] = digest(dict(row))
            item["evidence"] = [dict(e) for e in self.db.execute(
                "SELECT evidence_id,quote FROM fact_evidence_links WHERE fact_id=?", (row["fact_id"],))]
            result.append(item)
        return result

    def canonical(self, fact_id):
        seen = set()
        while fact_id not in seen:
            seen.add(fact_id)
            row = self.db.execute("SELECT canonical_id FROM fact_aliases WHERE alias_id=?", (fact_id,)).fetchone()
            if not row:
                return fact_id
            fact_id = row[0]
        raise ValueError("事实合并别名存在循环")

    def history_incremental(self, device, subject, cache):
        """首次读全量，后续仅回填变更事实；向量缓存由Retriever按内容指纹复用。"""
        key = (device, subject)
        latest = self.db.execute("SELECT COALESCE(MAX(seq),0) FROM fact_changes").fetchone()[0]
        if key not in cache:
            cache[key] = {"seq":latest, "rows":{r["fact_id"]:r for r in self.history(device,subject)}}
        else:
            entry = cache[key]
            changed = [r[0] for r in self.db.execute("SELECT DISTINCT fact_id FROM fact_changes WHERE seq>?",(entry["seq"],))]
            for fid in changed:
                entry["rows"].pop(fid,None)
                row=self.db.execute("SELECT * FROM fact_assertions WHERE fact_id=? AND device_id=? AND subject_key=? AND state IN ('accepted','conflicted')",
                                    (fid,device,subject)).fetchone()
                if row:
                    item=json.loads(row["data"])
                    item.update(fact_id=fid,content_key=row["content_key"],source_decision=row["source_decision"],
                                state=row["state"],target_version=digest(dict(row)))
                    item["evidence"]=[dict(e) for e in self.db.execute("SELECT evidence_id,quote FROM fact_evidence_links WHERE fact_id=?",(fid,))]
                    entry["rows"][fid]=item
            entry["seq"]=latest
        return list(cache[key]["rows"].values())

    def refresh_projection(self, fact_id):
        row = self.db.execute("SELECT * FROM fact_assertions WHERE fact_id=?", (fact_id,)).fetchone()
        item = json.loads(row["data"])
        count = self.db.execute("SELECT COUNT(DISTINCT session_id) FROM fact_support WHERE fact_id=?", (fact_id,)).fetchone()[0]
        dates = {json.loads(r[0]).get("occurred_at", "")[:10] for r in self.db.execute(
            "SELECT e.data FROM fact_support s JOIN fact_evidence e USING(evidence_id) WHERE s.fact_id=?", (fact_id,))}
        dates.discard("")
        state = projection_state(item, count, len(dates)) if row["state"] == "accepted" else row["state"]
        self.db.execute("UPDATE fact_projections SET state=?,data=? WHERE fact_id=?",
                        (state, json.dumps({"support_count": count, "distinct_dates":len(dates), "facets":item.get("facets",{})}, ensure_ascii=False), fact_id))

    def commit_session(self, session: dict[str, Any], candidates: list[dict[str, Any]], proofs: list[dict[str, Any]],
                       decisions: dict[str, dict[str, Any]], signals: list[dict[str, Any]], tasks: list[dict[str, Any]],
                       relationships: list[dict[str, Any]] | None = None, *, finalize=True, record_members=False) -> dict[str, Any]:
        """事实级落库，最后一次事务将 Session 标记收口；幂等写入不重复计支持。"""
        run_id = digest(["fact-driven-v1", session["device_id"], session["session_id"], session["input_hash"]])
        existing = self.db.execute("SELECT status FROM fact_sessions WHERE run_id=?", (run_id,)).fetchone()
        pending = self.db.execute("SELECT 1 FROM fact_tasks WHERE device_id=? AND session_id=? AND state='technical_retry' LIMIT 1",
                                  (session["device_id"], session["session_id"])).fetchone()
        if finalize and existing and existing["status"] == "closed" and pending is None:
            return {"status": "already_committed", "run_id": run_id, "counts": {}}
        by_proof = {row["candidate_id"]: row for row in proofs}
        counts: Counter[str] = Counter()
        mappings: list[dict[str, str]] = []
        candidate_to_fact: dict[str, str] = {}
        with self.db:
            order = {"equivalent": 0, "contradictory": 1, "changed_with_direction": 2}
            relationships = sorted(relationships or [], key=lambda r: (order.get(r["relation"],3), r.get("pair_id",""), r.get("left_candidate_id",""), r.get("right_fact_id","")))
            for relation in relationships:
                for side in ("left", "right"):
                    target = relation.get(side + "_fact_id")
                    expected = relation.get(side + "_version")
                    if target and expected:
                        actual = self.db.execute("SELECT * FROM fact_assertions WHERE fact_id=?", (target,)).fetchone()
                        if actual is None or digest(dict(actual)) != expected:
                            raise ValueError("历史目标版本变化，必须重新比对")
                        if actual["device_id"] != session["device_id"]:
                            raise ValueError("关系目标不属于当前用户")
            self.db.execute("INSERT OR REPLACE INTO fact_sessions VALUES(?,?,?,?,?,?,?)", (
                run_id, session["device_id"], session["session_id"], session["input_hash"], "writing", None,
                json.dumps({"signals": signals, "task_count": len(tasks)}, ensure_ascii=False)))
            self.db.execute("DELETE FROM fact_tasks WHERE device_id=? AND session_id=?", (session["device_id"], session["session_id"]))
            for turn in session["turns"]:
                self.db.execute("INSERT OR IGNORE INTO fact_evidence VALUES(?,?,?,?)", (
                    turn["evidence_id"], session["device_id"], session["session_id"], json.dumps(turn, ensure_ascii=False)))
            for task in tasks:
                self.db.execute("INSERT OR REPLACE INTO fact_tasks VALUES(?,?,?,?,?,?,?)", (
                    task["task_id"], session["device_id"], session["session_id"], task["stage"], task["state"],
                    task["reason"], json.dumps(task, ensure_ascii=False)))
            for candidate in candidates:
                proof = by_proof.get(candidate["candidate_id"], {})
                decision = decisions.get(candidate["candidate_id"], {"action": "provisional", "reason": "缺少维护决定"})
                source = proof.get("decision", "pending")
                item = dict(candidate, facets=facets(candidate), source_decision=source, content_key=content_key(candidate),
                            maintenance=decision)
                fact = item.get("fact", {})
                if source == "reject":
                    state = "rejected"
                elif source != "pass" or decision["action"] in ("provisional", "pending"):
                    state = "provisional"
                elif decision["action"] == "conflict":
                    state = "conflicted"
                elif decision["action"] == "supersede":
                    state = "accepted"
                else:
                    state = "accepted"
                old = self.db.execute("SELECT fact_id,data FROM fact_assertions WHERE device_id=? AND content_key=?",
                                      (session["device_id"], item["content_key"])).fetchone()
                fact_id = old["fact_id"] if old else digest(["fact", session["device_id"], item["content_key"]])[:32]
                canonical = self.canonical(fact_id)
                if canonical != fact_id:
                    fact_id = canonical
                    old = self.db.execute("SELECT fact_id,data FROM fact_assertions WHERE fact_id=?", (fact_id,)).fetchone()
                if old and state == "accepted":
                    previous_state = self.db.execute("SELECT state FROM fact_assertions WHERE fact_id=?", (fact_id,)).fetchone()[0]
                    if previous_state in ("conflicted", "superseded", "suspended"):
                        state = previous_state
                now = session["end"]
                # 每个候选均有可审计去向。后来的拒绝/待确认候选不能把已经
                # 通过来源审核的同一事实降级；它们只保留为独立审核记录。
                self.db.execute("INSERT OR REPLACE INTO fact_candidate_audits VALUES(?,?,?,?,?,?,?)", (
                    candidate["candidate_id"], session["device_id"], session["session_id"], fact_id, source,
                    decision["action"], json.dumps({"candidate": candidate, "proof": proof, "decision": decision}, ensure_ascii=False)))
                old_item = json.loads(old["data"]) if old else None
                if old_item and old_item.get("source_decision") == "pass" and (source != "pass" or state == "provisional"):
                    existing_projection = self.db.execute(
                        "SELECT state FROM fact_projections WHERE fact_id=? AND projection_type='profile'", (fact_id,)
                    ).fetchone()
                    mappings.append({"candidate_id": candidate["candidate_id"], "fact_id": fact_id,
                                     "state": state, "projection": existing_projection["state"] if existing_projection else "exploring"})
                    candidate_to_fact[candidate["candidate_id"]] = fact_id
                    counts[state] += 1
                    continue
                stored_item = old_item if old_item and old_item.get("content_key") != item["content_key"] else item
                self.db.execute("""INSERT INTO fact_assertions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(fact_id) DO UPDATE SET last_seen=excluded.last_seen,source_decision=excluded.source_decision,
                    state=excluded.state,data=excluded.data""", (
                    fact_id, session["device_id"], item.get("subject_key"), item["content_key"], fact.get("type", "attribute"),
                    fact.get("predicate", "statement"), fact.get("object", ""), fact.get("value", ""), fact.get("polarity", "positive"),
                    item.get("scope", "all_time"), source, state, now, now, json.dumps(stored_item, ensure_ascii=False)))
                for evidence in item.get("evidence", []):
                    self.db.execute("INSERT OR IGNORE INTO fact_evidence_links VALUES(?,?,?)", (fact_id, evidence["evidence_id"], evidence["quote"]))
                    if state in ("accepted", "conflicted"):
                        self.db.execute("INSERT OR IGNORE INTO fact_support VALUES(?,?,?)",
                                        (fact_id, session["session_id"], evidence["evidence_id"]))
                dimensions = map_legacy_dimensions(item)
                for dimension in dimensions:
                    self.db.execute("INSERT OR IGNORE INTO fact_dimension_maps VALUES(?,?,?,?)", (fact_id, dimension, "fact-map-v2", "结构化谓词精确映射，非正文关键词"))
                if dimensions:
                    self.db.execute("DELETE FROM fact_taxonomy_gaps WHERE fact_id=?", (fact_id,))
                else:
                    self.db.execute("INSERT INTO fact_taxonomy_gaps VALUES(?,?,?,?) ON CONFLICT(fact_id) DO UPDATE SET state=excluded.state,reason=excluded.reason,data=excluded.data", (
                        fact_id, "open", "尚无明确106消费映射，不影响事实保存", json.dumps({"facets": item["facets"], "claim": item.get("claim", "")}, ensure_ascii=False)))
                support_dates = {json.loads(row[0]).get("occurred_at", "")[:10] for row in self.db.execute(
                    "SELECT e.data FROM fact_evidence_links l JOIN fact_evidence e ON e.evidence_id=l.evidence_id WHERE l.fact_id=?", (fact_id,))}
                support_count = self.db.execute("SELECT COUNT(DISTINCT evidence_id) FROM fact_evidence_links WHERE fact_id=?", (fact_id,)).fetchone()[0]
                projection = state if state in ("provisional", "conflicted") else projection_state(item, support_count, len(support_dates))
                self.db.execute("INSERT INTO fact_projections VALUES(?,?,?,?,?) ON CONFLICT(fact_id,projection_type) DO UPDATE SET state=excluded.state,data=excluded.data", (
                    digest([fact_id, "profile"])[:32], fact_id, "profile", projection,
                    json.dumps({"support_count": support_count, "distinct_dates": len(support_dates), "facets": item["facets"]}, ensure_ascii=False)))
                mappings.append({"candidate_id": candidate["candidate_id"], "fact_id": fact_id, "state": state, "projection": projection})
                candidate_to_fact[candidate["candidate_id"]] = fact_id
                counts[state] += 1
            for relation in relationships or []:
                left = candidate_to_fact.get(relation["left_candidate_id"], relation.get("left_fact_id", ""))
                right = candidate_to_fact.get(relation["right_candidate_id"], relation.get("right_fact_id", ""))
                left, right = self.canonical(left), self.canonical(right)
                if not left or not right or left == right:
                    continue
                if relation["relation"] == "changed_with_direction":
                    if not relation.get("basis") or relation.get("direction") not in ("left_replaces_right", "right_replaces_left"):
                        raise ValueError("替代事务缺少更正证据或方向")
                    turns = {t["evidence_id"]: t for t in session["turns"]}
                    if any(e["evidence_id"] not in turns or not e.get("quote") or
                           e["quote"] not in turns[e["evidence_id"]].get("user_text", "")
                           for e in relation["basis"]):
                        raise ValueError("替代证据不属于当前用户原话")
                    previous = right if relation["direction"] == "left_replaces_right" else left
                    self.db.execute("UPDATE fact_assertions SET state='superseded' WHERE fact_id=?", (previous,))
                    self.db.execute("UPDATE fact_projections SET state='superseded' WHERE fact_id=?", (previous,))
                    for old_relation in self.db.execute("SELECT relation_id,left_fact_id,right_fact_id,data FROM fact_relations WHERE relation='contradictory' AND state='active'").fetchall():
                        if previous in (self.canonical(old_relation["left_fact_id"]), self.canonical(old_relation["right_fact_id"])):
                            evidence = json.loads(old_relation["data"])
                            evidence["resolution"] = {"basis": relation["basis"], "superseded": previous}
                            self.db.execute("UPDATE fact_relations SET state='resolved',data=? WHERE relation_id=?",
                                            (json.dumps(evidence,ensure_ascii=False),old_relation["relation_id"]))
                if relation["relation"] == "equivalent":
                    # 优先保留已有历史目标；同批使用稳定的ID顺序。
                    left, right = self.canonical(left), self.canonical(right)
                    if left == right:
                        continue
                    canonical = self.canonical(relation.get("right_fact_id") or relation.get("left_fact_id") or min(left, right))
                    duplicate = right if canonical == left else left
                    self.db.execute("""INSERT OR IGNORE INTO fact_evidence_links
                        SELECT ?,evidence_id,quote FROM fact_evidence_links WHERE fact_id=?""", (canonical, duplicate))
                    self.db.execute("UPDATE fact_assertions SET state='merged' WHERE fact_id=?", (duplicate,))
                    self.db.execute("UPDATE fact_projections SET state='merged' WHERE fact_id=?", (duplicate,))
                    self.db.execute("INSERT OR REPLACE INTO fact_aliases VALUES(?,?)", (duplicate, canonical))
                    self.db.execute("INSERT OR IGNORE INTO fact_support SELECT ?,session_id,evidence_id FROM fact_support WHERE fact_id=?",
                                    (canonical, duplicate))
                    self.db.execute("UPDATE fact_assertions SET last_seen=MAX(last_seen,?) WHERE fact_id=?",
                                    (session["end"],canonical))
                    self.db.execute("UPDATE fact_candidate_audits SET fact_id=? WHERE fact_id=?", (canonical, duplicate))
                relation_id = digest(["fact-relation-v1", left, right, relation["relation"]])[:32]
                self.db.execute("INSERT OR REPLACE INTO fact_relations VALUES(?,?,?,?,?,?,?,?)", (
                    relation_id, session["device_id"], left, right, relation["relation"], "active", relation["reason"],
                    json.dumps(relation, ensure_ascii=False)))
                if relation["relation"] == "contradictory":
                    for target in (left, right):
                        self.db.execute("UPDATE fact_assertions SET state='conflicted' WHERE fact_id=? AND state='accepted'", (target,))
                        self.db.execute("UPDATE fact_projections SET state='conflicted' WHERE fact_id=?", (target,))
            from agent_maintenance import applies
            for signal in signals:
                if signal.get("kind") != "withdrawal":
                    continue
                for row in self.db.execute("SELECT fact_id,data FROM fact_assertions WHERE device_id=?",
                                           (session["device_id"],)).fetchall():
                    if applies(signal, json.loads(row["data"])):
                        self.db.execute("UPDATE fact_assertions SET state='suspended' WHERE fact_id=?", (row["fact_id"],))
                        self.db.execute("UPDATE fact_projections SET state='suspended' WHERE fact_id=?", (row["fact_id"],))
            # 冲突依据已解决后，仅没有其他开放冲突的成员恢复有效状态。
            open_members = set()
            for r in self.db.execute("SELECT left_fact_id,right_fact_id FROM fact_relations WHERE relation='contradictory' AND state='active'"):
                open_members.update(self.canonical(x) for x in r)
            for r in self.db.execute("SELECT fact_id FROM fact_assertions WHERE state='conflicted'").fetchall():
                if r[0] not in open_members:
                    self.db.execute("UPDATE fact_assertions SET state='accepted' WHERE fact_id=?", (r[0],))
            close_state = ("closed_with_tasks" if tasks else "closed") if finalize else "processing"
            for row in self.db.execute("SELECT fact_id FROM fact_assertions WHERE device_id=?", (session["device_id"],)).fetchall():
                self.refresh_projection(row[0])
            for mapping in mappings:
                mapping["fact_id"] = self.canonical(mapping["fact_id"])
                row = self.db.execute("SELECT state FROM fact_assertions WHERE fact_id=?", (mapping["fact_id"],)).fetchone()
                mapping["state"] = row[0]
                mapping["projection"] = self.db.execute("SELECT state FROM fact_projections WHERE fact_id=?", (mapping["fact_id"],)).fetchone()[0]
            self.db.execute("UPDATE fact_sessions SET status=?,closed_at=? WHERE run_id=?", (close_state, session["end"], run_id))
            if record_members:
                for c in candidates:
                    self.db.execute("INSERT OR IGNORE INTO fact_group_members VALUES(?,?,?,?)",
                        (session["device_id"],session["session_id"],session["input_hash"],c["candidate_id"]))
        return {"status": "committed", "run_id": run_id, "counts": dict(counts), "mapping": mappings}

    def completed_candidates(self, session):
        return {r[0] for r in self.db.execute("SELECT candidate_id FROM fact_group_members WHERE device_id=? AND session_id=? AND input_hash=?",
            (session["device_id"],session["session_id"],session["input_hash"]))}

    def commit_groups(self, session, candidates, proofs, decisions, signals, tasks, relationships, *, final_tasks=None):
        """独立依赖组分别原子提交，最终收口Session；失败组保留检查点。"""
        from agent_fact_relations import dependency_groups
        self.save_candidates(session, candidates)
        for group in dependency_groups(candidates, relationships):
            ids = {c["candidate_id"] for c in group}
            linked = [r for r in relationships if r.get("left_candidate_id") in ids or r.get("right_candidate_id") in ids]
            group_tasks = [t for t in tasks if not t.get("candidate_ids") or ids.intersection(t["candidate_ids"])]
            try:
                self.commit_session(session, group, [p for p in proofs if p["candidate_id"] in ids],
                    decisions, [], tasks, linked, finalize=False,
                    record_members=not group_tasks and not any(decisions[c["candidate_id"]].get("technical_pending") for c in group))
            except ValueError as exc:
                tasks.append(dict(task_id=digest(["commit", session["input_hash"],sorted(ids)]),
                    stage="commit", candidate_ids=sorted(ids), state="technical_retry",
                    reason=str(exc), code="commit_validation", session_id=session["session_id"]))
        # 控制应用与收口具有自己的原子事务，不将模型错误伪装成完成。
        receipt = self.commit_session(session, [], [], {}, signals, tasks + (final_tasks or []), [], finalize=True)
        return receipt

    def save_analysis(self, observations: list[dict[str, Any]]) -> int:
        with self.db:
            for row in observations:
                self.db.execute("""INSERT OR REPLACE INTO analysis_observations
                    (observation_id, device_id, observation_type, state, period_start, period_end, data)
                    VALUES(?,?,?,?,?,?,?)""", (
                    row["observation_id"], row["device_id"], row["observation_type"], row["state"],
                    row.get("period_start"), row.get("period_end"), json.dumps(row, ensure_ascii=False)))
        return len(observations)
