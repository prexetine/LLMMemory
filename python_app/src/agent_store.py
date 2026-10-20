"""单一权威SQLite存储；书页、关键词与向量待生成记录共享memory_id。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent_data import Registry, digest
from agent_review import require_approval


class MemoryStore:
    """运行目录独立数据库；同用户串行、批次幂等，禁止覆盖未知历史。"""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(
          batch_id TEXT PRIMARY KEY, device_id TEXT, session_id TEXT,
          input_hash TEXT, start TEXT, end TEXT, status TEXT, coverage TEXT);
        CREATE TABLE IF NOT EXISTS evidence(
          evidence_id TEXT PRIMARY KEY, device_id TEXT, session_id TEXT, data TEXT);
        CREATE TABLE IF NOT EXISTS memories(
          memory_id TEXT PRIMARY KEY, device_id TEXT, dimension_id TEXT,
          subject TEXT, claim TEXT, scope TEXT, kind TEXT, state TEXT,
          created_at TEXT, last_seen TEXT, revision INTEGER, data TEXT);
        CREATE TABLE IF NOT EXISTS memory_evidence(
          memory_id TEXT REFERENCES memories(memory_id), evidence_id TEXT REFERENCES evidence(evidence_id),
          quote TEXT, PRIMARY KEY(memory_id,evidence_id,quote));
        CREATE TABLE IF NOT EXISTS keywords(
          word TEXT, memory_id TEXT REFERENCES memories(memory_id), PRIMARY KEY(word,memory_id));
        CREATE TABLE IF NOT EXISTS vector_queue(
          memory_id TEXT PRIMARY KEY REFERENCES memories(memory_id), revision INTEGER,
          text_hash TEXT, status TEXT);
        CREATE TABLE IF NOT EXISTS changes(
          batch_id TEXT, candidate_id TEXT, memory_id TEXT, operation TEXT,
          target_id TEXT, PRIMARY KEY(batch_id,candidate_id));
        CREATE TABLE IF NOT EXISTS reviews(
          review_id TEXT PRIMARY KEY, batch_id TEXT, candidate_id TEXT,
          decision TEXT, data TEXT);
        CREATE TABLE IF NOT EXISTS memory_reviews(
          memory_id TEXT REFERENCES memories(memory_id), revision INTEGER,
          review_id TEXT REFERENCES reviews(review_id), PRIMARY KEY(memory_id,revision));
        CREATE TABLE IF NOT EXISTS context_work(
          work_key TEXT PRIMARY KEY, device_id TEXT, session_id TEXT, data TEXT);
        """)
        self.db.commit()

    def history(self, device: str, before: str, query: str, limit: int, compact: bool = False) -> list[dict]:
        """仅召回历史快照中仍可使用的本设备记忆；关键词采用参数绑定。"""
        args: list = [device, before]
        sql = "SELECT DISTINCT m.* FROM memories m LEFT JOIN keywords k USING(memory_id) WHERE m.device_id=? AND m.last_seen<? AND m.state='reviewed' AND EXISTS (SELECT 1 FROM memory_reviews mr JOIN reviews r USING(review_id) WHERE mr.memory_id=m.memory_id AND mr.revision=m.revision AND r.decision='approve')"
        if query:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            sql += " AND (m.dimension_id=? OR k.word LIKE ? ESCAPE '\\' OR m.claim LIKE ? ESCAPE '\\')"
            args.extend([query, "%" + escaped + "%", "%" + escaped + "%"])
        sql += " ORDER BY m.last_seen DESC,m.memory_id LIMIT ?"
        args.append(limit)
        result = []
        for row in self.db.execute(sql, args):
            item = dict(row)
            if compact:
                evidence = self.db.execute(
                    "SELECT evidence_id,quote FROM memory_evidence WHERE memory_id=? ORDER BY evidence_id DESC LIMIT 2",
                    (row["memory_id"],)).fetchall()
                item = {k: item[k] for k in ("memory_id", "device_id", "dimension_id", "subject",
                                           "claim", "scope", "kind", "revision", "last_seen")}
                item["evidence"] = [{"evidence_id": e["evidence_id"], "quote": e["quote"],
                                     "data": {"user_text": e["quote"]}} for e in evidence]
                result.append(item)
                continue
            item["data"] = json.loads(item["data"])
            item["evidence"] = [dict(x) for x in self.db.execute(
                "SELECT e.evidence_id,e.data,me.quote FROM memory_evidence me JOIN evidence e USING(evidence_id) WHERE me.memory_id=?",
                (row["memory_id"],))]
            for ev in item["evidence"]:
                ev["data"] = json.loads(ev["data"])
            result.append(item)
        return result

    def load_work(self, key: str) -> dict | None:
        row = self.db.execute("SELECT data FROM context_work WHERE work_key=?", (key,)).fetchone()
        return json.loads(row["data"]) if row else None

    def save_work(self, key: str, session: dict, data: dict) -> None:
        # 暂存表不接入正式历史、书页和索引；正式入库门禁不变。
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO context_work VALUES(?,?,?,?)",
                            (key, session["device_id"], session["session_id"], json.dumps(data, ensure_ascii=False)))

    def commit(self, session: dict, candidates: list[dict], coverage: dict,
               version: str, status: str, history: list[dict],
               reviews: list[dict] | None = None) -> dict:
        """一次事务写入，任何目标版本错误则整批回滚。"""
        reviews = reviews or []
        if candidates and status != "completed":
            raise ValueError("未完成会话禁止提交正式记忆")
        for candidate in candidates:
            require_approval(candidate, reviews, session, history, version)
        batch_id = digest([session["device_id"], session["session_id"], session["input_hash"], version])
        old = self.db.execute("SELECT * FROM sessions WHERE batch_id=?", (batch_id,)).fetchone()
        if old:
            return {"status": "already_committed", "batch_id": batch_id, "memories": self.db.execute(
                "SELECT COUNT(*) FROM changes WHERE batch_id=?", (batch_id,)).fetchone()[0]}
        existing_session = self.db.execute("SELECT 1 FROM sessions WHERE device_id=? AND session_id=?",
                                          (session["device_id"], session["session_id"])).fetchone()
        if existing_session:
            raise ValueError("会话修订或策略变化需新运行目录，Demo不静默覆盖")
        known = {m["memory_id"]: m for m in history}
        created = []
        with self.db:
            self.db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?)",
                            (batch_id, session["device_id"], session["session_id"], session["input_hash"],
                             session["start"], session["end"], status, json.dumps(coverage, ensure_ascii=False)))
            for turn in session["turns"]:
                self.db.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?,?)",
                                (turn["evidence_id"], session["device_id"], session["session_id"],
                                 json.dumps(turn, ensure_ascii=False)))
            for review in reviews:
                self.db.execute("INSERT OR IGNORE INTO reviews VALUES(?,?,?,?,?)",
                                (review["review_id"], batch_id, review["candidate_id"],
                                 review["decision"], json.dumps(review, ensure_ascii=False)))
            for c in candidates:
                target = c["target_memory_id"]
                if target:
                    current = self.db.execute("SELECT * FROM memories WHERE memory_id=?", (target,)).fetchone()
                    if target not in known or current is None or current["device_id"] != session["device_id"] or current["revision"] != known[target]["revision"] or current["state"] != "reviewed":
                        raise ValueError("历史目标越界、已失效或版本冲突")
                mid = digest([session["device_id"], c["dimension_id"], c["subject"], c["claim"], c["scope"], c["kind"]])[:24]
                if c["operation"] == "reinforce":
                    if c["claim"] != known[target]["claim"]:
                        raise ValueError("追加证据必须保留同一结论，变化不能伪装reinforce")
                    mid = target
                prior = self.db.execute("SELECT * FROM memories WHERE memory_id=?", (mid,)).fetchone()
                if prior and prior["state"] != "reviewed":
                    raise ValueError("不可重新激活已撤回或被替代的结论")
                if c["operation"] in ("supersede", "conflict", "control") and target:
                    state = {"supersede": "superseded", "conflict": "conflict", "control": "pending_control"}[c["operation"]]
                    self.db.execute("UPDATE memories SET state=?,revision=revision+1 WHERE memory_id=?", (state, target))
                    self.db.execute("UPDATE vector_queue SET status='invalidated' WHERE memory_id=?", (target,))
                state = "reviewed"
                if c["operation"] in ("control", "conflict"):
                    state = "pending_control" if c["operation"] == "control" else "conflict"
                rev = prior["revision"] + 1 if prior else 1
                if prior:
                    self.db.execute("UPDATE memories SET last_seen=?,revision=? WHERE memory_id=?",
                                    (session["end"], rev, mid))
                else:
                    self.db.execute("INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                    (mid, session["device_id"], c["dimension_id"], c["subject"], c["claim"],
                                     c["scope"], c["kind"], state, session["end"], session["end"], rev,
                                     json.dumps(c, ensure_ascii=False)))
                for ev in c["evidence"]:
                    self.db.execute("INSERT OR IGNORE INTO memory_evidence VALUES(?,?,?)", (mid, ev["evidence_id"], ev["quote"]))
                for word in c["keywords"]:
                    self.db.execute("INSERT OR IGNORE INTO keywords VALUES(?,?)", (word, mid))
                self.db.execute("INSERT INTO vector_queue VALUES(?,?,?,?) ON CONFLICT(memory_id) DO UPDATE SET revision=excluded.revision,text_hash=excluded.text_hash,status=excluded.status",
                                (mid, rev, digest(c["claim"]), "disabled" if state == "reviewed" else "invalidated"))
                review = [r for r in reviews if r["candidate_id"] == c["candidate_id"]][-1]
                self.db.execute("INSERT INTO memory_reviews VALUES(?,?,?)",
                                (mid, rev, review["review_id"]))
                self.db.execute("INSERT INTO changes VALUES(?,?,?,?,?)",
                                (batch_id, c["candidate_id"], mid, c["operation"], target))
                created.append({"candidate_id": c["candidate_id"], "memory_id": mid, "operation": c["operation"]})
        return {"status": "committed", "batch_id": batch_id, "mapping": created,
                "vector": "disabled", "keywords": "committed"}

    def export_book(self, path: Path, registry: Registry) -> None:
        """一个Markdown内106个维度锚点，不制造106份文件。"""
        lines = ["# 用户记忆书页", "", "仅包含逐条Review通过的正式记忆，不代表人工确认或外部事实核实。向量未启用。", "", "## 目录", ""]
        sections = dict(registry.dimensions)
        if self.db.execute("SELECT 1 FROM memories WHERE dimension_id='UNCLASSIFIED' LIMIT 1").fetchone():
            sections["UNCLASSIFIED"] = {"name": "已审核内容 · 尚无产品标签（非新增产品维度）"}
        for dim, info in sections.items():
            lines.append(f"- [{dim} {info['name']}](#{dim.lower()})")
        for dim, info in sections.items():
            lines.extend(["", f'<a id="{dim.lower()}"></a>', f"## {dim} {info['name']}", ""])
            rows = list(self.db.execute("SELECT m.* FROM memories m WHERE dimension_id=? AND state='reviewed' AND EXISTS (SELECT 1 FROM memory_reviews mr JOIN reviews r USING(review_id) WHERE mr.memory_id=m.memory_id AND mr.revision=m.revision AND r.decision='approve') ORDER BY created_at", (dim,)))
            if not rows:
                lines.append("本次运行未存入可用记忆；不表示该用户不存在此信息。")
            for row in rows:
                c = json.loads(row["data"])
                lines.append(f"- [{row['memory_id']}] {row['claim'].replace(chr(10), ' ')}")
                lines.append(f"  - 主体：{row['subject']}；范围：{row['scope']}；状态：{row['state']}；关键词：{'、'.join(c['keywords'])}")
                if "retention" in c:
                    label = {"memory": "用户记忆", "event": "事件记忆", "session_record": "当次需求记录（非长期偏好）"}.get(c["retention"], c["retention"])
                    lines.append(f"  - 保存去向：{label}；依据：{c.get('retention_reason', '')}")
                for ev in self.db.execute("SELECT evidence_id,quote FROM memory_evidence WHERE memory_id=?", (row["memory_id"],)):
                    lines.append(f"  - 证据 {ev['evidence_id']}：{ev['quote'].replace(chr(10), ' ')}")
                for note in c.get("support_annotations", []):
                    if note["support_kind"] == "base_only":
                        lines.append(f"  - 仅支持基础喜好，不强化程度：{note['claim']}")
                if c.get("structure"):
                    detail = c["structure"]
                    lines.append(f"  - 内容结构：{detail['type']}；场景：{detail['context']}；主题：{detail['topic']}")
                    for step in detail["steps"]:
                        lines.append(f"  - {step['speaker']} / {step['kind']}：{step['text'].replace(chr(10), ' ')}（{step['evidence_id']}）")
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v3_state'").fetchone():
            lines.extend(["", "## 跨会话观察总结", "", "派生总结不等于长期偏好；只引用当前有效记忆。", ""])
            for state_row in self.db.execute("SELECT data FROM v3_state"):
                for summary in json.loads(state_row[0]).get("summaries", {}).values():
                    if summary["status"] != "active":
                        continue
                    lines.append(f"- {summary['claim']}（v{summary['revision']}）")
                    lines.append(f"  - 观察窗口：{summary['window_start']} 至 {summary['window_end']}")
                    for ref in summary["source_refs"]:
                        lines.append(f"  - 来源：{ref['memory_id']} / v{ref['revision']}；证据：{'、'.join(ref['evidence_ids'])}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    def close(self) -> None:
        self.db.close()
