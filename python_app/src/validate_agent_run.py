"""独立核验真实运行产物；不调用模型，不更改记忆数据库。"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from agent_data import digest


def validate(folder: Path) -> dict:
    """检查源证据、隔离、索引及真实跨会话读取。"""
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("review_version") == "source-maintenance-v3":
        from validate_source_v3 import validate as validate_v3
        return validate_v3(folder)
    db = sqlite3.connect((folder / "memory.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite完整性失败"
        assert not db.execute("PRAGMA foreign_key_check").fetchall(), "存在失效外键"
        memories = list(db.execute("SELECT * FROM memories"))
        review_enabled = bool(manifest.get("review_version"))
        for row in memories:
            if review_enabled and row["state"] == "reviewed":
                receipt = db.execute(
                    "SELECT r.data FROM memory_reviews mr JOIN reviews r USING(review_id) "
                    "WHERE mr.memory_id=? AND mr.revision=? AND r.decision='approve'",
                    (row["memory_id"], row["revision"])).fetchone()
                assert receipt, "正式记忆缺少当前版本Review凭证"
                review = json.loads(receipt["data"])
                assert review["candidate_hash"] == digest(review["candidate"]), "审核内容指纹不一致"
                assert review["review_id"] == digest({k: v for k, v in review.items() if k != "review_id"}), "审核记录被修改"
                assert review["device_id"] == row["device_id"], "审核主体跨设备"
                assert all(review["candidate"][k] == row[k] for k in
                           ("dimension_id", "subject", "claim", "scope", "kind")), "记忆与审核内容不一致"
            assert row["device_id"] == manifest["device_id"], "跨设备记忆"
            evidence = list(db.execute("SELECT e.*,me.quote FROM memory_evidence me JOIN evidence e USING(evidence_id) WHERE me.memory_id=?", (row["memory_id"],)))
            assert evidence, "记忆无证据"
            for ev in evidence:
                source = json.loads(ev["data"])
                assert ev["device_id"] == row["device_id"], "跨设备证据"
                assert ev["quote"] and ev["quote"] in source["user_text"], "非用户原话"
            assert db.execute("SELECT COUNT(*) FROM keywords WHERE memory_id=?", (row["memory_id"],)).fetchone()[0] > 0, "缺少关键词"
            assert db.execute("SELECT status FROM vector_queue WHERE memory_id=?", (row["memory_id"],)).fetchone()[0] in ("disabled", "invalidated"), "伪造向量状态"
        starts = {s["session_id"]: s["start"] for s in manifest["selected_sessions"]}
        history_sessions: set[str] = set()
        for line in (folder / "trace.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            bootstrap = event.get("history_bootstrap")
            if bootstrap and bootstrap["count"]:
                history_sessions.add(event["session_id"])
                for mid in bootstrap["memory_ids"]:
                    row = db.execute("SELECT device_id,created_at FROM memories WHERE memory_id=?", (mid,)).fetchone()
                    assert row and row["device_id"] == manifest["device_id"], "历史越界"
                    assert row["created_at"] < starts[event["session_id"]], "读取未来记忆"
        calls = [json.loads(x) for x in (folder / "model_calls.jsonl").read_text(encoding="utf-8").splitlines()]
        assert calls and all(c["provider"] in ("Doubao", "Deepseek") for c in calls), "未授权服务"
        book = (folder / "记忆书页.md").read_text(encoding="utf-8")
        assert book.count('<a id="') == 106, "书页不完整"
        selected_count = len(manifest["selected_sessions"])
        committed_count = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return {"storage_integrity": "passed", "run_status": report["status"],
                "selected_sessions": selected_count, "committed_sessions": committed_count,
                "memory_count": len(memories),
                "active_memory_count": sum(m["state"] == "reviewed" if review_enabled else m["state"] in ("candidate", "hypothesis") for m in memories),
                "review_gate": "checked" if review_enabled else "legacy_not_applied",
                "sessions_reading_prior_memory": len(history_sessions), "model_calls": len(calls),
                "book_dimensions": 106, "vector_status": "disabled",
                "chain_passed": bool(report["status"] == "completed" and selected_count == committed_count
                                     and memories and history_sessions),
                "quality_note": "结构与证据检查通过不代表模型结论已经人工确认"}
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = validate(args.folder)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["chain_passed"] else 2)
