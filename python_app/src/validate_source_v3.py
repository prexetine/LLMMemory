"""独立验证 v3 数据和实测调用；--live 仅跑两用户各前两个会话，不改生产配置。"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sqlite3

from agent_v3_contracts import VERSION, verify_seal

BASE = Path(__file__).resolve().parents[2]


def validate(folder):
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    db = sqlite3.connect((folder / "memory.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
        row = db.execute("SELECT data FROM v3_state WHERE device_id=?", (report["device_id"],)).fetchone()
        state = json.loads(row[0]) if row else {"memories": {}, "groups": {}, "versions": []}
        proofs = {r[0]: json.loads(r[1]) for r in db.execute("SELECT * FROM v3_source_reviews")}
        for p in proofs.values():
            verify_seal(p)
            assert p["policy"] == VERSION
        for m in state["memories"].values():
            assert m["device_id"] == report["device_id"]
            p = proofs[m["source_proof"]]
            assert p["decision"] == "pass"
            assert m["support_count"] == len(set(m["support_session_ids"]))
            assert len(m["support_evidence_ids"]) == len(set(m["support_evidence_ids"]))
            for e in m["evidence"]:
                raw = db.execute("SELECT data FROM evidence WHERE evidence_id=?", (e["evidence_id"],)).fetchone()
                assert raw and e["quote"] in json.loads(raw[0])["user_text"]
        versions = {(m["memory_id"], m["revision"]) for m in list(state["memories"].values()) + state["versions"]}
        for g in state["groups"].values():
            assert all((m["memory_id"], m["revision"]) in versions for m in g["members"])
            if g["status"] == "open":
                assert g["resolution"] is None
        for row in db.execute("SELECT data FROM v3_plans"):
            verify_seal(json.loads(row[0]))
        calls_path = folder / "model_calls.jsonl"
        calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()] if calls_path.exists() else []
        assert all(c["provider"] in ("Doubao", "Deepseek") for c in calls)
        assert not any(c["role"] in ("reviewer", "auditor", "planner") for c in calls)
        counts = Counter(c["role"] for c in calls)
        committed = db.execute("SELECT count(*) FROM v3_receipts").fetchone()[0]
        return {"device_id": report["device_id"], "integrity": "passed", "status": report["status"],
                "chain_passed": report["status"] == "completed" and committed == report["selected_sessions"],
                "sessions": committed, "memories": len(state["memories"]),
                "source_review_batches": counts["source_reviewer"], "history_compare_batches": counts["history_compare"],
                "model_calls": len(calls), "roles": dict(counts), "tokens": report["tokens"],
                "llm_seconds": report["llm_seconds"], "elapsed_seconds": report["elapsed_seconds"],
                "quality_note": "链路与引用校验，不是人工语义准确率评估；无历史调用时不能据此宣称已实测历史比对效果"}
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--live", action="store_true")
    args = p.parse_args()
    if args.live:
        from memory_agent import run
        if args.folder.exists():
            raise ValueError("验证目录已存在，请换新目录，不覆盖旧结果")
        config = json.loads((BASE / "python_app/config/memory_agent_source_v3.json").read_text(encoding="utf-8"))
        config.update(max_sessions=2, max_calls=60, output_dir=str(args.folder))
        config_path = args.folder.with_suffix(".config.json")
        if config_path.exists():
            raise ValueError("验证配置已存在，不覆盖")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run(config_path, "live")
    folders = [args.folder]
    if (args.folder / "batch_manifest.json").exists():
        manifest = json.loads((args.folder / "batch_manifest.json").read_text(encoding="utf-8"))
        folders = [args.folder / d for d in manifest["device_ids"]]
    results = [validate(folder) for folder in folders]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.live:
        (args.folder / "validation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(r["chain_passed"] for r in results) else 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
