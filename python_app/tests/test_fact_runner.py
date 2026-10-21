"""事实流程入口回归：合成输入、真实存储和导出，仅替换外部模型与编码服务。"""
from contextlib import closing, nullcontext, redirect_stdout
import gzip
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import memory_agent
from agent_fact_store import FactStore
from agent_result import summarize_results
import test_fact_driven as fixtures


class OfflineModel(fixtures.PipelineFake):
    def __init__(self):
        self.calls = []

    def ask(self, role, prompt, data):
        self.calls.append(role)
        if role == "source_control":
            turn = data["session"]["turns"][0]
            if "不要保存" in turn["user_text"]:
                return {"signals": [{
                    "kind": "withdrawal", "subject_key": None, "attribute_key": None,
                    "scope": "session", "target_type": "session", "target_text": "",
                    "evidence": [{"evidence_id": turn["evidence_id"], "speaker": "user",
                                  "start": None, "end": None}],
                    "switch_basis": "none", "switch_reference": None,
                }]}
        return super().ask(role, prompt, data)


class OfflineEncoder:
    signature = "fact-runner-offline-v1"
    dimension = 2

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FactRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.candidates = self.data / "candidate_data" / "u1"
        self.candidates.mkdir(parents=True)
        (self.data / "users.csv").write_text("device_id\nu1\n", encoding="utf-8")
        self.output = self.root / "run"
        self.user_output = self.output / "u1"
        self.database = self.user_output / "memory.sqlite3"
        self.config_path = self.root / "config.json"
        config = json.loads((fixtures.ROOT / "python_app/config/memory_agent_fact_driven_8E41_0907.json")
                            .read_text(encoding="utf-8"))
        config.update(session_protocol="session-reliable-v2", data_root=str(self.data),
                      user_selection={"mode": "selected", "device_ids": ["u1"]},
                      parallel_users=1,
                      credentials_file=str(self.root / "unused-credentials.yaml"))
        config["photos"]["enabled"] = False
        self.config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        self.model = OfflineModel()
        self.encoder = OfflineEncoder()
        self.write_dialogue("我喜欢恐龙")

    def write_rows(self, source, rows):
        with gzip.open(self.candidates / (source + ".jsonl.gz"), "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_dialogue(self, text):
        self.write_rows("dialogue", [{
            "disposition": "eligible", "source_kind": "dialogue", "record_key": "d1",
            "occurred_at": "2026-09-07T10:00:00+08:00",
            "source_file": "synthetic.xlsx", "sheet": "dialogue", "row": 1,
            "data": {"device_id": "u1", "dialog_id": "d1", "chat_id": "s1",
                     "status": "0", "del_flag": "0", "text": text, "content": ""},
        }])

    def write_locations(self, dates=("2026-09-07", "2026-09-08")):
        self.write_rows("location", [{
            "disposition": "eligible", "source_kind": "location", "record_key": f"loc-{index}",
            "occurred_at": date + "T10:00:00+08:00",
            "source_file": "synthetic.xlsx", "sheet": "location", "row": index,
            "data": {"accuracy": "50", "address": "广东省深圳市南山区测试路1号"},
        } for index, date in enumerate(dates, 1)])

    def run_batch(self, *, resume=False):
        with patch("memory_agent.JsonClient", return_value=self.model), \
                patch("bge_lifecycle.managed_encoder", return_value=nullcontext()), \
                patch("agent_retrieval.LocalEncoder", return_value=self.encoder), \
                redirect_stdout(io.StringIO()):
            return memory_agent.run(self.config_path, "live", str(self.output), resume=resume)

    def report(self):
        return json.loads((self.user_output / "report.json").read_text(encoding="utf-8"))

    def query(self, sql, parameters=()):
        with closing(sqlite3.connect(self.database)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(sql, parameters)]

    def assert_completed(self):
        report = self.report()
        self.assertEqual(report["status"], "completed", report.get("error"))
        self.assertIsNone(report["error"])
        self.assertEqual(report["technical_pending_sessions"], 0)
        batch = json.loads((self.output / "batch_report.json").read_text(encoding="utf-8"))
        self.assertEqual(batch["status"], "completed")
        self.assertEqual(batch["export_status"], "completed")
        for folder in (self.output, self.user_output):
            for name in ("index.html", "index_report.html"):
                self.assertTrue((folder / name).is_file())
        self.assertIn("状态：completed", (self.user_output / "运行报告.md").read_text(encoding="utf-8"))
        return report

    def test_withdrawal_completes_database_reports_and_html(self):
        self.write_dialogue("我喜欢恐龙，这次聊天不要保存")
        self.assertEqual(self.run_batch(), 0)
        report = self.assert_completed()
        self.assertEqual(report["candidate_outcomes"], {"storage_blocked": 1})
        result = report["results"][0]
        self.assertEqual(result["approved"], [])
        self.assertEqual(result["conclusion"], "complete_empty")
        self.assertEqual(result["session_summary"]["semantic_pending_count"], 0)
        self.assertEqual(self.query("SELECT state FROM fact_assertions"), [{"state": "suspended"}])
        self.assertEqual(self.query("SELECT state FROM fact_projections"), [{"state": "suspended"}])
        audit = json.loads(self.query("SELECT data FROM fact_candidate_audits")[0]["data"])
        control = audit["decision"]["storage_control"]
        self.assertEqual(control["status"], "blocked")
        self.assertTrue(control["reason"])
        self.assertIn("完整处理，无有效记忆", (self.user_output / "index.html").read_text(encoding="utf-8"))

    def test_resume_legacy_string_control_without_report_or_model_calls(self):
        self.write_dialogue("我喜欢恐龙，这次聊天不要保存")
        self.assertEqual(self.run_batch(), 0)
        # 模拟修复前已提交的收据；恢复兼容旧数据，不要求迁移数据库。
        with closing(sqlite3.connect(self.database)) as db, db:
            row = db.execute("SELECT candidate_id,data FROM fact_candidate_audits").fetchone()
            audit = json.loads(row[1])
            audit["decision"]["storage_control"] = "blocked"
            db.execute("UPDATE fact_candidate_audits SET data=? WHERE candidate_id=?",
                       (json.dumps(audit, ensure_ascii=False), row[0]))
        (self.user_output / "report.json").unlink()
        with patch.object(self.model, "ask", side_effect=AssertionError("已提交会话不应再次调用模型")):
            self.assertEqual(self.run_batch(resume=True), 0)
        report = self.assert_completed()
        self.assertEqual(report["candidate_outcomes"], {"storage_blocked": 1})
        self.assertEqual(report["results"][0]["conclusion"], "complete_empty")
        self.assertEqual(len(self.query("SELECT * FROM fact_assertions")), 1)
        self.assertEqual(len(self.query("SELECT * FROM fact_candidate_audits")), 1)

    def test_multiday_analysis_is_saved_exported_and_resume_is_idempotent(self):
        self.write_locations()
        self.assertEqual(self.run_batch(), 0)
        self.assert_completed()
        rows = self.query("SELECT * FROM analysis_observations")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["device_id"], "u1")
        self.assertEqual(row["observation_type"], "coarse_activity_region")
        self.assertEqual(row["state"], "exploring")
        self.assertEqual(row["period_start"], "2026-09-07T10:00:00+08:00")
        self.assertEqual(row["period_end"], "2026-09-08T10:00:00+08:00")
        detail = json.loads(row["data"])
        self.assertEqual(detail["observation_id"], row["observation_id"])
        self.assertEqual(detail["sample_count"], 2)
        self.assertEqual(detail["distinct_dates"], 2)
        self.assertEqual(detail["source_event_ids"], ["loc-1", "loc-2"])
        self.assertIn("跨日期常活动区域（exploring，覆盖2天）",
                      (self.user_output / "index.html").read_text(encoding="utf-8"))
        with patch.object(self.model, "ask", side_effect=AssertionError("续跑不应重新提取")):
            self.assertEqual(self.run_batch(resume=True), 0)
        self.assert_completed()
        self.assertEqual(self.query("SELECT * FROM analysis_observations"), rows)
        self.assertEqual(self.query("SELECT state FROM fact_assertions"), [{"state": "accepted"}])

    def test_single_day_analysis_stays_empty_without_failing_run(self):
        self.write_locations(("2026-09-07",))
        self.assertEqual(self.run_batch(), 0)
        self.assert_completed()
        self.assertEqual(self.query("SELECT * FROM analysis_observations"), [])
        self.assertEqual(self.query("SELECT state FROM fact_assertions"), [{"state": "accepted"}])
        self.assertIn("暂无跨日期分析观察", (self.user_output / "index.html").read_text(encoding="utf-8"))

    def test_analysis_failure_preserves_facts_and_resume_finishes_analysis(self):
        self.write_locations()
        with patch.object(FactStore, "save_analysis", side_effect=sqlite3.OperationalError("injected analysis failure")):
            self.assertEqual(self.run_batch(), 2)
        self.assertEqual(self.report()["status"], "failed")
        self.assertIn("analysis_observation:", self.report()["error"])
        self.assertEqual(self.query("SELECT state FROM fact_assertions"), [{"state": "accepted"}])
        self.assertEqual(self.query("SELECT * FROM analysis_observations"), [])
        with patch.object(self.model, "ask", side_effect=AssertionError("已提交事实不应重新提取")):
            self.assertEqual(self.run_batch(resume=True), 0)
        self.assert_completed()
        self.assertEqual(len(self.query("SELECT * FROM analysis_observations")), 1)
        self.assertEqual(len(self.query("SELECT * FROM fact_assertions")), 1)


class StorageControlSummaryTests(unittest.TestCase):
    def test_legacy_and_structured_controls_have_same_outcomes(self):
        for control, expected in (("blocked", "storage_blocked"),
                                  ({"status": "blocked", "reason": "明确不保存"}, "storage_blocked"),
                                  ("unresolved", "semantic_pending"),
                                  ({"status": "unresolved", "reason": "目标不明确"}, "semantic_pending")):
            with self.subTest(control=control):
                result = {"status": "completed", "commit": {"status": "committed"},
                          "reviews": [{"candidate_id": "c1", "decision": "pass"}],
                          "maintenance": [{"candidate_id": "c1", "action": "provisional", "storage_control": control}]}
                self.assertEqual(summarize_results([result], 1)["candidate_outcomes"], {expected: 1})


if __name__ == "__main__":
    unittest.main()
