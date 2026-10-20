"""输出协议重试耗尽后隔离Session，不放松入库门禁。"""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_source_agent_v3 import BASE, Fake, session
from legacy_memory_agent import run_one, CentralAgent
from agent_data import Registry
from agent_store import MemoryStore
from legacy_source_v3 import SourcePipeline
from agent_v3_contracts import ModelOutputError


class BadSubject(Fake):
    def __init__(self):
        super().__init__()
        self.bad_attempts = 0

    def ask(self, role, prompt, data):
        result = super().ask(role, prompt, data)
        if role.startswith("source_extractor:") and data["session"]["session_id"] == "bad" and result["candidates"]:
            self.bad_attempts += 1
            for c in result["candidates"]:
                c["subject"] = None
        return result


class RetryTests(unittest.TestCase):
    def cfg(self):
        cfg = json.loads((BASE / "python_app/config/memory_agent_source_v3.json").read_text(encoding="utf-8"))
        cfg["device_id"] = "test-user"
        return cfg

    def test_retry_exhausted_continues_and_resume_only_unfinished(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            cfg = self.cfg()
            sessions = [session("bad"), session("good")]
            bad = BadSubject()
            with patch("legacy_memory_agent.load_sessions", side_effect=lambda *a: (copy.deepcopy(sessions), {"device_id": "test-user"})), patch("legacy_memory_agent.JsonClient", return_value=bad), redirect_stdout(io.StringIO()):
                self.assertEqual(run_one(cfg, "live", str(out)), 2)
            self.assertEqual(bad.bad_attempts, 3)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "partial")
            self.assertEqual([r["status"] for r in report["results"]], ["failed", "completed"])
            self.assertIn("subject 必须为string", report["results"][0]["error"])
            self.assertTrue(report["results"][0]["retryable"])
            store = MemoryStore(out / "memory.sqlite3")
            try:
                self.assertEqual([r[0] for r in store.db.execute("SELECT session_id FROM sessions")], ["good"])
            finally:
                store.close()
            good = Fake()
            with patch("legacy_memory_agent.load_sessions", side_effect=lambda *a: (copy.deepcopy(sessions), {"device_id": "test-user"})), patch("legacy_memory_agent.JsonClient", return_value=good), redirect_stdout(io.StringIO()):
                self.assertEqual(run_one(cfg, "live", str(out), resume=True), 0)
            # bad路由已在第一轮缓存，重试只做失败部分；good不再跑。
            self.assertFalse(any(d.get("session", {}).get("session_id") == "good" for _, d in good.calls))
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")

    def test_malformed_json_retries_but_unexpected_error_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "m.db")
            try:
                cfg = self.cfg()
                fake = Fake()
                pipeline = SourcePipeline(CentralAgent(fake, Registry(BASE / cfg["rules_file"]), store, cfg, Path(tmp) / "trace.jsonl"), session())
                with patch.object(fake, "ask", side_effect=[json.JSONDecodeError("bad", "", 0), {"ok": True}]) as call:
                    self.assertEqual(pipeline.ask("test", "prompt", {}, lambda r: None), {"ok": True})
                    self.assertEqual(call.call_count, 2)
                with patch.object(fake, "ask", side_effect=RuntimeError("storage failure")) as call:
                    with self.assertRaises(RuntimeError):
                        pipeline.ask("test", "other", {}, lambda r: None)
                    self.assertEqual(call.call_count, 1)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
