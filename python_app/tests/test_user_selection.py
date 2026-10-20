"""用户选择及隔离批次运行测试，不调用模型。"""
import json
import gzip
import io
import sqlite3
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_memory_agent import BASE, FakeClient, session
from legacy_memory_agent import run, run_one, select_devices, select_sessions


class UserSelectionTests(unittest.TestCase):
    def test_seed_users_roster_and_sessions(self):
        from agent_data import load_sessions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.setup_data(root)
            (root / "core_50_provisional.csv").rename(root / "users.csv")
            self.assertEqual(select_devices(cfg), ["u1"])
            sessions, manifest = load_sessions(root, "u1")
            self.assertEqual(len(sessions), 1)
            self.assertEqual(manifest["eligible_turns"], 1)

    def test_legacy_roster_keeps_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.setup_data(root)
            (root / "users.csv").write_text("device_id\nextra\n", encoding="utf-8")
            self.assertEqual(select_devices(cfg), ["u1"])

    def test_user_session_count_logged_before_prepare(self):
        config = json.loads((BASE / "python_app/config/memory_agent.json").read_text(encoding="utf-8"))
        config.update(device_id="log-test", max_sessions=2)
        sessions = [session(str(i)) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("legacy_memory_agent.load_sessions", return_value=(sessions, {})), self.assertLogs(
                    level="INFO") as logs, redirect_stdout(io.StringIO()):
                self.assertEqual(run_one(config, "prepare", tmp), 0)
            text = "\n".join(logs.output)
            self.assertIn("用户=log-test", text)
            self.assertIn("有效Session总数=5，本次选中=2，已完成=0，待处理=2", text)

    def test_all_sessions_is_not_limited_to_four(self):
        sessions = [{"session_id": str(i)} for i in range(12)]
        self.assertEqual(select_sessions(sessions, "all"), sessions)
        self.assertEqual(len(select_sessions(sessions, 4)), 4)
        for invalid in (None, 0, -1, True, "4", "ALL"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                select_sessions(sessions, invalid)

    def setup_data(self, root):
        (root / "core_50_provisional.csv").write_text("device_id\nu1\nu2\n", encoding="utf-8")
        for device in ("u1", "u2"):
            folder = root / "candidate_data" / device
            folder.mkdir(parents=True)
            item = {"disposition": "eligible", "occurred_at": "2026-01-01T10:00:00+08:00",
                    "source_file": "synthetic", "sheet": "test", "row": 1,
                    "data": {"device_id": device, "status": 0, "del_flag": 0,
                             "text": "测试话轮", "content": "", "dialog_id": "1", "chat_id": "1"}}
            with gzip.open(folder / "dialogue.jsonl.gz", "wt", encoding="utf-8") as f:
                f.write(json.dumps(item) + "\n")
        return {"data_root": str(root), "user_selection": {"mode": "selected", "device_ids": ["u1"]},
                "max_sessions": 4, "max_calls": 60, "gap_minutes": 30}

    def test_full_list_printed_before_any_worker_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.setup_data(root)
            cfg["user_selection"] = {"mode": "all", "device_ids": []}
            path = root / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertLogs(level="INFO") as logs:
                def workers(*args):
                    text = "\n".join(logs.output)
                    self.assertIn("[1/2] 用户=u1 有效Session总数=1", text)
                    self.assertIn("[2/2] 用户=u2 有效Session总数=1", text)
                    self.assertIn("清单总计：用户=2，本次处理Session=2", text)
                    return iter([{"device_id": d, "status": "prepared", "exit_code": 0}
                                 for d in ("u1", "u2")])
                with patch("legacy_memory_agent.iter_users", side_effect=workers):
                    self.assertEqual(run(path, "prepare", str(root / "out")), 0)

    def test_single_multiple_and_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.setup_data(Path(tmp))
            self.assertEqual(select_devices(cfg), ["u1"])
            cfg["user_selection"]["device_ids"] = ["u2", "u1"]
            self.assertEqual(select_devices(cfg), ["u2", "u1"])
            cfg["user_selection"] = {"mode": "all", "device_ids": []}
            self.assertEqual(select_devices(cfg), ["u1", "u2"])

    def test_invalid_selection_fails_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.setup_data(Path(tmp))
            bad = [
                {"mode": "all", "device_ids": ["u1"]},
                {"mode": "selected", "device_ids": []},
                {"mode": "selected", "device_ids": ["u1", "u1"]},
                {"mode": "selected", "device_ids": ["unknown"]},
                {"mode": "selected", "device_ids": ["../escape"]},
                {"mode": "selected", "device_ids": [1]},
                {"mode": "selected", "device_ids": "u1"},
                {"mode": "other", "device_ids": []},
                {"mode": "all", "device_ids": [], "extra": True},
            ]
            for selection in bad:
                cfg["user_selection"] = selection
                with self.subTest(selection=selection), self.assertRaises(ValueError):
                    select_devices(cfg)
            cfg["user_selection"] = {"mode": "selected", "device_ids": ["u1"]}
            cfg["device_id"] = "u1"
            with self.assertRaises(ValueError):
                select_devices(cfg)

    def test_missing_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.setup_data(Path(tmp))
            (Path(tmp) / "candidate_data/u1/dialogue.jsonl.gz").unlink()
            with self.assertRaises(ValueError):
                select_devices(cfg)

    def test_batch_isolation_failure_continuation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.setup_data(root)
            cfg["user_selection"] = {"mode": "all", "device_ids": []}
            cp = root / "config.json"
            cp.write_text(json.dumps(cfg), encoding="utf-8")
            out = root / "output"
            calls = []
            def fake(single, mode, path, resume):
                calls.append((single["device_id"], path, resume))
                Path(path).mkdir(exist_ok=True)
                (Path(path) / "marker").write_text("fixture", encoding="utf-8")
                if single["device_id"] == "u1" and not resume:
                    raise ValueError("模拟首用户失败")
                return 0
            with patch("legacy_memory_agent.run_one", side_effect=fake):
                self.assertEqual(run(cp, "live", str(out)), 2)
                self.assertEqual([c[0] for c in calls], ["u1", "u2"])
                self.assertNotEqual(calls[0][1], calls[1][1])
                self.assertEqual(run(cp, "live", str(out), True), 0)
                self.assertTrue(all(c[2] for c in calls[-2:]))
                with self.assertRaises(ValueError):
                    run(cp, "live", str(out))
            summary = json.loads((out / "batch_report.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            cfg["user_selection"] = {"mode": "selected", "device_ids": ["u1"]}
            cp.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaises(ValueError):
                run(cp, "live", str(out), True)

    def test_real_config_selection_without_model(self):
        cfg = json.loads((BASE / "python_app/config/memory_agent.json").read_text(encoding="utf-8"))
        selected = select_devices(cfg)
        self.assertGreater(len(selected), 0)
        self.assertEqual(selected, cfg["user_selection"]["device_ids"])
        self.assertEqual(len(set(selected)), len(selected))

    def test_two_users_end_to_end_with_fake_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self.setup_data(root)
            cfg = json.loads((BASE / "python_app/config/memory_agent.json").read_text(encoding="utf-8"))
            cfg.update(data_root=fixture["data_root"],
                       parallel_users=1,
                       user_selection={"mode": "all", "device_ids": []})
            cp = root / "config.json"
            cp.write_text(json.dumps(cfg), encoding="utf-8")
            def load(root, device, gap):
                s = session()
                s["device_id"] = device
                return [s], {"device_id": device}
            with patch("legacy_memory_agent.load_sessions", side_effect=load), patch(
                    "legacy_memory_agent.JsonClient", side_effect=lambda *a: FakeClient()), redirect_stdout(io.StringIO()):
                self.assertEqual(run(cp, "live", str(root / "out")), 0)
                self.assertEqual(run(cp, "live", str(root / "out"), True), 0)
            for device in ("u1", "u2"):
                self.assertFalse((root / "out" / device / "用户记忆维度展示.html").exists())
                db = sqlite3.connect(root / "out" / device / "memory.sqlite3")
                try:
                    self.assertEqual(db.execute("SELECT device_id,state FROM memories").fetchall(),
                                     [(device, "reviewed")])
                finally:
                    db.close()
            self.assertTrue((root / "out/index.html").is_file())
            self.assertTrue((root / "out/index_report.html").is_file())
            self.assertIn('index_report.html', (root / "out/index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
