"""并行调度的真实spawn、目录锁及旧配置兼容测试；无模型调用。"""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch, Mock

from test_memory_agent import BASE
from legacy_memory_agent import run
from legacy_parallel import batch_lock


class ParallelTests(unittest.TestCase):
    def test_stop_signal_blocks_next_model_call(self):
        from agent_llm import JsonClient
        event = Mock()
        event.is_set.return_value = True
        client = JsonClient.__new__(JsonClient)
        with patch("agent_parallel._STOP", event):
            with self.assertRaises(KeyboardInterrupt):
                client._ask_once("test", "test", {})

    def test_batch_lock_rejects_second_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with batch_lock(folder):
                with self.assertRaises(RuntimeError):
                    with batch_lock(folder):
                        pass
            with batch_lock(folder):
                self.assertTrue((folder / ".batch.lock").exists())

    def test_real_three_process_prepare_no_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = json.loads((BASE / "python_app/config/memory_agent.json").read_text(encoding="utf-8"))
            from legacy_memory_agent import select_devices
            all_config = dict(config, user_selection={"mode": "all", "device_ids": []})
            config["user_selection"] = {"mode": "selected", "device_ids": select_devices(all_config)[:3]}
            config["parallel_users"] = 3
            path = folder / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(run(path, "prepare", str(folder / "out")), 0)
            report = json.loads((folder / "out/batch_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["parallel_users"], 3)
            self.assertEqual(len(report["users"]), 3)
            self.assertTrue(all(r["worker_pid"] != os.getpid() for r in report["users"]))
            self.assertLessEqual(len({r["worker_pid"] for r in report["users"]}), 3)
            for row in report["users"]:
                user_dir = Path(row["output"])
                self.assertTrue((user_dir / "manifest.json").exists())
                self.assertFalse((user_dir / "model_calls.jsonl").exists())
                self.assertFalse((user_dir / "memory.sqlite3").exists())

    def test_serial_manifest_accepts_parallel_only_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = json.loads((BASE / "python_app/config/memory_agent.json").read_text(encoding="utf-8"))
            config["user_selection"]["device_ids"] = config["user_selection"]["device_ids"][:1]
            config.pop("parallel_users")
            path = folder / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            device = config["user_selection"]["device_ids"][0]
            row = {"device_id": device, "status": "completed", "exit_code": 0,
                   "output": str(folder / "out" / device), "error": None}
            with patch("legacy_memory_agent.iter_users", return_value=iter([row.copy()])), patch(
                    "export_memory_html.export_index"):
                self.assertEqual(run(path, "live", str(folder / "out")), 0)
            config["parallel_users"] = 3
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch("legacy_memory_agent.iter_users", return_value=iter([row.copy()])) as iterator, patch(
                    "export_memory_html.export_index"):
                self.assertEqual(run(path, "live", str(folder / "out"), True), 0)
                self.assertEqual(iterator.call_args.args[-1], 3)
            config["model_alias"] = "Doubao-Changed"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                run(path, "live", str(folder / "out"), True)

    def test_parallel_range(self):
        config = json.loads((BASE / "python_app/config/memory_agent.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for value in (0, 4, True, "3"):
                config["parallel_users"] = value
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.subTest(value=value), self.assertRaises(ValueError):
                    run(path, "prepare", str(Path(tmp) / "out"))


if __name__ == "__main__":
    unittest.main()
