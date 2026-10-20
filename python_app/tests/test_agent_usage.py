"""Session token日志离线回归，不调用模型或读取凭据。"""
import json
from pathlib import Path
import tempfile
import unittest

from agent_usage import session_usage


class SessionUsageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "calls.jsonl"

    def append(self, usage):
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"usage": usage}, ensure_ascii=False) + "\n")

    def test_all_calls_and_previous_sessions_excluded(self):
        self.append({"total_tokens": 9999})
        with self.assertLogs("agent_usage", level="INFO") as captured:
            with session_usage(self.path, "user-A", "session-1"):
                self.append({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                             "completion_tokens_details": {"reasoning_tokens": 2}})
                self.append({"input_tokens": 20, "output_tokens": 8})
        line = captured.output[-1]
        for part in ("user-A", "session-1", "本次模型调用=2", "输入token=30", "输出token=13", "总token=43"):
            self.assertIn(part, line)

    def test_failure_includes_known_and_unknown_calls(self):
        with self.assertLogs("agent_usage", level="INFO") as captured:
            with self.assertRaisesRegex(RuntimeError, "original"):
                with session_usage(self.path, "user", "retry"):
                    self.append({"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})
                    self.append(None)
                    raise RuntimeError("original")
        self.assertIn("status=failed", captured.output[-1])
        self.assertIn("总token=已知6(另1次未知)", captured.output[-1])

    def test_no_calls_and_next_session_are_zero(self):
        self.append({"total_tokens": 50})
        with self.assertLogs("agent_usage", level="INFO") as captured:
            with session_usage(self.path, "user", "cached") as state:
                state["status"] = "partial"
        self.assertIn("本次模型调用=0", captured.output[-1])
        self.assertIn("总token=0", captured.output[-1])
        self.assertIn("status=partial", captured.output[-1])

    def test_interrupt_and_missing_usage(self):
        with self.assertLogs("agent_usage", level="INFO") as captured:
            with self.assertRaises(KeyboardInterrupt):
                with session_usage(self.path, "user", "interrupted"):
                    self.append({})
                    raise KeyboardInterrupt()
        self.assertIn("status=interrupted", captured.output[-1])
        self.assertIn("总token=未知(缺失1次)", captured.output[-1])


if __name__ == "__main__":
    unittest.main()
