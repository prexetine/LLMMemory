"""离线汇报状态、隐私最小化、用户隔离及双版本导出回归。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import inspect

from agent_store import MemoryStore
from export_memory_html import export_report, render, build_data, export_user, export_index, main
from memory_view_data import build_report_user, candidate_state


def candidate(cid="c1"):
    return dict(candidate_id=cid, dimension_id="G16", claim="测试候选", subject="测试用户",
                scope="当前会话", kind="self_report", keywords=["测试"],
                evidence=[{"evidence_id": "e1", "quote": "测试原话"}])


class ReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.store = MemoryStore(self.folder / "memory.sqlite3")
        self.addCleanup(self.store.close)
        self.formal = dict(device_id="u1", run_status="partial", selected_sessions=1,
                           sessions=[], memories=[], dimensions=[])

    def build(self, results):
        (self.folder / "report.json").write_text(json.dumps(dict(
            device_id="u1", status="partial", selected_sessions=len(results),
            results=results, source_scope="dialogue_only")), encoding="utf-8")
        return build_report_user(self.folder, self.formal)

    def result(self, sid="s1", **extra):
        return dict(session_id=sid, status="partial", turns=1, candidates=[candidate()],
                    approved=[], coverage={}, **extra)

    def review(self, decision="defer", sid="s1", privacy="pass"):
        return dict(candidate_id="c1", candidate=candidate(), device_id="u1", session_id=sid,
                    decision=decision, reason="证据不足", checks={"privacy": privacy})

    def test_latest_review_overrides_earlier_coverage(self):
        result = self.result(reviews=[self.review("reject"), self.review("defer")])
        result["coverage"] = {"rejected": [{"candidate_id": "c1", "reason": "旧拒绝"}]}
        data = self.build([result])
        self.assertEqual(data["cards"][0]["status"], "pending")
        self.assertEqual(data["cards"][0]["label"], "待确认")

    def test_window_approval_and_uncommitted_review_never_confirm(self):
        for review in (None, self.review("approve"), self.review("revise")):
            state, _, _ = candidate_state(candidate(), review, {})
            self.assertEqual(state, "pending")

    def test_same_candidate_id_is_scoped_to_session(self):
        data = self.build([self.result(reviews=[self.review("reject")]), self.result("s2")])
        self.assertEqual(data["counts"], {"rejected": 1, "pending": 1})
        self.assertEqual(len({c["id"] for c in data["cards"]}), 2)

    def test_foreign_review_is_rejected(self):
        review = self.review()
        review["device_id"] = "other-user"
        with self.assertRaises(ValueError):
            self.build([self.result(reviews=[review])])

    def test_committed_candidate_is_not_counted_again(self):
        self.store.db.execute("INSERT INTO sessions(batch_id,device_id,session_id) VALUES('b','u1','s1')")
        self.store.db.execute("INSERT INTO changes(batch_id,candidate_id,memory_id) VALUES('b','c1','m1')")
        self.store.db.commit()
        self.formal["memories"] = [dict(id="m1", dimension="G16", claim="正式记忆")]
        data = self.build([self.result()])
        self.assertEqual(data["counts"], {"confirmed": 1})
        self.assertEqual(data["cards"][0]["id"], "m1")

    def test_partial_uses_only_referenced_window(self):
        self.store.save_work("old", {"device_id": "u1", "session_id": "s1"},
                             {"coverage": {"rejected": [{"candidate_id": "c1", "reason": "旧拒绝"}]}})
        data = self.build([self.result(windows=[{"window_hash": "new"}])])
        self.assertEqual(data["counts"], {"pending": 1})
        self.store.save_work("new", {"device_id": "u1", "session_id": "s1"},
                             {"coverage": {"rejected": [{"candidate_id": "c1", "reason": "本轮拒绝"}]}})
        data = self.build([self.result(windows=[{"window_hash": "new"}])])
        self.assertEqual(data["cards"][0]["reason"], "本轮拒绝")
        self.assertEqual(data["counts"], {"rejected": 1})

    def test_privacy_content_not_embedded(self):
        self.formal["internal_view"] = False
        review = self.review(privacy="uncertain")
        review["candidate"]["claim"] = "PRIVATE_CLAIM"
        review["reason"] = "PRIVATE_REASON"
        data = self.build([self.result(reviews=[review])])
        content = json.dumps(data, ensure_ascii=False)
        self.assertNotIn("PRIVATE_", content)
        self.assertNotIn("测试原话", content)
        self.assertTrue(data["cards"][0]["restricted"])

    def test_export_name_link_and_html_escape(self):
        data = self.build([self.result()])
        output = export_report([data], self.folder / "正式 版.html")
        self.assertEqual(output.name, "正式 版_report.html")
        content = output.read_text(encoding="utf-8")
        self.assertIn("%E6%AD%A3%E5%BC%8F%20%E7%89%88.html", content)
        self.assertNotIn("__MEMORY_DATA__", content)
        safe = render({"value": "</script><script>alert(1)</script>"}, "__MEMORY_DATA__")
        self.assertNotIn("<script>", safe)
        self.assertEqual(json.loads(safe)["value"], "</script><script>alert(1)</script>")

    def test_internal_view_keeps_content_and_review_state(self):
        # 不传开关：自动和手工导出的默认模式都应完整展示。
        self.assertNotIn("internal_view", self.formal)
        self.formal["restricted_reviews"] = [["s1", "c1"]]
        for decision, status in (("defer", "pending"), ("reject", "rejected")):
            data = self.build([self.result(reviews=[self.review(decision, privacy="fail")])])
            card = data["cards"][0]
            self.assertEqual(card["claim"], "测试候选")
            self.assertEqual(card["subject"], "测试用户")
            self.assertEqual(card["evidence"][0]["quote"], "测试原话")
            self.assertEqual(card["reason"], "证据不足")
            self.assertEqual(card["checks"]["privacy"], "fail")
            self.assertEqual(card["status"], status)
            self.assertFalse(card["restricted"])
            output = export_report([data], self.folder / "index.html")
            content = output.read_text(encoding="utf-8")
            self.assertIn("内部查看版", content)
            self.assertNotIn("受限候选已最小化展示", content)

    def test_export_entrypoints_default_internal_and_redacted_opt_in(self):
        for func in (build_data, export_user, export_index):
            self.assertIs(inspect.signature(func).parameters["internal_view"].default, True)
        for flag, expected in (([], True), (["--internal"], True), (["--redacted"], False)):
            with patch("sys.argv", ["export_memory_html", "--input", str(self.folder), *flag]), patch("export_memory_html.export_user", return_value=self.folder / "index.html") as export, patch("builtins.print"):
                main()
            self.assertIs(export.call_args.kwargs["internal_view"], expected)


if __name__ == "__main__":
    unittest.main()
