"""逐条Review、失败关闭、修订复审和正式库隔离的离线测试。"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_review import CHECKS, ItemReviewer, make_record, validate_result
from agent_store import MemoryStore
from agent_schema import response_schema
from legacy_memory_agent import CentralAgent, EXTRACTOR, validate_candidate
from test_memory_agent import REGISTRY, FakeClient, audit, candidate, session, review_result


class ReviewTests(unittest.TestCase):
    def checked(self, s=None):
        s = s or session()
        return validate_candidate(candidate(s), "interests_preferences", REGISTRY, s, [])

    def test_gate_missing_modified_wrong_context_and_nonapproved(self):
        s, c = session(), self.checked()
        good = make_record(c, s, [], REGISTRY.version, review_result(c), "test")
        changed = dict(c, claim="被篡改的结论")
        other = dict(s, device_id="other-user")
        cases = [(c, [], s), (changed, [good], s), (c, [good], other)]
        for decision in ("revise", "defer", "reject"):
            bad = make_record(c, s, [], REGISTRY.version,
                              dict(review_result(c), decision=decision), "test")
            cases.append((c, [bad], s))
        for item, records, current in cases:
            with self.subTest(decision=records[0]["decision"] if records else "missing"):
                with tempfile.TemporaryDirectory() as tmp:
                    store = MemoryStore(Path(tmp) / "db")
                    with self.assertRaises(ValueError):
                        store.commit(current, [item], audit([item]), REGISTRY.version,
                                     "completed", [], records)
                    self.assertEqual(store.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 0)
                    self.assertEqual(store.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 0)
                    store.close()

    def test_failed_check_extra_keys_unknown_evidence_and_hypothesis(self):
        s, c = session(), self.checked()
        a = review_result(c)
        invalid = [dict(a, extra=True), dict(a, evidence_ids=["invented"]),
                   dict(a, checks=dict(a["checks"], subject="uncertain"))]
        for result in invalid:
            with self.assertRaises(ValueError):
                validate_result(result, c, s, [])
        with self.assertRaises(ValueError):
            validate_result(a, dict(c, kind="hypothesis"), s, [])
        schema = response_schema("reviewer", {})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]["checks"]["required"]), set(CHECKS))

    def test_revision_is_reviewed_again_and_preserves_audit(self):
        class Client:
            def __init__(self):
                self.calls = []
            def ask(self, role, prompt, data):
                self.calls.append(role)
                if role == "revision_router":
                    return {"dimension_ids": ["G16"]}
                if role.startswith("extractor:"):
                    revised = candidate()
                    revised["claim"] = "当前发言者自述喜欢恐龙"
                    return {"candidates": [revised], "notes": "修订主体措辞"}
                result = review_result(data["candidate"])
                if self.calls.count("reviewer") == 1:
                    result.update(decision="revise", reason="主体措辞需调整")
                return result
        client = Client()
        reviewer = ItemReviewer(client, REGISTRY, validate_candidate, EXTRACTOR)
        accepted, records = reviewer.review([self.checked()], session(), [], lambda r: None)
        self.assertEqual(client.calls, ["reviewer", "revision_router", "extractor:review_revision", "reviewer"])
        self.assertEqual([r["decision"] for r in records], ["revise", "approve"])
        self.assertEqual(len(accepted), 1)
        self.assertNotEqual(accepted[0]["candidate_id"], self.checked()["candidate_id"])

    def test_revision_loop_is_bounded(self):
        class Client:
            def ask(self, role, prompt, data):
                if role == "revision_router":
                    return {"dimension_ids": ["G16"]}
                if role.startswith("extractor:"):
                    return {"candidates": [candidate()], "notes": "仍未改正"}
                return dict(review_result(data["candidate"]), decision="revise")
        accepted, records = ItemReviewer(Client(), REGISTRY, validate_candidate, EXTRACTOR).review(
            [self.checked()], session(), [], lambda r: None)
        self.assertEqual(accepted, [])
        self.assertEqual([r["decision"] for r in records], ["revise", "revise", "defer"])

    def test_review_failure_never_writes_memory_or_indexes(self):
        class Client(FakeClient):
            def ask(self, role, prompt, data):
                if role == "reviewer":
                    raise TimeoutError("模拟服务超时")
                return super().ask(role, prompt, data)
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "db")
            agent = CentralAgent(Client(), REGISTRY, store,
                                 {"max_agent_steps": 6, "max_audit_rounds": 2, "history_limit": 10},
                                 Path(tmp) / "trace")
            result = agent.run_session(session())
            self.assertEqual(result["approved"], [])
            self.assertEqual(result["reviews"][0]["decision"], "defer")
            for table in ("memories", "keywords", "vector_queue", "memory_reviews"):
                self.assertEqual(store.db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0], 0)
            self.assertEqual(store.db.execute("SELECT decision FROM reviews").fetchone()[0], "defer")
            store.close()

    def test_formal_history_requires_active_review_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "db")
            s, c = session(), self.checked()
            record = make_record(c, s, [], REGISTRY.version, review_result(c), "test")
            store.commit(s, [c], audit([c]), REGISTRY.version, "completed", [], [record])
            self.assertEqual(len(store.history(s["device_id"], "2027", "", 10)), 1)
            store.db.execute("UPDATE memories SET state='candidate'")
            self.assertEqual(store.history(s["device_id"], "2027", "", 10), [])
            book = Path(tmp) / "book.md"
            store.export_book(book, REGISTRY)
            self.assertNotIn(c["claim"], book.read_text(encoding="utf-8"))
            store.db.execute("UPDATE memories SET state='reviewed',revision=2")
            self.assertEqual(store.history(s["device_id"], "2027", "", 10), [])
            store.close()

    def test_latest_rejection_invalidates_previous_approval(self):
        s, c = session(), self.checked()
        first = make_record(c, s, [], REGISTRY.version, review_result(c), "test")
        last = make_record(c, s, [], REGISTRY.version,
                           dict(review_result(c), decision="reject"), "test")
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "db")
            with self.assertRaises(ValueError):
                store.commit(s, [c], audit([c]), REGISTRY.version, "completed", [], [first, last])
            store.close()


if __name__ == "__main__":
    unittest.main()
