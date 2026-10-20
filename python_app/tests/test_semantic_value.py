"""以8EB7真实错误片段验证控制链；Mock语义响应不代表线上准确率。"""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import test_source_agent_v4 as fixtures
from test_chain_facts import fact
from agent_fact_policy import same_fact, relation_guard, REVISION
from legacy_source_v4 import SourcePipeline
from agent_maintenance import V4Store, apply_decision, relevant, StaleHistory
from agent_contracts import validate_candidate, validate_comparison, subject_id
from agent_contracts import seal
from export_memory_html import build_data, export_user
from memory_view_data import build_report_user


class SemanticValueTests(unittest.TestCase):
    def setUp(self):
        self.h = fixtures.V4Tests()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)

    def raw(self, s, claim, predicate="想要获得", obj="牛奶", value="牛奶", **kw):
        return fixtures.raw(s, claim, assertion_mode="request", fact=fact(predicate, obj, value, "request"), **kw)

    def test_four_milk_variants_one_record(self):
        s = fixtures.session(text="我要牛奶。")
        self.h.fake.items = [self.raw(s, claim, p, value=v) for claim, p, v in [
            ("想要牛奶", "想要获得", "牛奶"),
            ("当前用户当前想要牛奶", "想要获取", "牛奶"),
            ("当前用户想要获得牛奶", "愿望", "想要获得牛奶"),
            ("当前用户提出想要牛奶", "请求获取", "要牛奶")]]
        self.h.fake.action = "reinforce"
        r = self.h.run_session(s)
        memories = self.h.state()["memories"]
        self.assertEqual(len(memories), 1)
        self.assertEqual(r["batch_metrics"]["session_exact_duplicates"], 2)
        m = next(iter(memories.values()))
        self.assertEqual(m["retention"], "session_record")
        self.assertEqual(m["support_count"], 1)
        self.assertEqual(r["commit"]["counts"]["reinforce"], 1)
        before = len(self.h.fake.calls)
        self.assertEqual(self.h.run_session(s), r)
        self.assertEqual(len(self.h.fake.calls), before)

    def test_semantic_reinforcement_not_overridden_by_predicate(self):
        s = fixtures.session(text="我要牛奶。")
        a = validate_candidate(self.raw(s, "要牛奶"), "interests_preferences", self.h.registry, s, "w")
        b = dict(a, fact=fact("愿望", "牛奶", "想要获得牛奶", "request"), memory_id="m", revision=1, state="reviewed")
        d = dict(action="reinforce", targets=[dict(memory_id="m", revision=1)], reason="同义", basis=[], semantic_verified=True)
        self.assertEqual(relation_guard(a, d, [b])["action"], "reinforce")
        b["fact"]["polarity"] = "negative"
        self.assertEqual(relation_guard(a, d, [b])["action"], "pending")

    def test_same_quote_two_independent_preferences(self):
        s = fixtures.session(text="我喜欢恐龙和汽车")
        self.h.fake.items = [fixtures.raw(s, "喜欢恐龙", fact=fact()), fixtures.raw(s, "喜欢汽车", fact=fact(obj="汽车"))]
        self.assertEqual(len(self.h.run_session(s)["approved"]), 2)

    def test_cross_skill_cross_dimension_same_evidence(self):
        s = fixtures.session(text="我要牛奶")
        a = validate_candidate(self.raw(s, "想要牛奶"), "interests_preferences", self.h.registry, s, "w")
        b = dict(a, dimension_id="G34", candidate_id="second", attribute_key="free-word", fact=fact("愿望", "牛奶", "想要牛奶", "request"))
        p = SourcePipeline(self.h.agent, s)
        self.h.fake.action = "reinforce"
        receipt, _ = p.maintain([a, b], [p.proof(c, fixtures.verdict(c), s["turns"], []) for c in (a, b)], [], {})
        self.assertEqual(receipt["counts"]["reinforce"], 1)
        m = next(iter(self.h.state()["memories"].values()))
        self.assertEqual(m["dimension_ids"], ["G16", "G34"])
        self.assertEqual(m["support_count"], 1)

    def test_vague_other_not_current_user(self):
        s = fixtures.session(text="他说他不要霸王龙。")
        for name in ("未明确身份的他人", "身份不明的人", "某个人", "他"):
            c = self.h.normalized(s, subject=name, subject_role="other")
            self.assertEqual(c["subject_role"], "unresolved")
            self.assertIsNone(c["subject_key"])
        self.assertEqual(self.h.normalized(fixtures.session(text="喜欢恐龙"))["subject_key"], subject_id("test-user"))
        self.assertNotEqual(self.h.normalized(s, subject="爸爸", subject_role="other")["subject_key"], subject_id("test-user"))

    def test_empty_responses_discarded_not_reviewed(self):
        s = fixtures.session(text="对呀。")
        self.h.fake.items = [fixtures.raw(s, "当前用户使用'对呀'回应确认当前对话轮次对应提及的内容", fact=fact("应答", "对话", "对呀"))]
        r = self.h.run_session(s)
        self.assertEqual(r["commit"]["discarded"], 1)
        self.assertEqual(r["commit"]["counts"]["pending"], 0)
        self.assertFalse(r["approved"])
        self.assertFalse(any(role == "source_reviewer" for role, _ in self.h.fake.calls))

    def test_contextual_yes_is_a_preference_not_discarded(self):
        s = fixtures.session(text="对呀。")
        s["turns"].insert(0, dict(fixtures.session("q", "你问吧")["turns"][0], assistant_text="你喜欢恐龙吗？"))
        self.h.fake.items = [fixtures.raw(s, "喜欢恐龙", fact=fact(), evidence=[dict(evidence_id="s1:1", quote="对呀")])]
        self.assertEqual(self.h.run_session(s)["approved"][0]["retention"], "memory")

    def test_multi_target_merge_atomic_and_support_union(self):
        s = fixtures.session(text="我喜欢恐龙")
        c = self.h.normalized(s, fact=fact())
        p = SourcePipeline(self.h.agent, s)
        proof = p.proof(c, fixtures.verdict(c), s["turns"], [])
        state = dict(memories={}, versions=[], groups={})
        base = dict(candidate_id=c["candidate_id"], action="new", targets=[], reason="新增", basis=[])
        apply_decision(state, c, base, proof, s, [])
        m = next(iter(state["memories"].values()))
        duplicate = dict(copy.deepcopy(m), memory_id="duplicate")
        state["memories"]["duplicate"] = duplicate
        self.h.store.db.execute("INSERT INTO v3_state VALUES(?,?)", ("test-user", json.dumps(state, ensure_ascii=False)))
        self.h.store.db.commit()
        original = self.h.fake.ask
        def ask(role, prompt, data):
            if role == "history_compare":
                return {"results": [dict(candidate_id=x["candidate_id"], action="reinforce", targets=[dict(memory_id=h["memory_id"], revision=h["revision"]) for h in data["history"]], related_memory_ids=[], reason="所有目标等价", basis=[]) for x in data["candidates"]]}
            return original(role, prompt, data)
        with patch.object(self.h.fake, "ask", side_effect=ask):
            receipt, _ = p.maintain([c], [proof], [], {})
        result = self.h.state()
        active = [m for m in result["memories"].values() if m["state"] == "reviewed"]
        retired = [m for m in result["memories"].values() if m["state"] == "merged"]
        self.assertEqual(len(active), 1)
        self.assertEqual(len(retired), 1)
        self.assertEqual(retired[0]["merged_into"], active[0]["memory_id"])
        self.assertEqual(active[0]["support_count"], 1)
        self.assertEqual(receipt["merged_records"], 1)
        self.assertEqual(p.store.receipt(s), receipt)

    def test_invalid_row_does_not_retry_good_row(self):
        s = fixtures.session(text="我喜欢恐龙和汽车")
        a = self.h.normalized(s, fact=fact())
        b = self.h.normalized(s, claim="喜欢汽车", fact=fact(obj="汽车"))
        histories = {c["candidate_id"]: [dict(c, memory_id=c["candidate_id"], revision=1)] for c in (a, b)}
        p = SourcePipeline(self.h.agent, s)
        calls = []
        def ask(role, prompt, data):
            calls.append([c["candidate_id"] for c in data["candidates"]])
            return {"results": [dict(candidate_id=c["candidate_id"], action="reinforce", targets=[] if c == b else [dict(memory_id=c["candidate_id"], revision=1)], reason="测试协议失败", basis=[]) for c in data["candidates"]]}
        with patch.object(self.h.fake, "ask", side_effect=ask):
            r = p.compare_rows(dict(candidates=[a, b], history=sum(histories.values(), []), signals=[]), histories)
        self.assertEqual([x["action"] for x in r["results"]], ["reinforce", "pending"])
        self.assertTrue(all(a["candidate_id"] not in row for row in calls[1:]))
        self.assertEqual(len(calls), 3)

    def test_comparison_still_rejects_missing_duplicate_extra_and_injection(self):
        s = fixtures.session(); c = self.h.normalized(s)
        row = dict(candidate_id=c["candidate_id"], action="new", targets=[], related_memory_ids=[], reason="无历史", basis=[])
        for rows in ([], [row, row], [dict(row, candidate_id="extra")], [dict(row, semantic_verified=True)]):
            with self.assertRaises(ValueError):
                validate_comparison({"results": rows}, [c], [], s)

    def test_events_different_sessions_and_conflict_members_never_merged(self):
        s = fixtures.session(text="去水上乐园玩")
        c = self.h.normalized(s, fact=fact("出游", "水上乐园", "游玩", "event"))
        m = dict(c, memory_id="m", revision=1, state="reviewed", origin_session_id="another")
        d = dict(action="reinforce", targets=[dict(memory_id="m", revision=1)], reason="同义", basis=[], semantic_verified=True)
        self.assertEqual(relation_guard(c, d, [m])["action"], "pending")
        m.update(origin_session_id=c["origin_session_id"], state="conflicted")
        other = dict(m, memory_id="other")
        d["targets"].append(dict(memory_id="other", revision=1))
        self.assertEqual(relation_guard(c, d, [m, other])["action"], "pending")

    def test_dual_export_categories_and_database_read_only(self):
        results = []
        for sid, text, raw in [
            ("likes", "我喜欢恐龙", lambda s: fixtures.raw(s, "喜欢恐龙", fact=fact())),
            ("milk", "我要牛奶", lambda s: self.raw(s, "想要牛奶")),
            ("event", "去了水上乐园", lambda s: fixtures.raw(s, "去了水上乐园", fact=fact("出游", "水上乐园", "游玩", "event"))),
            ("noise", "In binder.", lambda s: fixtures.raw(s, "当前用户说出英文语句In binder"))]:
            s = fixtures.session(sid, text)
            self.h.fake.items = [raw(s)]
            results.append(self.h.run_session(s))
        folder = self.h.out
        (folder / "report.json").write_text(json.dumps(dict(device_id="test-user", status="completed", selected_sessions=4, results=results, review_version="source-maintenance-v4")), encoding="utf-8")
        (folder / "manifest.json").write_text(json.dumps(dict(review_version="source-maintenance-v4")), encoding="utf-8")
        before = self.h.store.db.total_changes
        export_user(folder, folder / "index.html")
        cards = build_report_user(folder, build_data(folder, fixtures.BASE / self.h.cfg["rules_file"]))
        self.assertEqual(cards["counts"], dict(confirmed=1, session_record=1, event=1, discarded=1))
        self.assertEqual(next(c for c in cards["cards"] if c["status"] == "discarded")["source_decision"], "skipped")
        self.assertEqual(self.h.store.db.total_changes, before)
        self.assertIn("当次需求记录", (folder / "index_report.html").read_text(encoding="utf-8"))

    def test_previous_revision_is_read_only(self):
        self.h.run_session()
        row = self.h.store.db.execute("SELECT batch_id,data FROM v3_receipts").fetchone()
        receipt = json.loads(row["data"]); receipt.pop("implementation_revision")
        self.h.store.db.execute("UPDATE v3_receipts SET data=? WHERE batch_id=?", (json.dumps(receipt), row["batch_id"]))
        self.h.store.db.commit()
        with self.assertRaisesRegex(ValueError, "旧结果"):
            V4Store(self.h.store)

    def test_transaction_failure_rolls_back_and_can_resume(self):
        s = fixtures.session(text="我喜欢恐龙")
        c = self.h.normalized(s, fact=fact())
        p = SourcePipeline(self.h.agent, s)
        proof = p.proof(c, fixtures.verdict(c), s["turns"], [])
        original = p.store.commit
        def corrupt(session, candidates, proofs, plan, signals, coverage):
            altered = {k: v for k, v in plan.items() if k != "proof_hash"}
            altered["result_hash"] = "bad-hash"
            return original(session, candidates, proofs, seal(altered), signals, coverage)
        with patch.object(p.store, "commit", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "预演"):
                p.maintain([c], [proof], [], {})
        self.assertFalse(self.h.state()["memories"])
        self.assertIsNone(p.store.receipt(s))
        receipt, _ = p.maintain([c], [proof], [], {})
        self.assertEqual(receipt["counts"]["new"], 1)

    def test_unrelated_reference_is_not_a_write_target(self):
        s = fixtures.session(); c = self.h.normalized(s)
        m = dict(c, memory_id="m", revision=1)
        row = dict(candidate_id=c["candidate_id"], action="pending", targets=[], related_memory_ids=["m"], reason="关联但尚不能确定", basis=[])
        validate_comparison({"results": [row]}, [c], [m], s)
        with self.assertRaises(ValueError):
            validate_comparison({"results": [dict(row, related_memory_ids=["missing"])]}, [c], [m], s)


if __name__ == "__main__":
    unittest.main()
