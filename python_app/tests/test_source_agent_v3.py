"""v3 验收：来源定位、契约、维护、事务和幂等。全部离线，不消耗API。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_data import digest, Registry
from agent_store import MemoryStore
from agent_store_v3 import V3Store, empty_state, relevant, apply_decision, suspend, StaleHistory
from agent_v3_contracts import VERSION, CHECKS, seal, source_result, verify_evidence, validate_comparison
from legacy_source_v3 import SourcePipeline
from legacy_memory_agent import CentralAgent

BASE = Path(__file__).resolve().parents[2]


def session(sid="s1", text="我叫小明，我喜欢恐龙", eid=None, date="2026-09-01"):
    t = {"evidence_id": eid or sid + ":1", "user_text": text, "assistant_text": "你好", "occurred_at": date + "T10:00:00+08:00",
         "source_file": "fixture", "sheet": "test", "source_row": 1}
    return dict(device_id="test-user", session_id=sid, start=t["occurred_at"], end=t["occurred_at"], input_hash=digest([sid, t]), turns=[t])


def candidate(s, claim="喜欢恐龙", cid="c1", subject="小明", scope="日常偏好"):
    return dict(candidate_id=cid, candidate_version=1, dimension_id="G16", subject=subject, subject_key=subject,
                attribute_key="兴趣", claim=claim, scope=scope, kind="self_report", keywords=["恐龙"],
                evidence=[{"evidence_id": s["turns"][0]["evidence_id"], "quote": s["turns"][0]["user_text"]}],
                operation="new", target_memory_id="", valid_from=None, valid_to=None,
                source_windows=["w2"], source_skill_id="interests_preferences", skill_version="fixture")


def verdict(c, decision="pass"):
    return dict(candidate_id=c["candidate_id"], decision=decision, checks={k: "pass" for k in CHECKS},
                reason="原话支持", evidence_ids=[e["evidence_id"] for e in c["evidence"]], needs_context=False, context_ids=[])


def proof(c, s, decision="pass"):
    return seal(dict(verdict(c, decision), policy=VERSION, candidate_hash=digest(c), input_hash=s["input_hash"],
                     candidate_version=c["candidate_version"], evidence_pack=s["turns"], context_hash=digest(s["turns"]),
                     device_id=s["device_id"], session_id=s["session_id"]))


def decision(c, action="new", targets=(), basis=()):
    return dict(candidate_id=c["candidate_id"], action=action, targets=[{"memory_id": t["memory_id"], "revision": t["revision"]} for t in targets], reason="测试关系", basis=list(basis))


class Fake:
    def __init__(self):
        self.calls = []
        self.claim = "喜欢恐龙"

    def ask(self, role, prompt, data):
        self.calls.append((role, copy.deepcopy(data)))
        if role == "source_router":
            return {"skills": ["interests_preferences"], "signals": [], "reason": "偏好"}
        if role.startswith("source_extractor:"):
            if "G16" not in {d["dimension_id"] for d in data["skill_rules"]}:
                return {"candidates": [], "notes": "无"}
            c = candidate(data["session"], self.claim)
            for k in ("candidate_id", "candidate_version", "source_windows", "source_skill_id", "skill_version"):
                c.pop(k)
            return {"candidates": [c], "notes": ""}
        if role == "source_reviewer":
            return {"results": [verdict(c) for c in data["candidates"]]}
        if role == "history_compare":
            return {"results": [decision(c, "reinforce", data["history"][:1]) for c in data["candidates"]]}
        raise AssertionError("禁止旧角色：" + role)


class V3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3")
        self.v3 = V3Store(self.store)
        self.config = json.loads((BASE / "python_app/config/memory_agent_source_v3.json").read_text(encoding="utf-8"))
        self.registry = Registry(BASE / "J20核心用户记忆维度与提取存储规划.md")
        self.fake = Fake()
        self.agent = CentralAgent(self.fake, self.registry, self.store, self.config, Path(self.tmp.name) / "trace.jsonl")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def commit(self, s, cs, ds, signals=(), tamper=None):
        state = self.v3.snapshot(s["device_id"])
        before = digest(state)
        ps = [proof(c, s) for c in cs]
        for c, d, p in zip(cs, ds, ps):
            d.update(history_hash=digest(relevant(state, c)), source_proof=p["proof_hash"])
            apply_decision(state, c, d, p, s, signals)
        suspend(state, signals)
        plan = seal(dict(policy=VERSION, input_hash=s["input_hash"], base_hash=before, signals_hash=digest(signals), decisions=ds, result_hash=digest(state)))
        if tamper:
            tamper(plan)
        return self.v3.commit(s, cs, ps, plan, signals, {})

    def first(self):
        s = session()
        c = candidate(s)
        self.commit(s, [c], [decision(c)])
        return next(iter(self.v3.snapshot("test-user")["memories"].values()))

    def test_source_pack_uses_actual_window2_and_cross_window(self):
        s = session()
        s["turns"] += session("s2", "我现在不喜欢恐龙")['turns'] + session("s3", "我喜欢鸟")['turns']
        c = candidate(s)
        c["evidence"] = [{"evidence_id": "s2:1", "quote": "不喜欢恐龙"}, {"evidence_id": "s3:1", "quote": "我喜欢鸟"}]
        p = SourcePipeline(self.agent, s)
        out = p.review([c], [])
        self.assertEqual(out[0]["decision"], "pass")
        self.assertTrue({"s2:1", "s3:1"}.issubset({t["evidence_id"] for t in self.fake.calls[-1][1]["evidence_pack"]}))

    def test_bad_quotes_missing_evidence_blocked(self):
        for ev in ({"evidence_id": "no", "quote": "我"}, {"evidence_id": "s1:1", "quote": "我八岁"}):
            with self.assertRaises(ValueError):
                verify_evidence([ev], session())

    def test_batch_ids_must_be_exhaustive_unique_no_extra_keys(self):
        c = candidate(session())
        for rs in ([], [verdict(c), verdict(c)], [dict(verdict(c), candidate_id="no")], [dict(verdict(c), extra=True)]):
            with self.assertRaises(ValueError):
                source_result({"results": rs}, [c], {"s1:1"})

    def test_asr_pending_preserved(self):
        c = candidate(session())
        r = dict(verdict(c), decision="pending", reason="ASR数字歧义", checks={k: "uncertain" for k in CHECKS})
        source_result({"results": [r]}, [c], {"s1:1"})
        self.assertEqual(r["decision"], "pending")

    def test_candidate_change_invalidates_proof(self):
        s = session(); c = candidate(s); p = proof(c, s); c["claim"] = "喜欢数学"
        with self.assertRaises(ValueError):
            apply_decision(empty_state(), c, decision(c), p, s, [])

    def test_same_session_repetitions_count_once(self):
        s = session(); c = candidate(s); c2 = candidate(s, cid="c2")
        state = empty_state(); p = proof(c, s)
        m = apply_decision(state, c, decision(c), p, s, [])
        apply_decision(state, c2, decision(c2, "reinforce", [state["memories"][m["memory_id"]]]), proof(c2, s), s, [])
        self.assertEqual(state["memories"][m["memory_id"]]["support_count"], 1)

    def test_duplicate_import_no_support_increment(self):
        m = self.first(); s = session("s2", eid="s1:1"); c = candidate(s)
        self.commit(s, [c], [decision(c, "reinforce", [m])])
        self.assertEqual(self.v3.snapshot("test-user")["memories"][m["memory_id"]]["support_count"], 1)

    def test_new_session_support_increments_same_id(self):
        m = self.first(); s = session("s2"); c = candidate(s)
        self.commit(s, [c], [decision(c, "reinforce", [m])])
        self.assertEqual(self.v3.snapshot("test-user")["memories"][m["memory_id"]]["support_count"], 2)

    def test_conflict_third_member_and_reinforcement(self):
        m = self.first()
        s = session("s2", "我不喜欢恐龙"); c = candidate(s, "不喜欢恐龙")
        self.commit(s, [c], [decision(c, "conflict", [m])])
        st = self.v3.snapshot("test-user")
        self.assertEqual({x["state"] for x in st["memories"].values()}, {"conflicted"})
        s = session("s3", "我只喜欢飞龙"); c = candidate(s, "只喜欢飞龙")
        self.commit(s, [c], [decision(c, "conflict", list(st["memories"].values()))])
        st = self.v3.snapshot("test-user")
        self.assertEqual(len(next(iter(st["groups"].values()))["members"]), 3)
        old = st["memories"][m["memory_id"]]
        s = session("s4"); c = candidate(s)
        self.commit(s, [c], [decision(c, "reinforce", [old])])
        st = self.v3.snapshot("test-user")
        self.assertEqual(next(iter(st["groups"].values()))["status"], "open")

    def test_explicit_update_retains_old(self):
        m = self.first(); s = session("s2", "我改了，现在喜欢鸟"); c = candidate(s, "喜欢鸟")
        signal = dict(kind="change", subject_key="小明", attribute_key="兴趣", scope="日常", evidence=c["evidence"])
        self.commit(s, [c], [decision(c, "supersede", [m], c["evidence"])], [signal])
        st = self.v3.snapshot("test-user")
        self.assertEqual(st["memories"][m["memory_id"]]["state"], "superseded")
        self.assertTrue(st["versions"])
        self.assertTrue(any(x.get("supersedes") for x in st["memories"].values()))

    def test_no_update_without_explicit_signal(self):
        m = self.first(); s = session("s2"); c = candidate(s, "喜欢鸟")
        with self.assertRaises(ValueError):
            self.commit(s, [c], [decision(c, "supersede", [m], c["evidence"])])

    def test_late_log_no_latest_wins(self):
        m = self.first(); s = session("s2", date="2026-08-01"); c = candidate(s, "喜欢鸟")
        signal = dict(kind="change", subject_key="小明", attribute_key="兴趣", scope="日常", evidence=c["evidence"])
        with self.assertRaises(ValueError):
            self.commit(s, [c], [decision(c, "supersede", [m], c["evidence"])], [signal])

    def test_subject_unknown_not_merged(self):
        s = session(); c = candidate(s); c["subject_key"] = None
        with self.assertRaises(ValueError):
            self.commit(s, [c], [decision(c)])

    def test_device_alias_not_a_confirmed_person(self):
        s = session(); c = candidate(s); c["subject_key"] = s["device_id"]
        with self.assertRaises(ValueError):
            self.commit(s, [c], [decision(c)])

    def test_coexist(self):
        m = self.first(); s = session("s2"); c = candidate(s, "在学校喜欢恐龙", scope="在学校")
        self.commit(s, [c], [decision(c, "coexist", [m])])
        st = self.v3.snapshot("test-user")
        self.assertEqual(len(st["memories"]), 2)
        self.assertFalse(st["groups"])

    def test_withdrawal_stops_existing_without_deleting(self):
        m = self.first(); s = session("s2", "不要保存我的爱好")
        signal = dict(kind="withdrawal", subject_key="小明", attribute_key="兴趣", scope="爱好", evidence=[{"evidence_id": "s2:1", "quote": s["turns"][0]["user_text"]}])
        self.commit(s, [], [], [signal])
        self.assertEqual(self.v3.snapshot("test-user")["memories"][m["memory_id"]]["state"], "pending_control")

    def test_transaction_rollback_no_half_conflict(self):
        m = self.first(); before = self.v3.snapshot("test-user")
        s = session("s2"); c = candidate(s, "不喜欢恐龙")
        def bad(plan):
            plan["result_hash"] = "bad"
            plan["proof_hash"] = digest({k: v for k, v in plan.items() if k != "proof_hash"})
        with self.assertRaises(ValueError):
            self.commit(s, [c], [decision(c, "conflict", [m])], tamper=bad)
        self.assertEqual(self.v3.snapshot("test-user"), before)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM sessions").fetchone()[0], 1)

    def test_full_pipeline_and_resume_no_old_roles(self):
        s = session()
        result = self.agent.run_session(s)
        self.assertEqual(result["status"], "completed")
        count = len(self.fake.calls)
        self.assertEqual(self.agent.run_session(s), result)
        self.assertEqual(len(self.fake.calls), count)
        self.assertFalse({"reviewer", "auditor", "planner"} & {x[0] for x in self.fake.calls})

    def test_source_batch_single_call_for_multiple_candidates(self):
        s = session(); p = SourcePipeline(self.agent, s)
        cs = [candidate(s, cid=str(i)) for i in range(5)]
        self.assertEqual(len(p.review(cs, [])), 5)
        self.assertEqual(sum(role == "source_reviewer" for role, _ in self.fake.calls), 1)

    def test_semantic_reinforce_and_history_paging(self):
        self.agent.run_session(session())
        self.fake.claim = "对恐龙有兴趣"
        r = self.agent.run_session(session("s2"))
        self.assertEqual(r["commit"]["counts"]["reinforce"], 1)
        self.assertEqual(len(self.v3.snapshot("test-user")["memories"]), 1)

    def test_changed_revision_gate(self):
        m = self.first(); s = session("s2"); c = candidate(s)
        wrong = dict(m, revision=m["revision"] + 1)
        with self.assertRaises(StaleHistory):
            self.commit(s, [c], [decision(c, "reinforce", [wrong])])

    def test_context_supplement_once(self):
        s = session(); s["turns"] += session("extra", "这是我的爱好")['turns']
        calls = []
        def ask(role, prompt, data):
            calls.append(data)
            return {"results": [dict(verdict(c), decision="pending", needs_context=True, context_ids=["extra:1"]) for c in data["candidates"]]}
        with patch.object(self.fake, "ask", side_effect=ask):
            results = SourcePipeline(self.agent, s).review([candidate(s)], [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(results[0]["decision"], "pending")
        self.assertFalse(results[0]["needs_context"])

    def test_later_window_signals_sent_to_source_review(self):
        s = session(); c = candidate(s)
        s["turns"] += session("later", "刚才说错了，不要记住我喜欢恐龙")['turns']
        signal = dict(kind="withdrawal", subject_key="小明", attribute_key="兴趣", scope="偏好",
                      evidence=[{"evidence_id": "later:1", "quote": "不要记住我喜欢恐龙"}])
        p = SourcePipeline(self.agent, s)
        with patch("legacy_source_v3.windows", return_value=[s, s]), patch.object(p, "extract_window", side_effect=[([c], []), ([], [signal])]):
            r = p.run()
        self.assertEqual(r["commit"]["counts"]["pending"], 1)
        review_data = next(data for role, data in self.fake.calls if role == "source_reviewer")
        self.assertIn(signal, review_data["signals"])
        self.assertFalse(self.v3.snapshot("test-user")["memories"])

    def test_atomic_receipt_recovery_does_not_call_model(self):
        s = session(); original = self.agent.run_session(s)
        key = digest([VERSION, "session_result", s["input_hash"], digest(self.config)])
        with self.store.db:
            self.store.db.execute("DELETE FROM context_work WHERE work_key=?", (key,))
        count = len(self.fake.calls)
        restored = self.agent.run_session(s)
        self.assertEqual(restored["commit"], original["commit"])
        self.assertEqual(len(self.fake.calls), count)

    def test_all_history_pages_checked_no_top_n(self):
        s = session(); c = candidate(s)
        state = empty_state()
        for i in range(35):
            m = dict(c, claim="不同内容" + str(i), memory_id="mid" + str(i), revision=1, state="reviewed", device_id="test-user", last_seen=s["end"])
            state["memories"][m["memory_id"]] = m
        sent = []
        def ask(role, prompt, data):
            sent.extend(x["memory_id"] for x in data["history"])
            return {"results": [decision(c)]}
        with patch.object(self.fake, "ask", side_effect=ask):
            d = SourcePipeline(self.agent, s).compare(c, relevant(state, c), [], state)
        self.assertEqual(d["action"], "new")
        self.assertEqual(set(sent), set(state["memories"]))
        self.assertEqual(len(sent), 35)

    def test_same_batch_semantic_duplicates_not_double_inserted(self):
        s = session(); a = candidate(s); b = candidate(s, "对恐龙感兴趣", "c2")
        p = SourcePipeline(self.agent, s)
        receipt, _ = p.maintain([a, b], [proof(a, s), proof(b, s)], [], {})
        self.assertEqual(receipt["counts"]["reinforce"], 1)
        self.assertEqual(len(self.v3.snapshot("test-user")["memories"]), 1)

    def test_conflict_resolution_requires_covering_all_members(self):
        m = self.first(); s = session("s2"); c = candidate(s, "不喜欢恐龙")
        self.commit(s, [c], [decision(c, "conflict", [m])])
        st = self.v3.snapshot("test-user")
        s = session("s3", "刚才都说错了，我其实喜欢鸟"); c = candidate(s, "喜欢鸟")
        signal = dict(kind="correction", subject_key="小明", attribute_key="兴趣", scope="偏好", evidence=c["evidence"])
        self.commit(s, [c], [decision(c, "supersede", list(st["memories"].values()), c["evidence"])], [signal])
        self.assertEqual(next(iter(self.v3.snapshot("test-user")["groups"].values()))["status"], "resolved")

    def test_partial_conflict_resolution_keeps_group_open(self):
        m = self.first(); s = session("s2"); c = candidate(s, "不喜欢恐龙")
        self.commit(s, [c], [decision(c, "conflict", [m])])
        st = self.v3.snapshot("test-user")
        s = session("s3", "刚才说错了，我喜欢鸟"); c = candidate(s, "喜欢鸟")
        signal = dict(kind="correction", subject_key="小明", attribute_key="兴趣", scope="偏好", evidence=c["evidence"])
        self.commit(s, [c], [decision(c, "supersede", [st["memories"][m["memory_id"]]], c["evidence"])], [signal])
        st = self.v3.snapshot("test-user")
        self.assertEqual(next(iter(st["groups"].values()))["status"], "open")
        self.assertEqual(sum(x["state"] == "conflicted" for x in st["memories"].values()), 2)

    def test_semantic_pending_does_not_become_new(self):
        s = session(); c = candidate(s); m = dict(c, memory_id="old", revision=1, state="reviewed", device_id="test-user", last_seen=s["end"])
        m["claim"] = "别的内容"
        def ask(role, prompt, data):
            return {"results": [decision(c, "pending")]}
        with patch.object(self.fake, "ask", side_effect=ask):
            d = SourcePipeline(self.agent, s).compare(c, [m], [], empty_state())
        self.assertEqual(d["action"], "pending")

    def test_privacy_gate_cannot_be_pass_by_forged_checks(self):
        s = session(); c = candidate(s); p = proof(c, s)
        p["checks"]["privacy"] = "fail"
        p = seal({k: v for k, v in p.items() if k != "proof_hash"})
        with self.assertRaises(ValueError):
            apply_decision(empty_state(), c, decision(c), p, s, [])

    def test_independent_history_candidates_are_batched(self):
        s = session(); a = candidate(s); b = candidate(s, "九岁", "c2")
        b.update(dimension_id="G04", attribute_key="年龄")
        state = empty_state()
        for c in (a, b):
            m = dict(c, memory_id="old-" + c["candidate_id"], revision=1, state="reviewed", device_id="test-user", last_seen=s["end"], claim="同义历史")
            state["memories"][m["memory_id"]] = m
        sent = []
        def ask(role, prompt, data):
            sent.append(data)
            return {"results": [decision(c, "reinforce", [m for m in data["history"] if m["dimension_id"] == c["dimension_id"]]) for c in data["candidates"]]}
        with patch.object(self.fake, "ask", side_effect=ask):
            ds = SourcePipeline(self.agent, s).compare_batch([a, b], state, [])
        self.assertEqual(len(sent), 1)
        self.assertEqual(len(sent[0]["candidates"]), 2)
        self.assertEqual({d["action"] for d in ds.values()}, {"reinforce"})

    def test_dual_html_keeps_conflicted_and_superseded(self):
        from export_memory_html import build_data, export_user
        from memory_view_data import build_report_user
        m = self.first()
        s = session("s2"); c = candidate(s, "不喜欢恐龙")
        self.commit(s, [c], [decision(c, "conflict", [m])])
        folder = Path(self.tmp.name)
        results = []
        for sid, in self.store.db.execute("SELECT session_id FROM sessions"):
            results.append(dict(session_id=sid, status="completed", turns=1, approved=[], coverage={}, candidates=[], reviews=[]))
        report = dict(device_id="test-user", status="completed", selected_sessions=2, results=results, source_scope="dialogue_only")
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        (folder / "manifest.json").write_text(json.dumps({"review_version": VERSION}), encoding="utf-8")
        rules = BASE / "J20核心用户记忆维度与提取存储规划.md"
        formal = build_data(folder, rules)
        self.assertEqual({m["state"] for m in formal["memories"]}, {"conflicted"})
        self.assertEqual(build_report_user(folder, formal)["counts"]["conflicted"], 2)
        export_user(folder, folder / "index.html", rules)
        self.assertTrue((folder / "index_report.html").exists())
        self.assertIn("conflicted", (folder / "index_report.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
