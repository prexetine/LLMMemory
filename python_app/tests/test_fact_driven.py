"""事实驱动新库的离线回归；不调用模型、BGE 或外部服务。"""
import json
import sqlite3
import unittest
import copy
import gzip
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_analysis_observations import aggregate, load_candidate_events
from agent_fact_model import deterministic_relation, facets, projection_state
from agent_fact_store import FactStore
from agent_data import Registry
from agent_store import MemoryStore
from memory_agent import CentralAgent
from agent_contracts import CHECKS, VERSION
from agent_flow_data import validate_flow

ROOT = Path(__file__).resolve().parents[2]


def session(session_id="s1", date="2026-09-07T10:00:00+08:00"):
    return {"device_id": "u1", "session_id": session_id, "input_hash": session_id + "-hash", "start": date, "end": date,
            "turns": [{"evidence_id": "u1:dialogue:" + session_id, "user_text": "我喜欢恐龙", "assistant_text": "", "occurred_at": date}]}


def candidate(session_id="s1", claim="我喜欢恐龙"):
    return {"candidate_id": "c-" + session_id, "device_id": "u1", "subject": "我", "subject_key": "account:u1:self",
            "subject_role": "current_user", "claim": claim, "scope": "all_time", "assertion_mode": "self_report",
            "fact": {"type": "attribute", "predicate": "喜欢", "object": "恐龙", "value": "恐龙", "polarity": "positive", "time_text": ""},
            "evidence": [{"evidence_id": "u1:dialogue:" + session_id, "quote": "我喜欢恐龙"}]}


def proof(session_id="s1", decision="pass"):
    return {"candidate_id": "c-" + session_id, "decision": decision}


class FactStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.addCleanup(self.db.close)
        self.store = FactStore(self.db)

    def test_same_fact_reinforces_without_dimension_identity(self):
        first, second = session("s1"), session("s2", "2026-09-08T10:00:00+08:00")
        self.store.commit_session(first, [candidate("s1")], [proof("s1")], {"c-s1": {"action": "new", "reason": "new"}}, [], [])
        receipt = self.store.commit_session(second, [candidate("s2")], [proof("s2")], {"c-s2": {"action": "reinforce", "reason": "same"}}, [], [])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0], 1)
        self.assertEqual(receipt["mapping"][0]["projection"], "stable")

    def test_receipt_survives_before_extraction(self):
        self.store.save_receipt(session())
        self.assertEqual(self.db.execute("SELECT status FROM fact_sessions").fetchone()[0], "evidence_only")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM fact_evidence").fetchone()[0], 1)
        self.assertFalse(self.store.session_completed(session()))

    def test_export_is_readonly_and_reports_separate_counts(self):
        import tempfile
        from export_fact_memory_html import export_fact_html
        self.store.commit_session(session(), [candidate()], [proof()],
                                  {"c-s1": {"action": "new", "reason": "supported"}}, [], [])
        before = list(self.db.iterdump())
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "index.html"
            export_fact_html(self.db, output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("有效事实引用1", text)
            self.assertIn("技术待补0", text)
            self.assertIn("完整处理，有有效记忆", text)
        self.assertEqual(before, list(self.db.iterdump()))

    def test_source_pass_does_not_bypass_maintenance(self):
        self.store.commit_session(session(), [candidate()], [proof()],
                                  {"c-s1": {"action": "provisional", "reason": "BGE unavailable"}}, [], [])
        self.assertEqual(self.db.execute("SELECT state FROM fact_assertions").fetchone()[0], "provisional")
        self.assertEqual(self.db.execute("SELECT state FROM fact_projections").fetchone()[0], "provisional")

    def test_reject_is_traceable_but_not_activated(self):
        row = candidate("s1")
        receipt = self.store.commit_session(session("s1"), [row], [proof("s1", "reject")], {"c-s1": {"action": "provisional", "reason": "bad evidence"}}, [], [])
        self.assertEqual(receipt["mapping"][0]["state"], "rejected")
        self.assertEqual(self.db.execute("SELECT state FROM fact_projections").fetchone()[0], "rejected")

    def test_later_rejected_duplicate_does_not_demote_accepted_fact(self):
        self.store.commit_session(session("s1"), [candidate("s1")], [proof("s1")], {"c-s1": {"action": "new", "reason": "new"}}, [], [])
        self.store.commit_session(session("s2"), [candidate("s2")], [proof("s2", "reject")], {"c-s2": {"action": "provisional", "reason": "bad"}}, [], [])
        self.assertEqual(self.db.execute("SELECT state FROM fact_assertions").fetchone()[0], "accepted")
        self.assertEqual(self.db.execute("SELECT source_decision FROM fact_candidate_audits WHERE candidate_id='c-s2'").fetchone()[0], "reject")

    def test_retry_task_prevents_false_completed_resume_state(self):
        first = session("s1")
        self.store.commit_session(first, [candidate("s1")], [proof("s1")], {"c-s1": {"action": "new", "reason": "new"}}, [], [
            {"task_id": "retry-1", "stage": "semantic_relation", "state": "technical_retry", "reason": "offline"}
        ])
        self.assertFalse(self.store.session_completed(first))
        self.store.commit_session(first, [candidate("s1")], [proof("s1")], {"c-s1": {"action": "new", "reason": "new"}}, [], [])
        self.assertTrue(self.store.session_completed(first))

    def test_analysis_requires_multiple_dates_and_keeps_coarse_location(self):
        events = [
            {"event_id": "1", "device_id": "u1", "source_type": "location", "occurred_at": "2026-09-07T10:00:00+08:00", "coarse_region": "深圳南山"},
            {"event_id": "2", "device_id": "u1", "source_type": "location", "occurred_at": "2026-09-08T10:00:00+08:00", "coarse_region": "深圳南山"},
        ]
        output = aggregate(events)
        self.assertEqual(output[0]["observation_type"], "coarse_activity_region")
        self.assertIn("粗粒度", output[0]["inference_limit"])

    def test_candidate_event_loader_removes_precise_location(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate_data" / "u1"; path.mkdir(parents=True)
            row = {"disposition": "eligible", "source_kind": "location", "record_key": "p1", "occurred_at": "2026-09-07T10:00:00+08:00",
                   "data": {"accuracy": "50", "address": "广东省深圳市南山区科技园某路1号"}, "source_file": "x", "sheet": "s", "row": 1}
            with gzip.open(path / "location.jsonl.gz", "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            loaded = load_candidate_events(Path(directory), "u1")
            self.assertEqual(loaded[0]["coarse_region"], "深圳市 南山区")
            self.assertNotIn("address", loaded[0])

    def test_fact_rules_do_not_use_dimension(self):
        left, right = candidate("s1"), candidate("s2")
        left["content_key"] = "same"; right["content_key"] = "same"
        self.assertEqual(deterministic_relation(left, right), "equivalent")
        self.assertEqual(facets(left)["category"], "preference_attitude")
        self.assertEqual(projection_state(dict(left, source_decision="pass"), 1, 1), "exploring")

    def test_merge_support_and_replay_are_idempotent(self):
        self.store.commit_session(session(), [candidate()], [proof()], {"c-s1":{"action":"new"}}, [], [])
        target = self.store.history("u1", "account:u1:self")[0]
        second = candidate("s2", "我爱恐龙")
        second["fact"]["predicate"] = "爱"
        relation = dict(left_candidate_id="c-s2",right_candidate_id="",left_fact_id="",
                        right_fact_id=target["fact_id"],right_version=target["target_version"],
                        relation="equivalent",reason="同义")
        args = (session("s2","2026-09-08T10:00:00+08:00"),[second],[proof("s2")],
                {"c-s2":{"action":"reinforce"}},[],[],[relation])
        self.store.commit_session(*args)
        self.store.commit_session(*args)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM fact_assertions WHERE state='accepted'").fetchone()[0], 1)
        data = json.loads(self.db.execute("SELECT data FROM fact_projections WHERE fact_id=?", (target["fact_id"],)).fetchone()[0])
        self.assertEqual(data["support_count"],2)
        self.assertEqual(len(self.store.session_result(args[0])["approved"]),1)

    def test_stale_target_rolls_back_all_writes(self):
        self.store.commit_session(session(),[candidate()],[proof()],{"c-s1":{"action":"new"}},[],[])
        target = self.store.history("u1","account:u1:self")[0]
        rel = dict(left_candidate_id="c-s2",right_candidate_id="",right_fact_id=target["fact_id"],
                   right_version="stale",relation="contradictory",reason="test")
        with self.assertRaises(ValueError):
            self.store.commit_session(session("s2"),[candidate("s2")],[proof("s2")],{"c-s2":{"action":"conflict"}},[],[],[rel])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM fact_candidate_audits").fetchone()[0],1)

    def test_conflict_marks_both_and_new_support_does_not_clear_it(self):
        self.store.commit_session(session(),[candidate()],[proof()],{"c-s1":{"action":"new"}},[],[])
        old = self.store.history("u1","account:u1:self")[0]
        other = candidate("s2","我不喜欢恐龙"); other["fact"]["polarity"]="negative"
        rel = dict(left_candidate_id="c-s2",right_candidate_id="",right_fact_id=old["fact_id"],
                   relation="contradictory",reason="矛盾")
        self.store.commit_session(session("s2"),[other],[proof("s2")],{"c-s2":{"action":"conflict"}},[],[],[rel])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM fact_assertions WHERE state='conflicted'").fetchone()[0],2)
        self.store.commit_session(session("s3"),[candidate("s3")],[proof("s3")],{"c-s3":{"action":"reinforce"}},[],[])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM fact_assertions WHERE state='conflicted'").fetchone()[0],2)

    def test_explicit_update_preserves_old_version(self):
        self.store.commit_session(session(),[candidate()],[proof()],{"c-s1":{"action":"new"}},[],[])
        old = self.store.history("u1","account:u1:self")[0]
        other = candidate("s2","现在不喜欢恐龙"); other["fact"]["polarity"]="negative"
        rel=dict(left_candidate_id="c-s2",right_candidate_id="",right_fact_id=old["fact_id"],
                 relation="changed_with_direction",reason="明确变化",direction="left_replaces_right",
                 basis=[{"evidence_id":"u1:dialogue:s2","quote":"现在不喜欢恐龙"}])
        changed_session = session("s2")
        changed_session["turns"][0]["user_text"] = "现在不喜欢恐龙"
        self.store.commit_session(changed_session,[other],[proof("s2")],{"c-s2":{"action":"supersede"}},[],[],[rel])
        self.assertEqual(self.db.execute("SELECT state FROM fact_assertions WHERE fact_id=?",(old["fact_id"],)).fetchone()[0],"superseded")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0],2)


class PipelineFake:
    def ask(self, role, prompt, data):
        if role == "source_control":
            return {"signals": []}
        if role == "source_router":
            return {"skills": ["interests_preferences"], "signals": [], "reason": "离线测试"}
        if role.startswith("source_extractor:"):
            turn = data["session"]["turns"][0]
            return {"candidates": [{
                "dimension_id": "UNCLASSIFIED", "subject": "我", "claim": "我喜欢恐龙", "scope": "all_time",
                "kind": "self_report", "keywords": ["喜欢", "恐龙"],
                "evidence": [{"evidence_id": turn["evidence_id"], "speaker": "user", "start": None, "end": None}],
                "operation": "new", "target_memory_id": "", "subject_role": "current_user", "assertion_mode": "self_report",
                "subject_key": None, "attribute_key": None, "valid_from": None, "valid_to": None,
                "fact": {"type": "attribute", "predicate": "喜欢", "object": "恐龙", "value": "恐龙", "polarity": "positive", "cardinality": "single", "time_reference": None},
                "structure": {"type": "fact", "context": "real", "topic": "恐龙", "steps": []},
                "retention": "memory", "retention_reason": "明确用户偏好"
            }], "notes": ""}
        if role == "source_reviewer":
            return {"results": [{"candidate_id": item["candidate_id"], "decision": "pass",
                "checks": {key: "pass" for key in CHECKS}, "reason": "用户原话支持",
                "evidence_ids": [item["evidence"][0]["evidence_id"]], "needs_context": False, "context_ids": [],
                "repair": {"kind": "none", "reason": ""}} for item in data["candidates"]]}
        raise AssertionError(role)


class FactPipelineTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3"); self.addCleanup(self.store.close)
        config = json.loads((ROOT / "python_app/config/memory_agent_fact_driven_8E41_0907.json").read_text(encoding="utf-8"))
        config.update(flow_version="fact-driven-v1", extraction_execution="isolated-skill-batch-repair-v1")
        config["session_protocol"] = "session-reliable-v2"
        self.config = config
        self.agent = CentralAgent(PipelineFake(), Registry(ROOT / config["rules_file"]), self.store, config, Path(self.tmp.name) / "trace.jsonl")
        from unittest.mock import patch
        recall = patch("agent_fact_flow.FactDrivenPipeline._recall", return_value={})
        recall.start()
        self.addCleanup(recall.stop)

    def test_end_to_end_fact_flow_writes_new_tables_only(self):
        self.assertEqual(validate_flow(self.config), "fact-driven-v1")
        result = self.agent.run_session(session())
        self.assertEqual(result["commit"]["status"], "committed")
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0], 1)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 0)
        self.assertEqual(self.store.db.execute("SELECT state FROM fact_projections").fetchone()[0], "exploring")

    def test_missing_relation_never_becomes_new(self):
        from agent_fact_flow import FactDrivenPipeline
        pipeline = FactDrivenPipeline(self.agent, session())
        row = candidate()
        result = pipeline._decisions([row], [proof()], {}, {"pair": (row, candidate("s2"))})
        self.assertTrue(result["c-s1"]["technical_pending"])
        self.assertEqual(result["c-s1"]["action"], "provisional")

    def test_control_failure_preserves_candidates_without_activation(self):
        from agent_llm import TransientModelError
        class FailedControl(PipelineFake):
            def ask(self, role, prompt, data):
                if role == "source_control":
                    raise TransientModelError("offline control")
                return super().ask(role, prompt, data)
        self.agent.client.client = FailedControl()
        result = self.agent.run_session(session())
        self.assertTrue(result["retry_tasks"])
        self.assertEqual(result["conclusion"], "partial")
        self.assertEqual(self.store.db.execute("SELECT state FROM fact_assertions").fetchone()[0], "provisional")
        self.assertFalse(FactStore(self.store.db).session_completed(session()))

    def test_router_failure_uses_all_skills_and_does_not_drop_session(self):
        from agent_fact_flow import FactDrivenPipeline
        from agent_data import SKILL_NAMES
        from agent_llm import TransientModelError
        from unittest.mock import patch
        class FailedRouter(PipelineFake):
            def ask(self, role, prompt, data):
                if role == "source_router":
                    raise TransientModelError("offline")
                return super().ask(role,prompt,data)
        self.agent.client.client = FailedRouter()
        with patch.object(FactDrivenPipeline,"extract_skill_window",return_value=[]) as extract:
            result = self.agent.run_session(session())
        self.assertEqual(extract.call_count,len(SKILL_NAMES))
        self.assertEqual(result["conclusion"],"complete_empty")
        self.assertTrue(FactStore(self.store.db).session_completed(session()))

    def test_resume_reuses_successful_model_work(self):
        from agent_llm import TransientModelError
        from collections import Counter
        class Flaky(PipelineFake):
            counts = Counter()
            fail = True
            def ask(self,role,prompt,data):
                self.counts[role] += 1
                if role == "source_reviewer" and self.fail:
                    raise TransientModelError("offline review")
                return super().ask(role,prompt,data)
        fake = Flaky()
        self.agent.client.client = fake
        first = self.agent.run_session(session())
        self.assertTrue(first["retry_tasks"])
        before = dict(fake.counts)
        fake.fail = False
        self.agent.session_attempt_budget = Counter()
        second = self.agent.run_session(session())
        self.assertFalse(second["retry_tasks"])
        self.assertEqual(fake.counts["source_router"],before["source_router"])
        self.assertEqual(fake.counts["source_extractor:interests_preferences"],before["source_extractor:interests_preferences"])


if __name__ == "__main__":
    unittest.main()
