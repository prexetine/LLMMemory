"""Source proofs, single-user export, and session metric isolation; no external services."""
from collections import Counter
from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import memory_agent
from agent_contracts import verify_source
from agent_data import Registry
from agent_fact_flow import FactDrivenPipeline
from agent_fact_model import content_key
from agent_flow_data import content_hash
from agent_store import MemoryStore
import test_fact_driven as fixtures
import test_fact_runner as runner_fixtures


def capture(folder, test_id):
    destination = os.environ.get("LLMMEMORY_TEST_EVIDENCE")
    if destination:
        shutil.copytree(folder, Path(destination) / test_id.rsplit(".", 1)[-1], dirs_exist_ok=True)


class FactBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = MemoryStore(self.root / "memory.sqlite3")
        self.addCleanup(self.store.close)
        self.addCleanup(capture, self.root, self.id())
        self.config = json.loads((fixtures.ROOT / "python_app/config/memory_agent_fact_driven_8E41_0907.json")
                                 .read_text(encoding="utf-8"))
        self.config.update(session_protocol="session-reliable-v2")
        self.config.pop("dimension_mapping", None)
        self.encoder = runner_fixtures.OfflineEncoder()
        encoder_patch = patch("agent_retrieval.LocalEncoder", return_value=self.encoder)
        encoder_patch.start()
        self.addCleanup(encoder_patch.stop)
        self.agent = memory_agent.CentralAgent(
            fixtures.PipelineFake(), Registry(fixtures.ROOT / self.config["rules_file"]),
            self.store, self.config, self.root / "trace.jsonl")

    def reviewed(self):
        session = fixtures.session()
        pipeline = FactDrivenPipeline(self.agent, session)
        candidates, signals, _ = pipeline._collect()
        proofs = pipeline._review(candidates, signals)
        self.assertEqual(len(candidates), 1)
        verify_source(proofs[0], candidates[0], session)
        return pipeline, session, candidates[0], proofs[0]

    def test_derived_content_key_preserves_review_binding(self):
        _, session, candidate, proof = self.reviewed()
        before = content_hash(candidate)
        candidate["content_key"] = content_key(candidate)
        result = dict(before_hash=before, after_hash=content_hash(candidate),
                      candidate=candidate, proof=proof)
        (self.root / "binding.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertEqual(before, content_hash(candidate))
        verify_source(proof, candidate, session)

    def test_completed_pipeline_facts_replay_their_source_proofs(self):
        session = fixtures.session()
        result = self.agent.run_session(session)
        (self.root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertEqual(result["conclusion"], "complete_with_facts")
        for review in result["reviews"]:
            proof = {key: value for key, value in review.items() if key != "candidate"}
            verify_source(proof, review["candidate"], session)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM fact_assertions WHERE state='accepted'").fetchone()[0], 1)

    def test_forged_derived_key_is_rejected(self):
        _, session, candidate, proof = self.reviewed()
        candidate["content_key"] = "forged-key"
        with self.assertRaisesRegex(ValueError, "派生.*标识"):
            verify_source(proof, candidate, session)

    def test_fact_evidence_and_unknown_fields_remain_bound(self):
        _, session, original, proof = self.reviewed()
        mutations = {
            "claim": lambda c: c.update(claim="用户讨厌恐龙"),
            "fact": lambda c: c["fact"].update(value="汽车"),
            "evidence": lambda c: c["evidence"][0].update(quote="并非用户原话"),
            "subject": lambda c: c.update(subject_key="account:u1:person:other"),
            "unknown_field": lambda c: c.update(unreviewed_payload="extra"),
        }
        for name, mutate in mutations.items():
            with self.subTest(field=name):
                candidate = copy.deepcopy(original)
                candidate["content_key"] = content_key(candidate)
                mutate(candidate)
                with self.assertRaises(ValueError):
                    verify_source(proof, candidate, session)

    def test_proof_tampering_remains_rejected(self):
        _, session, candidate, proof = self.reviewed()
        proof["reason"] = "修改审核结论但不重新审核"
        with self.assertRaisesRegex(ValueError, "凭证内容已改变"):
            verify_source(proof, candidate, session)

    def test_final_fact_mutation_is_blocked_before_commit(self):
        original = FactDrivenPipeline._relation_tasks

        def mutate(pipeline, active, recalled):
            result = original(pipeline, active, recalled)
            active[0]["claim"] = "用户讨厌恐龙"
            return result

        with patch.object(FactDrivenPipeline, "_relation_tasks", mutate):
            with self.assertRaisesRegex(ValueError, "事实内容已改变"):
                self.agent.run_session(fixtures.session())
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0], 0)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM fact_candidate_audits").fetchone()[0], 1)

    def test_cached_retriever_metrics_belong_to_current_session(self):
        first = FactDrivenPipeline(self.agent, fixtures.session())
        first.run()
        first_metrics = dict(first.stats)
        retriever = self.agent.fact_retriever
        second = FactDrivenPipeline(self.agent, fixtures.session("s2", "2026-09-08T10:00:00+08:00"))
        result = second.run()
        metrics = dict(first_before=first_metrics, first_after=dict(first.stats),
                       second=dict(second.stats), same_retriever=self.agent.fact_retriever is retriever)
        (self.root / "session_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self.assertIs(self.agent.fact_retriever, retriever)
        self.assertIs(retriever.stats, second.stats)
        self.assertEqual(dict(first.stats), first_metrics)
        self.assertGreater(result["batch_metrics"].get("retrieval_seconds", 0), 0)
        self.assertGreater(result["batch_metrics"].get("embedding_resident_hits", 0), 0)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0], 1)


class SingleUserExportTests(unittest.TestCase):
    def setUp(self):
        self.case = runner_fixtures.FactRunnerTests("test_withdrawal_completes_database_reports_and_html")
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.addCleanup(capture, self.case.root, self.id())
        config = json.loads(self.case.config_path.read_text(encoding="utf-8"))
        config.pop("user_selection")
        config.update(device_id="u1", export_html=True, output_dir=str(self.case.output))
        self.case.config_path.write_text(json.dumps(config), encoding="utf-8")

    def run_single(self, *, override=True):
        with patch("memory_agent.JsonClient", return_value=self.case.model), \
                patch("bge_lifecycle.managed_encoder", return_value=nullcontext()), \
                patch("agent_retrieval.LocalEncoder", return_value=self.case.encoder), \
                redirect_stdout(io.StringIO()):
            return memory_agent.run(self.case.config_path, "live", str(self.case.output) if override else None)

    def assert_exported(self):
        report = json.loads((self.case.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "completed")
        page = (self.case.output / "index.html").read_text(encoding="utf-8")
        self.assertIn("我喜欢恐龙", page)
        with sqlite3.connect(self.case.output / "memory.sqlite3") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM fact_assertions WHERE state='accepted'").fetchone()[0], 1)

    def test_single_user_export_with_output_override(self):
        self.assertEqual(self.run_single(), 0)
        self.assert_exported()

    def test_single_user_export_with_configured_output(self):
        self.assertEqual(self.run_single(override=False), 0)
        self.assert_exported()

    def test_single_user_export_failure_is_reported_and_store_closed(self):
        opened = []

        def open_store(path):
            store = MemoryStore(path)
            opened.append(store)
            return store

        with patch("memory_agent.MemoryStore", side_effect=open_store), \
                patch("export_fact_memory_html.export_fact_html", side_effect=RuntimeError("offline export failure")):
            self.assertEqual(self.run_single(), 2)
        report = json.loads((self.case.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["export_status"], "failed")
        self.assertIn("offline export failure", report["export_error"])
        self.assertGreaterEqual(len(opened), 2)
        with self.assertRaises(sqlite3.ProgrammingError):
            opened[-1].db.execute("SELECT 1")
        with sqlite3.connect(self.case.output / "memory.sqlite3") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM fact_assertions WHERE state='accepted'").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
