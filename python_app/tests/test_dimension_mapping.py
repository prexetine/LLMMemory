"""106维协议、持久化和展示开发回归；禁止调用模型或BGE。"""
import copy
import json
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_dimension_mapping import AGGREGATE, CROSS_SOURCE, VERSION, catalog, project_session, validate
from agent_data import Registry
from agent_fact_store import FactStore
from agent_fact_flow import FactDrivenPipeline
from test_fact_driven import candidate, session, proof

ROOT = Path(__file__).resolve().parents[2]


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.registry = Registry(ROOT / "J20核心用户记忆维度与提取存储规划.md")
        self.rules = catalog(self.registry)
        self.fact = dict(fact_id="f1", evidence=[{"evidence_id": "e1"}], support_sessions=2, support_dates=2)

    def response(self, dimension="G16"):
        return {"results": [{"fact_id": "f1", "matches": [
            {"dimension_id": dimension, "reason": "已有事实满足规则", "evidence_ids": ["e1"]}],
            "deferred": [], "reason": "映射已有事实，不新增事实"}]}

    def test_all_106_have_complete_rules_and_explicit_route(self):
        self.assertEqual(len(self.rules), 106)
        for rule in self.rules:
            with self.subTest(dimension=rule["dimension_id"]):
                self.assertEqual(rule["rules"], self.registry.dimensions[rule["dimension_id"]]["rules"])
                self.assertTrue(rule["rules"])
                value = self.response(rule["dimension_id"])
                if rule["dimension_id"] in AGGREGATE | CROSS_SOURCE:
                    with self.assertRaises(ValueError):
                        validate(value, [self.fact], self.rules)
                    link = value["results"][0]["matches"].pop()
                    link.pop("evidence_ids")
                    value["results"][0]["deferred"].append(link)
                validate(value, [self.fact], self.rules)

    def test_repetition_not_single_claim(self):
        fact = dict(self.fact, support_sessions=1, support_dates=1)
        with self.assertRaises(ValueError):
            validate(self.response("G20"), [fact], self.rules)
        validate(self.response(), [fact], self.rules)

    def test_reject_missing_duplicate_extra_and_foreign_evidence(self):
        invalid = []
        value = self.response(); value["results"] = []; invalid.append(value)
        value = self.response(); value["results"] *= 2; invalid.append(value)
        value = self.response(); value["results"][0]["unexpected"] = "bad"; invalid.append(value)
        value = self.response(); value["results"][0]["fact_id"] = "other"; invalid.append(value)
        value = self.response("G99"); invalid.append(value)
        value = self.response(); value["results"][0]["matches"][0]["evidence_ids"] = ["assistant:other"]; invalid.append(value)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                validate(value, [self.fact], self.rules)

    def test_empty_is_legitimate_not_missing(self):
        value = self.response()
        value["results"][0]["matches"] = []
        validate(value, [self.fact], self.rules)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.addCleanup(self.db.close)
        self.store = FactStore(self.db)
        self.store.commit_session(session(), [candidate()], [proof()], {"c-s1": {"action": "new"}}, [], [])
        self.p = object.__new__(FactDrivenPipeline)
        self.p.fact_store = self.store
        self.p.s = session()
        self.p.stats = Counter()
        self.p.fact_tasks = []
        self.p.runtime = SimpleNamespace(records={})
        self.p.a = SimpleNamespace(store=SimpleNamespace(db=self.db), config={"model_alias": "mock"},
                                   registry=Registry(ROOT / "J20核心用户记忆维度与提取存储规划.md"))
        self.calls = []
        self.p.ask = self.ask

    def ask(self, role, prompt, data, check):
        self.calls.append(data)
        result = {"results": [dict(fact_id=f["fact_id"], matches=[
            dict(dimension_id="G16", reason="原话明确喜欢恐龙", evidence_ids=[e["evidence_id"] for e in f["evidence"]])],
            deferred=[dict(dimension_id="G20", reason="尚无跨日期支持")], reason="单次明确自述")
            for f in data["facts"]]}
        check(result)
        return result

    def test_cache_does_not_change_facts_or_support_and_report_readonly(self):
        from fact_report_view import build_fact_report
        before = [tuple(r) for r in self.db.execute("SELECT * FROM fact_assertions")]
        support = [tuple(r) for r in self.db.execute("SELECT * FROM fact_support")]
        project_session(self.p)
        project_session(self.p)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.calls[0]["catalog"]), 106)
        self.assertEqual(before, [tuple(r) for r in self.db.execute("SELECT * FROM fact_assertions")])
        self.assertEqual(support, [tuple(r) for r in self.db.execute("SELECT * FROM fact_support")])
        snapshot = list(self.db.iterdump())
        view = build_fact_report(self.db)
        self.assertEqual(len(view["dimensions"]), 107)  # 106产品维度+未分类容器
        self.assertIn("条件待补 G20", view["cards"][0]["reason"])
        self.assertEqual(view["cards"][0]["dimension_ids"], ["G16"])
        self.assertEqual(snapshot, list(self.db.iterdump()))

    def test_failure_preserves_accepted_facts_and_registers_retry(self):
        self.p.ask = lambda *a: (_ for _ in ()).throw(ValueError("invalid response"))
        project_session(self.p)
        self.assertEqual(self.db.execute("SELECT state FROM fact_assertions").fetchone()[0], "accepted")
        self.assertEqual(self.db.execute("SELECT status FROM fact_sessions").fetchone()[0], "closed_with_tasks")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM fact_dimension_maps").fetchone()[0], 0)
        self.assertEqual(self.p.fact_tasks[0]["stage"], "dimension_mapping")

    def test_cache_corruption_and_evidence_changes_recompute(self):
        project_session(self.p)
        self.db.execute("UPDATE fact_dimension_results SET data='{}'")
        self.db.commit()
        project_session(self.p)
        self.assertEqual(len(self.calls), 2)
        self.store.commit_session(session("s2"), [candidate("s2")], [proof("s2")],
                                  {"c-s2": {"action": "reinforce"}}, [], [])
        self.p.s = session("s2")
        project_session(self.p)
        self.assertEqual(len(self.calls), 3)

    def test_html_export_contains_mapping_and_evidence(self):
        from export_fact_memory_html import export_fact_html
        project_session(self.p)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.html"
            export_fact_html(self.db, path)
            text = path.with_name("index_report.html").read_text(encoding="utf-8")
            self.assertIn("我喜欢恐龙", text)
            self.assertIn("条件待补 G20", text)
            self.assertIn("原话明确喜欢恐龙", text)

    def test_projection_receipt_survives_interruption(self):
        marker = dict(task_id="projection", stage="dimension_mapping", state="technical_retry",
                      candidate_ids=[], reason="投影未完成")
        # 使用另一Session，验证已提交事实和后续投影具有独立恢复边界。
        self.store.commit_groups(session("s2"), [candidate("s2")], [proof("s2")],
                                 {"c-s2": {"action": "reinforce"}}, [], [], [], final_tasks=[marker])
        self.assertFalse(self.store.session_completed(session("s2")))
        self.assertEqual(self.store.completed_candidates(session("s2")), {"c-s2"})
        self.p.s = session("s2")
        support = list(self.db.execute("SELECT * FROM fact_support"))
        project_session(self.p)
        self.assertTrue(self.store.session_completed(session("s2")))
        self.assertEqual(support, list(self.db.execute("SELECT * FROM fact_support")))

    def test_cached_mapping_removes_legacy_projection_reintroduced_by_commit(self):
        project_session(self.p)
        fid = self.db.execute("SELECT fact_id FROM fact_assertions").fetchone()[0]
        self.db.execute("INSERT INTO fact_dimension_maps VALUES(?,?,?,?)", (fid, "G22", "legacy", "错误旧投影"))
        self.db.commit()
        project_session(self.p)
        self.assertEqual([r[0] for r in self.db.execute("SELECT dimension_id FROM fact_dimension_maps")], ["G16"])
        self.assertEqual(len(self.calls), 1)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        import test_fact_driven as fixtures
        fixtures.FactPipelineTests.setUp(self)
        self.config["dimension_mapping"] = VERSION
        self.roles = []
        outer = self

        class Fake(fixtures.PipelineFake):
            def ask(self, role, prompt, data):
                outer.roles.append(role)
                if role == "dimension_mapper":
                    return {"results": [dict(fact_id=f["fact_id"], matches=[
                        dict(dimension_id="G16", reason="明确表达喜欢",
                             evidence_ids=[e["evidence_id"] for e in f["evidence"]])],
                        deferred=[], reason="直接自述") for f in data["facts"]]}
                return super().ask(role, prompt, data)
        self.agent.client.client = Fake()

    def test_pipeline_calls_mapping_only_after_memory_commit(self):
        result = self.agent.run_session(session())
        self.assertEqual(self.roles[-1], "dimension_mapper")
        self.assertFalse(result["retry_tasks"])
        self.assertTrue(FactStore(self.store.db).session_completed(session()))
        self.assertEqual(self.store.db.execute("SELECT mapping_version FROM fact_dimension_maps").fetchone()[0], VERSION)

    def test_interruption_before_mapping_resumes_without_new_extraction_calls(self):
        from unittest.mock import patch
        with patch("agent_dimension_mapping.project_session", side_effect=RuntimeError("offline interruption")):
            with self.assertRaises(RuntimeError):
                self.agent.run_session(session())
        reader = FactStore(self.store.db)
        self.assertFalse(reader.session_completed(session()))
        previous = list(self.roles)
        result = self.agent.run_session(session())
        self.assertEqual(self.roles, previous + ["dimension_mapper"])
        self.assertFalse(result["retry_tasks"])
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM fact_support").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
