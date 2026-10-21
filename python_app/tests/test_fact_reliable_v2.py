"""v2开发回归：合成故障，不调用远端模型或本地编码服务。"""
import copy
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from test_fact_driven import candidate, session, proof, FactStoreTests, FactPipelineTests
from agent_fact_model import content_key, map_legacy_dimensions
from agent_fact_relations import unsafe_equivalence
from agent_recovery import resume_report


class IdentityTests(unittest.TestCase):
    def test_requests_do_not_share_identity(self):
        a,b=candidate(),candidate("s2")
        for c in (a,b):
            c["fact"]["type"]="request"; c["scope"]="session"
        self.assertNotEqual(content_key(a),content_key(b))
        self.assertEqual(content_key(candidate()),content_key(candidate("s2")))

    def test_mapping_is_not_substring_matching(self):
        self.assertEqual(map_legacy_dimensions({"claim":"参加班会","fact":{}}),[])
        self.assertEqual(map_legacy_dimensions(candidate()),["G16"])
        a=candidate(); a["fact"]["predicate"]="不会"; a["claim"]="我不会游泳"
        self.assertNotEqual(map_legacy_dimensions(a),["G22"])

    def test_nontransitive_equivalence(self):
        a,b,c=candidate("a"),candidate("b"),candidate("c")
        pairs={"ab":(a,b),"bc":(b,c)}
        rel={key:{"relation":"equivalent"} for key in pairs}
        self.assertEqual(unsafe_equivalence(pairs,rel),{"c-a","c-b","c-c"})


class StoreV2Tests(FactStoreTests):
    def test_incremental_history_reads_only_changed_facts(self):
        self.store.commit_session(session(),[candidate()],[proof()],{"c-s1":{"action":"new"}},[],[])
        cache={}
        self.store.history_incremental("u1","account:u1:self",cache)
        with patch.object(self.store,"history",side_effect=AssertionError("full reload")):
            self.store.commit_session(session("s2"),[candidate("s2")],[proof("s2")],{"c-s2":{"action":"reinforce"}},[],[])
            rows=self.store.history_incremental("u1","account:u1:self",cache)
        self.assertEqual(len(rows),1)
        self.assertEqual(len(rows[0]["evidence"]),2)

    def test_accepted_history_is_sent_to_retriever(self):
        from types import SimpleNamespace
        from collections import Counter
        from agent_fact_flow import FactDrivenPipeline
        self.store.commit_session(session(),[candidate()],[proof()],{"c-s1":{"action":"new"}},[],[])
        seen=[]
        class Retriever:
            def __init__(self,db,config,stats): self.db=db
            def recall(self,candidates,state):
                seen.extend(state["memories"].values())
                return {c["candidate_id"]:list(state["memories"].values()) for c in candidates}
            def vectors(self,records): return [],[[1.0] for _ in records]
        p=object.__new__(FactDrivenPipeline)
        p.a=SimpleNamespace(store=SimpleNamespace(db=self.db),config={"retrieval":{}})
        p.fact_store=self.store; p.stats=Counter(); p.s=session("s2")
        with patch("agent_fact_flow.Retriever",Retriever):
            result=p._recall([candidate("s2")])
        self.assertEqual(len(result["c-s2"]),1)
        self.assertEqual(seen[0]["state"],"reviewed")

    def test_report_has_dimension_cards_and_is_readonly(self):
        from fact_report_view import build_fact_report
        self.store.commit_session(session(),[candidate()],[proof()],{"c-s1":{"action":"new"}},[],[])
        before=list(self.db.iterdump())
        view=build_fact_report(self.db,{"selected_sessions":[{"session_id":"s1","turns":1},{"session_id":"s2","turns":2}]})
        self.assertEqual(view["cards"][0]["dimension"],"G16")
        self.assertEqual(view["run_status"],"partial")
        self.assertEqual(view["selected_sessions"],2)
        self.assertEqual(list(self.db.iterdump()),before)

    def test_one_bad_group_does_not_rollback_other(self):
        self.store.commit_session(session(),[candidate()],[proof()],{"c-s1":{"action":"new"}},[],[])
        old=self.store.history("u1","account:u1:self")[0]
        a,b=candidate("s2"),candidate("independent")
        b["fact"]["object"]="汽车"; b["fact"]["value"]="汽车"
        tasks=[]
        relation=dict(left_candidate_id=a["candidate_id"],right_candidate_id="",right_fact_id=old["fact_id"],
            right_version="stale",relation="contradictory",reason="test")
        self.store.commit_groups(session("s2"),[a,b],[proof("s2"),proof("independent")],
            {a["candidate_id"]:{"action":"conflict"},b["candidate_id"]:{"action":"new"}},[],tasks,[relation])
        self.assertEqual(len(tasks),1)
        self.assertTrue(self.db.execute("SELECT 1 FROM fact_assertions WHERE object_text='汽车' AND state='accepted'").fetchone())
        self.assertEqual(self.store.session_result(session("s2"))["conclusion"],"partial")

    def test_missing_report_recovers_from_fact_database(self):
        self.store.ensure_protocol()
        self.store.commit_groups(session(),[candidate()],[proof()],{"c-s1":{"action":"new"}},[],[],[])
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            disk=sqlite3.connect(out/"memory.sqlite3")
            self.db.backup(disk); disk.close()
            report=resume_report(out,[session()])
            self.assertEqual(report["results"][0]["conclusion"],"complete_with_facts")
            (out/"report.json").write_text("{broken",encoding="utf-8")
            self.assertEqual(resume_report(out,[session()])["results"][0]["approved"][0]["claim"],"我喜欢恐龙")

    def test_processing_receipt_is_not_complete(self):
        self.store.save_receipt(session())
        self.store.save_candidates(session(),[candidate()])
        view=self.store.session_result(session())
        self.assertEqual(view["conclusion"],"partial")
        self.assertTrue(view["retry_tasks"])


class PairV2Tests(FactPipelineTests):
    def test_successful_pair_not_resent(self):
        from agent_fact_flow import FactDrivenPipeline
        p=FactDrivenPipeline(self.agent,session())
        a,b,c=candidate(),candidate("b"),candidate("c")
        a["fact"]["predicate"]="喜爱"
        calls=[]
        class Model:
            def ask(self,role,prompt,data):
                ids=[x["pair_id"] for x in data["tasks"]]; calls.append(ids)
                selected=ids[:1] if len(calls)==1 else ids
                return {"results":[dict(pair_id=i,relation="related_but_independent",reason="不同",basis_refs=[],direction="none") for i in selected]}
        self.agent.client.client=Model()
        rows=p._judge_relations({"ab":(a,b),"ac":(a,c)},[])
        self.assertEqual(set(rows),{"ab","ac"})
        self.assertEqual(calls,[["ab","ac"],["ac"]])
        p._judge_relations({"ac":(a,c),"ab":(a,b)},[])
        self.assertEqual(len(calls),2)


if __name__=="__main__":
    unittest.main()
