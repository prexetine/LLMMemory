"""稀疏召回、任务门禁、上下文与预警回归。"""
import copy
import json
from collections import Counter
from unittest.mock import patch
from agent_sparse_relations import select_pairs, batch_manifest, verify_manifest, relevant_signals
from agent_retrieval import Retriever
from agent_relations import tasks_for
from test_unified_relations import UnifiedTests, FakeEncoder
import test_source_agent_v4 as f
import test_generic_flow as g


class SparseTests(UnifiedTests):
    def setUp(self):
        super().setUp()
        self.h.cfg['relation_policy']='sparse-relations-v2'
        self.h.cfg['retrieval']['batch_top_k']=1

    def test_sparse_neighbors_not_all_pairs(self):
        cs=[dict(candidate_id=str(i),device_id='u',subject_key='u',origin_session_id='s',evidence=[],fact={},claim=str(i)) for i in range(4)]
        pairs=select_pairs(cs,[[1,0],[.99,.01],[-1,0],[-.99,-.01]],1)
        self.assertEqual(pairs,[['0','1'],['2','3']])
        cs[0]['event_id']=cs[3]['event_id']='same-event'
        self.assertIn(['0','3'],select_pairs(cs,[[1,0],[.99,.01],[-1,0],[-.99,-.01]],1))

    def test_manifest_tamper_and_missing_vector(self):
        r=self.run_session(f.session('s1',text='我喜欢恐龙'))
        cs=r['candidates']
        retriever=Retriever(self.h.store.db,self.h.cfg['retrieval'],Counter(),FakeEncoder())
        manifest=batch_manifest(retriever,cs,1)
        verify_manifest(self.h.store.db,manifest,cs)
        bad=copy.deepcopy(manifest);bad['pairs']=[['fake','fake2']]
        with self.assertRaises(ValueError):verify_manifest(self.h.store.db,bad,cs)
        self.h.store.db.execute('DELETE FROM dense_cache')
        with self.assertRaises(ValueError):verify_manifest(self.h.store.db,manifest,cs)

    def test_signal_other_attribute_removed_global_retained(self):
        c=dict(subject_key='u',attribute_key='name',evidence=[])
        ss=[dict(subject_key='u',attribute_key='name',evidence=[]),dict(subject_key='u',attribute_key='food',evidence=[]),dict(subject_key='u',attribute_key=None,evidence=[])]
        self.assertEqual(relevant_signals(ss,[c]),[ss[0],ss[2]])

    def test_budget_block_is_not_success(self):
        self.h.cfg.update(relation_max_batches=1,relation_batch_size=1)
        s=f.session(text='我喜欢恐龙汽车苹果')
        items=[]
        for x in ('恐龙','汽车','苹果'):
            c=g.raw(s,'喜欢'+x);c['fact']['object']=x;items.append(c)
        r=self.run_session(s,items)
        self.assertEqual(r['commit_status'],'not_committed')
        self.assertTrue(any(t['code']=='relation_budget_exceeded' for t in r['retry_tasks']))
        self.assertFalse(any(role=='source_relations' for role,_ in self.fake.calls))

    def test_source_pairs_required_even_when_not_selected(self):
        s=f.session(text='我喜欢恐龙和汽车')
        a=g.raw(s);b=g.raw(s,'喜欢汽车');b['fact']['object']='汽车'
        r=self.run_session(s,[a,b])
        cs=r['candidates'];ps=[]
        for p in r['reviews']:
            q=copy.deepcopy(p);q['decision']=q['source_decision'];ps.append(q)
        _,tasks=tasks_for(cs,ps,{},[],dict(pairs=[]))
        self.assertEqual(len(tasks),1)
        self.assertEqual(tasks[0]['purpose'],['source'])

    def test_unselected_pairs_do_not_block_atomic_commit(self):
        s=f.session(text='我喜欢恐龙')
        first=copy.deepcopy(s['turns'][0]);items=[];s['turns']=[]
        for i,name in enumerate(('恐龙','汽车','苹果','音乐')):
            turn=copy.deepcopy(first)
            turn.update(evidence_id='s1:'+str(i+1),user='我喜欢'+name)
            s['turns'].append(turn)
            c=g.raw(dict(s,turns=[turn]),'喜欢'+name)
            c['fact'].update(object=name,value=name);items.append(c)
        r=self.run_session(s,items)
        self.assertEqual(r['commit']['status'],'committed')
        self.assertEqual(len(r['approved']),4)
        tasks=[t for role,data in self.fake.calls if role=='source_relations' for t in data['tasks']]
        self.assertLess(len(tasks),6)
