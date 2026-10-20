"""统一关系开发回归：Mock只验证流程，非真实模型质量验收。"""
import copy
import json
import unittest
from unittest.mock import patch
import test_generic_flow as g
import test_source_agent_v4 as f
from agent_relations import FLOW, tasks_for, dependency, validate_row, RelationEngine
from agent_retrieval import Retriever, RetrievalError, compatible, structured
from agent_contracts import verify_source
from agent_data import digest


class FakeEncoder:
    signature='test-encoder-v1'
    dimension=2
    def encode(self,texts): return [[1.0,0.0] for _ in texts]


class UnifiedTests(unittest.TestCase):
    def setUp(self):
        self.base=g.GenericTests(); self.base.setUp(); self.addCleanup(self.base.doCleanups)
        self.h=self.base.h; self.fake=self.base.fake
        self.h.cfg.update(flow_version=FLOW,retrieval=dict(service_config='test',top_k=20),relation_batch_size=16)
        self.h.agent.embedding_client=FakeEncoder()
        self.original=self.fake.ask
        self.answer='compatible'; self.bad_once=False; self.bad_always=False
        self.basis_refs=[]
        def respond(role,prompt,data):
            if role!='source_relations': return self.original(role,prompt,data)
            self.fake.calls.append((role,copy.deepcopy(data)))
            rows=[dict(pair_id=t['pair_id'],relation=self.answer,reason='测试明确关系',basis_refs=copy.deepcopy(self.basis_refs)) for t in data['tasks']]
            if self.bad_once or self.bad_always:
                rows[-1]['relation']='bad'; self.bad_once=False
            return dict(results=rows)
        self.patch=patch.object(self.fake,'ask',side_effect=respond); self.patch.start(); self.addCleanup(self.patch.stop)

    def run_session(self,s,items=None):
        self.fake.items=items
        return self.h.agent.run_session(s)

    def test_exact_repeat_is_zero_relation_calls_and_receipt_idempotent(self):
        first=self.run_session(f.session('s1',text='我喜欢恐龙'))
        second=self.run_session(f.session('s2',text='我喜欢恐龙'))
        self.assertEqual(second['commit']['counts']['reinforce'],1)
        self.assertFalse(any(r=='source_relations' for r,_ in self.fake.calls))
        before=self.h.store.db.execute('SELECT data FROM v3_state').fetchone()[0]
        self.run_session(f.session('s2',text='我喜欢恐龙'))
        self.assertEqual(before,self.h.store.db.execute('SELECT data FROM v3_state').fetchone()[0])
        for p in second['reviews']:
            proof=json.loads(self.h.store.db.execute('SELECT data FROM v3_source_reviews WHERE proof_hash=?',(p['proof_hash'],)).fetchone()[0])
            verify_source(proof,p['candidate'],f.session('s2',text='我喜欢恐龙'))

    def test_combined_relations_and_selective_retry(self):
        s=f.session(text='我喜欢恐龙也喜欢汽车和苹果')
        items=[]
        for name in ('恐龙','汽车','苹果'):
            c=g.raw(s,'喜欢'+name); c['fact']['object']=name; c['fact']['value']=name; items.append(c)
        self.bad_once=True
        r=self.run_session(s,items)
        calls=[d for role,d in self.fake.calls if role=='source_relations']
        self.assertEqual([len(d['tasks']) for d in calls],[3,1])
        self.assertEqual(r['commit']['status'],'committed')
        self.assertFalse(any(role in ('history_compare','source_consistency') for role,_ in self.fake.calls))
        self.assertEqual(len(r['approved']),3)

    def test_same_source_conflict_blocks_without_mutating_proofs(self):
        s=f.session(text='我喜欢恐龙不喜欢汽车')
        a=g.raw(s,'喜欢恐龙'); b=g.raw(s,'不喜欢汽车'); b['fact'].update(object='汽车',polarity='negative')
        self.answer='conflict'
        r=self.run_session(s,[a,b])
        self.assertEqual(r['commit']['status'],'committed')
        self.assertEqual(r['commit']['counts']['pending'],2)
        self.assertTrue(all(p['source_decision']=='pass' for p in r['reviews']))
        self.assertTrue(all('同源' in d['reason'] for d in r['maintenance']))

    def test_historical_conflict_three_members(self):
        self.run_session(f.session('s1',text='我喜欢恐龙'))
        self.answer='conflict'
        for i in (2,3):
            s=f.session('s'+str(i),text='我不喜欢恐龙')
            c=g.raw(s,'不喜欢恐龙'+str(i)); c['fact'].update(value='恐龙'+str(i),polarity='negative')
            r=self.run_session(s,[c]); self.assertEqual(r['commit']['status'],'committed')
        state=json.loads(self.h.store.db.execute('SELECT data FROM v3_state').fetchone()[0])
        self.assertEqual(len(next(iter(state['groups'].values()))['members']),3)

    def test_missing_relation_is_technical_and_resume_only_remaining(self):
        s=f.session(text='我喜欢恐龙和汽车')
        a=g.raw(s); b=g.raw(s,'喜欢汽车'); b['fact']['object']='汽车'
        self.bad_always=True
        r=self.run_session(s,[a,b]); self.assertEqual(r['commit_status'],'not_committed')
        self.bad_always=False
        self.h.agent.session_attempt_budget=None
        r=self.run_session(s,[a,b]); self.assertEqual(r['commit_status'],'committed')

    def test_service_failure_is_not_empty_history(self):
        with patch.object(self.h.agent.embedding_client,'encode',side_effect=RetrievalError('offline')):
            r=self.run_session(f.session())
        self.assertEqual(r['commit_status'],'not_committed')
        self.assertTrue(any(t['code']=='retrieval_unavailable' for t in r['retry_tasks']))

    def test_recall_structural_and_cross_subject_isolation(self):
        a=dict(device_id='u',subject_key='self',attribute_key='',claim='恐龙',fact={'object':'恐龙'},evidence=[],origin_session_id='s')
        self.assertFalse(structured(a,a))
        self.assertFalse(compatible(a,dict(a,subject_key='father')))
        history={str(i):dict(a,memory_id=str(i),revision=1,state='reviewed',attribute_key='preference',fact={'object':str(i)}) for i in range(30)}
        c=dict(a,candidate_id='c',attribute_key='preference',fact={'object':'29'})
        retriever=Retriever(self.h.store.db,dict(top_k=1),{},FakeEncoder())
        from collections import Counter
        retriever.stats=Counter()
        result=retriever.recall([c],dict(memories=history,groups={}))
        self.assertIn('29',[m['memory_id'] for m in result['c']])

    def test_update_without_evidence_rejected(self):
        s=f.session(); task=dict(pair_id='p')
        with self.assertRaises(ValueError):
            validate_row(dict(pair_id='p',relation='left_updates_right',reason='newest',basis_refs=[]),task,dict(evidence_pack=s['turns'],signals=[]),s)

    def test_clear_update_keeps_old_version(self):
        first=f.session('s1',text='我喜欢恐龙')
        self.run_session(first)
        s=f.session('s2',text='我以前喜欢恐龙，现在不喜欢恐龙了')
        self.fake.signals=[f.control(s,s['turns'][0]['user_text'],kind='change',target='item',text='喜欢恐龙',scope='all_time')]
        self.answer='left_updates_right'
        self.basis_refs=[dict(evidence_id=s['turns'][0]['evidence_id'],speaker='user',start=None,end=None)]
        c=g.raw(s,'不喜欢恐龙'); c['fact']['polarity']='negative'
        r=self.run_session(s,[c])
        self.assertEqual(r['commit']['counts']['supersede'],1)
        state=json.loads(self.h.store.db.execute('SELECT data FROM v3_state').fetchone()[0])
        self.assertTrue(any(m['state']=='superseded' for m in state['memories'].values()))
        self.assertTrue(state['versions'])

    def test_nontransitive_equivalence_is_not_merged(self):
        self.run_session(f.session('s1',text='我喜欢恐龙'))
        s=f.session('s2',text='我也喜欢霸王龙')
        c=g.raw(s,'喜欢霸王龙'); c['fact'].update(object='霸王龙',value='霸王龙')
        self.answer='compatible'; self.run_session(s,[c])
        s=f.session('s3',text='我对恐龙感兴趣')
        c=g.raw(s,'对恐龙感兴趣'); c['fact'].update(predicate='感兴趣',value='感兴趣')
        self.answer='equivalent'
        r=self.run_session(s,[c])
        self.assertEqual(r['maintenance'][0]['action'],'pending')
        self.assertIn('互证',r['maintenance'][0]['reason'])

    def test_receipt_recovers_after_result_cache_removed(self):
        s=f.session(); r=self.run_session(s)
        before=len(self.fake.calls)
        with patch.object(self.h.store,'load_work',return_value=None):
            r=self.run_session(s)
        self.assertEqual(before,len(self.fake.calls))
        self.assertEqual(r['commit_status'],'committed')

    def test_tampered_plan_cannot_commit(self):
        from agent_maintenance import V4Store
        from agent_contracts import seal
        original=V4Store.commit
        def corrupt(store,session,candidates,proofs,plan,signals,coverage):
            plan=copy.deepcopy(plan); plan['decisions'][0]['action']='pending'
            plan=seal({k:v for k,v in plan.items() if k!='proof_hash'})
            return original(store,session,candidates,proofs,plan,signals,coverage)
        with patch.object(V4Store,'commit',corrupt):
            with self.assertRaisesRegex(ValueError,'维护动作'):
                self.run_session(f.session())
        self.assertEqual(self.h.store.db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0],0)
        self.assertEqual(self.h.store.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0],0)

    def test_invalid_vector_cache_regenerates(self):
        from collections import Counter
        r=Retriever(self.h.store.db,{},Counter(),FakeEncoder())
        record=dict(device_id='u',subject='我',claim='喜欢恐龙')
        keys,_=r.vectors([record])
        self.h.store.db.execute('UPDATE dense_cache SET data=? WHERE cache_key=?',('[0,0]',keys[0])); self.h.store.db.commit()
        _,values=r.vectors([record])
        self.assertEqual(values,[[1.0,0.0]])
        self.assertEqual(r.stats['embedding_cache_invalidated'],1)

    def test_empty_extraction_needs_no_encoder(self):
        with patch.object(self.h.agent.embedding_client,'encode',side_effect=AssertionError('must not call')):
            r=self.run_session(f.session(),[])
        self.assertEqual(r['commit_status'],'committed')

    def test_old_output_cannot_resume_new_flow(self):
        from agent_flow_data import validate_flow
        with self.assertRaises(ValueError): validate_flow(self.h.cfg,{'flow_version':'generic-flow-v2'})

    def test_relation_evidence_ids_speaker_and_extra_keys(self):
        s=f.session(); task={'pair_id':'p'}; dep=dict(evidence_pack=s['turns'],signals=[])
        for ref in (dict(evidence_id='missing',speaker='user',start=None,end=None),
                    dict(evidence_id=s['turns'][0]['evidence_id'],speaker='assistant',start=None,end=None)):
            with self.assertRaises(ValueError):
                validate_row(dict(pair_id='p',relation='compatible',reason='r',basis_refs=[ref]),task,dep,s)
        with self.assertRaises(ValueError):
            validate_row(dict(pair_id='p',relation='compatible',reason='r',basis_refs=[],extra='x'),task,dep,s)

    def test_relation_cache_survives_support_revision_change(self):
        from agent_generic_flow import GenericPipeline
        s=f.session(); result=self.run_session(s)
        c=result['candidates'][0]
        proof=json.loads(self.h.store.db.execute('SELECT data FROM v3_source_reviews').fetchone()[0])
        m=copy.deepcopy(c); m.update(memory_id='old',revision=1,claim='喜欢汽车'); m['fact']['object']='汽车'
        p=GenericPipeline(self.h.agent,s)
        for revision in (1,2):
            m['revision']=revision
            facts,tasks=tasks_for([c],[proof],{c['candidate_id']:[m]},[])
            rows=RelationEngine(p).run(facts,tasks,[proof],[])
            self.assertEqual(len(rows),1)
        self.assertEqual(sum(role=='source_relations' for role,_ in self.fake.calls),1)
        self.assertEqual(p.stats['source_relations:item_cached'],1)

    def test_both_html_show_source_pass_and_relation_block(self):
        import test_memory_management as management
        from export_memory_html import export_user
        s=f.session(text='我喜欢恐龙不喜欢汽车')
        a=g.raw(s); b=g.raw(s,'不喜欢汽车'); b['fact'].update(object='汽车',polarity='negative')
        self.answer='conflict'; result=self.run_session(s,[a,b])
        case=management.ManagementTests(); case.h=self.h; case.write_report([result])
        before=self.h.store.db.execute('SELECT data FROM v3_state').fetchone()[0]
        export_user(self.h.out,self.h.out/'index.html')
        for name in ('index.html','index_report.html'):
            text=(self.h.out/name).read_text(encoding='utf-8')
            self.assertIn('喜欢恐龙',text); self.assertIn('同源提取分歧',text)
        self.assertEqual(before,self.h.store.db.execute('SELECT data FROM v3_state').fetchone()[0])

    def test_photo_observation_uses_existing_evidence_channel(self):
        import io
        from types import SimpleNamespace
        from test_agent_photos import fixture,pixels,config
        from agent_photos import bind_image
        folder=self.h.out/'photo-input'; folder.mkdir()
        s=fixture(folder)
        with patch('agent_photos.urllib.request.build_opener',return_value=SimpleNamespace(open=lambda *a,**kw:io.BytesIO(pixels()))):
            bind_image(s,self.h.out,config(),self.fake)
        c=g.raw(s,'画面为红色',kind='observation',assertion_mode='observation',subject_role='other',subject='照片画面',scope='仅本图')
        c['fact'].update(type='observation',predicate='画面颜色',object='画面',value='红色')
        r=self.run_session(s,[c])
        self.assertEqual(r['commit_status'],'committed')
        self.assertEqual(r['approved'][0]['subject_key'],s['turns'][0]['evidence_id'])

    def test_single_capacity_error_blocks_without_repeating_request(self):
        from agent_context import ContextCapacityError
        s=f.session(text='我喜欢恐龙和汽车'); a=g.raw(s); b=g.raw(s,'喜欢汽车'); b['fact']['object']='汽车'
        original=self.fake.ask
        count=[]
        def reject(role,prompt,data):
            if role=='source_relations': count.append(role); raise ContextCapacityError('single too large')
            return original(role,prompt,data)
        with patch.object(self.fake,'ask',side_effect=reject):
            r=self.run_session(s,[a,b])
        self.assertEqual(len(count),1)
        self.assertEqual(r['commit_status'],'not_committed')
        self.assertTrue(any(t['code']=='context_capacity' for t in r['retry_tasks']))


if __name__=='__main__': unittest.main()
