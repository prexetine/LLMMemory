"""恢复与同源标签分歧开发回归；仅临时SQLite和Mock，无远端/本地模型。"""
import copy
import json
import unittest
import io
from contextlib import redirect_stdout
from unittest.mock import patch

import test_unified_relations as u
import test_source_agent_v4 as f
import test_generic_flow as g
from agent_recovery import resume_report
from agent_retrieval import compatible
from agent_fact_policy import retention
from agent_relations import compatible_cache_key, dependency, cref, mref


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.base = u.UnifiedTests(); self.base.setUp(); self.addCleanup(self.base.doCleanups)
        self.h = self.base.h

    def test_missing_report_recovers_without_model_or_mutation(self):
        s = f.session()
        first = self.base.run_session(s)
        before = self.h.store.db.iterdump()
        snapshot = '\n'.join(before)
        with patch.object(self.base.fake, 'ask', side_effect=AssertionError('no models')):
            report = resume_report(self.h.out, [s, f.session('unprocessed')])
        self.assertEqual(len(report['results']), 1)
        self.assertEqual(first['commit'], report['results'][0]['commit'])
        self.assertEqual(snapshot, '\n'.join(self.h.store.db.iterdump()))

    def test_corrupt_report_recovers(self):
        s = f.session(); self.base.run_session(s)
        (self.h.out/'report.json').write_text('{broken', encoding='utf-8')
        self.assertEqual(len(resume_report(self.h.out, [s])['results']), 1)

    def test_empty_database_is_not_completed(self):
        self.assertEqual(resume_report(self.h.out, [f.session()])['results'], [])

    def test_stale_report_rebuilds_committed_only(self):
        s = f.session(); self.base.run_session(s)
        (self.h.out/'report.json').write_text(json.dumps({'results': [], 'elapsed_seconds': 12}), encoding='utf-8')
        r = resume_report(self.h.out, [s])
        self.assertEqual(r['elapsed_seconds'], 12); self.assertEqual(len(r['results']), 1)

    def test_input_change_and_missing_receipt_fail_closed(self):
        s = f.session(); self.base.run_session(s)
        with self.assertRaises(ValueError):
            resume_report(self.h.out, [dict(s, input_hash='changed')])
        self.h.store.db.execute('DELETE FROM v3_receipts'); self.h.store.db.commit()
        with self.assertRaises(ValueError): resume_report(self.h.out, [s])

    def test_runner_missing_report_skips_committed_session_entirely(self):
        from memory_agent import run_one, CentralAgent
        base=g.GenericTests(); base.setUp(); self.addCleanup(base.doCleanups)
        cfg=dict(base.h.cfg,device_id='test-user'); cfg.pop('user_selection',None)
        s=f.session(text='我喜欢恐龙呀。'); out=base.h.out/'runner'
        with patch('memory_agent.JsonClient',return_value=base.fake), patch('memory_agent.load_inputs',return_value=([s],{'device_id':'test-user'})), redirect_stdout(io.StringIO()):
            self.assertEqual(run_one(cfg,'live',str(out)),0)
            (out/'report.json').unlink()  # 仅本测试临时目录，模拟提交成功但报告缺失。
            with patch.object(CentralAgent,'run_session',side_effect=AssertionError('已提交会话不得进入流水线')):
                self.assertEqual(run_one(cfg,'live',str(out),resume=True),0)
        report=json.loads((out/'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['committed_sessions'],1)
        self.assertEqual(report['technical_pending_sessions'],0)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.base = u.UnifiedTests(); self.base.setUp(); self.addCleanup(self.base.doCleanups)
        self.h = self.base.h

    def test_real_game_same_source_same_claim_merges_after_relation(self):
        s = f.session(text='我会用沙子堆成山再挖隧道')
        a = g.raw(s, s['turns'][0]['user_text'])
        a['fact'].update(predicate='会', object='沙山隧道', value='自述会')
        a['scope'] = 'sandplay_tunnel_building'
        b = copy.deepcopy(a); b['scope'] = '沙玩造隧道'; b['structure']['context'] = 'game'
        self.base.answer = 'equivalent'
        r = self.base.run_session(s, [a, b])
        self.assertEqual(r['commit']['counts']['reinforce'], 1)
        memories = self.h.state()['memories']
        self.assertEqual(len(memories), 1)
        self.assertEqual(next(iter(memories.values()))['support_count'], 1)
        # 再次恢复只读取原回执，不增加支持或调用。
        count = len(self.base.fake.calls)
        self.base.run_session(s, [a, b])
        self.assertEqual(len(self.base.fake.calls), count)

    def test_context_exception_does_not_relax_other_sources_or_story(self):
        a = dict(device_id='u', subject_key='self', origin_session_id='s', claim='会造隧道',
                 evidence=[dict(evidence_id='s:1', quote='我会')], fact={'type': 'attribute'},
                 structure={'context': 'real'}, retention='memory')
        b = dict(a, structure={'context': 'game'})
        self.assertTrue(compatible(a, b))
        self.assertFalse(compatible(a, dict(b, structure={'context': 'story'})))
        self.assertFalse(compatible(a, dict(b, origin_session_id='other')))
        self.assertFalse(compatible(a, dict(b, claim='不会造隧道')))

    def test_equivalent_with_incompatible_context_is_pending_not_duplicate(self):
        s = f.session(); a = g.raw(s); b = copy.deepcopy(a)
        b['structure']['context'] = 'story'
        self.base.answer = 'equivalent'
        r = self.base.run_session(s, [a, b])
        self.assertEqual(r['commit']['counts']['pending'], 2)
        self.assertFalse(self.h.state()['memories'])

    def test_real_event_cannot_be_long_term_despite_wrong_fact_tag(self):
        c = dict(claim='已经游览过公园', retention='memory', fact={'type':'attribute'},
                 structure={'type':'real_event'})
        self.assertEqual(retention(c)[0], 'event')

    def test_bad_step_reports_exact_evidence_id(self):
        from agent_flow_data import materialize, EvidenceReferenceError
        s=f.session(); c=g.raw(s)
        c['structure']['steps']=[dict(kind='context', reference=c['evidence'][0])]
        with self.assertRaises(EvidenceReferenceError) as raised: materialize(c,s)
        self.assertEqual(raised.exception.code,'step_role_mismatch')
        self.assertEqual(raised.exception.evidence_id,s['turns'][0]['evidence_id'])


class CompatibleCacheTests(unittest.TestCase):
    def test_key_preserves_scope_subject_content_and_signal_boundaries(self):
        a = dict(device_id='u', subject_key='self', fact={'type':'attribute'}, scope='all_time',
                 assertion_mode='self_report', retention='memory', claim='喜欢火车', evidence=[])
        b = dict(a, claim='喜欢鸟')
        task = dict(left='c:a', right='m:b:1', purpose=['history'])
        facts = {'c:a': a, 'm:b:1': b}
        key = compatible_cache_key(task, facts, {'signals':[]}, {}, 'model')
        self.assertIsNotNone(key)
        a['evidence'] = [{'evidence_id':'new', 'quote':'喜欢火车'}]
        self.assertEqual(key, compatible_cache_key(task,facts,{'signals':[]},{},'model'))
        for change in ({'scope':'session:s'}, {'fact':{'type':'event'}}, {'state':'conflicted'}, {'valid_from':'2020'}):
            self.assertIsNone(compatible_cache_key(task,dict(facts, **{'c:a':dict(a,**change)}),{'signals':[]},{},'model'))
        self.assertIsNone(compatible_cache_key(task,facts,{'signals':[{'kind':'change'}]},{},'model'))
        self.assertIsNone(compatible_cache_key(dict(task,purpose=['source']),facts,{'signals':[]},{},'model'))

    def test_relation_cache_reuses_unrelated_result_across_sessions(self):
        base=u.UnifiedTests(); base.setUp(); self.addCleanup(base.doCleanups)
        s1=f.session('s1',text='我喜欢恐龙'); a=g.raw(s1); a['scope']='all_time'; base.run_session(s1,[a])
        for sid in ('s2','s3'):
            s=f.session(sid,text='我喜欢汽车'); c=g.raw(s,'喜欢汽车')
            c['fact'].update(object='汽车',value='汽车')
            c['scope']='all_time'
            r=base.run_session(s,[c])
        self.assertEqual(r['batch_metrics'].get('relation_compatible_reused'),1)
        self.assertEqual(len([x for x in base.fake.calls if x[0]=='source_relations']),1)

    def test_history_pack_drops_assistant_only_without_signals(self):
        s=f.session(); s['turns'][0]['assistant_text']='无关助手长文'*1000
        c=dict(candidate_id='c', evidence=[{'evidence_id':s['turns'][0]['evidence_id']}])
        h=dict(memory_id='m',revision=1,evidence=[])
        task=dict(left=cref(c),right=mref(h),purpose=['history'])
        facts={cref(c):c,mref(h):h}
        pack=dependency(task,facts,[],s,[])['evidence_pack']
        self.assertEqual(pack[0]['user_text'],s['turns'][0]['user_text'])
        self.assertEqual(pack[0]['assistant_text'],'')
        signal={'kind':'change','evidence':c['evidence']}
        self.assertTrue(dependency(task,facts,[],s,[signal])['evidence_pack'][0]['assistant_text'])


if __name__ == '__main__':
    unittest.main()
