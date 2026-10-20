"""开发回归：容量/引用故障、代表补边和临时事实；不调用外部模型。"""
import copy
import unittest
from unittest.mock import patch

import test_generic_flow as g
import test_source_agent_v4 as f
import test_unified_relations as u
from agent_generic_flow import GenericPipeline
from agent_flow_data import materialize
from agent_context import ContextCapacityError
from agent_fact_policy import retention
from agent_sparse_relations import select_pairs


class SourceRepairTests(unittest.TestCase):
    def setUp(self):
        self.base = g.GenericTests(); self.base.setUp(); self.addCleanup(self.base.doCleanups)
        self.h, self.fake = self.base.h, self.base.fake

    def exercise(self, fault=None):
        s = f.session(text='我喜欢恐龙和汽车')
        p = GenericPipeline(self.h.agent, s)
        raws = [g.raw(s, '喜欢恐龙'), g.raw(s, '喜欢汽车')]
        raws[1]['fact']['object'] = '汽车'
        cs = [p.normalize_candidate(materialize(x,s), 'interests_preferences', s, 'w') for x in raws]
        originals = copy.deepcopy(cs)
        ps = [p.proof(c, f.verdict(c,'pending','split'), p.pack([c]), []) for c in cs]
        original_proofs = copy.deepcopy(ps)
        calls = []; base = self.fake.ask
        def respond(role, prompt, data):
            if role != 'source_repair':
                return base(role,prompt,data)
            calls.append(copy.deepcopy(data))
            self.assertEqual(data['shared_facts'], [])
            self.assertIn('source_parents_v2', data['repair_mode'])
            if fault == 'capacity' and len(data['parents']) > 1:
                raise ContextCapacityError('injected capacity')
            rows = []
            for parent in data['parents']:
                cid = parent['candidate']['candidate_id']
                index = next(i for i,c in enumerate(originals) if c['candidate_id']==cid)
                raw = copy.deepcopy(raws[index])
                if index == 1 and ((fault == 'span' and len(calls)==1) or fault == 'permanent'):
                    raw['evidence'][0].update(start=0,end=9999)
                rows.append(dict(parent_candidate_id=cid,candidates=[raw],reason='收窄'))
            if fault == 'extra':
                rows.append(dict(parent_candidate_id='foreign',candidates=[],reason='bad'))
            if fault == 'missing' and len(calls)==1:
                rows=rows[:1]
            return dict(results=rows)
        with patch.object(self.fake,'ask',side_effect=respond):
            output, proofs = p.repair_once(cs,ps,[],8)
            count = len(calls)
            second, _ = p.repair_once(copy.deepcopy(originals),copy.deepcopy(original_proofs),[],8)
            self.assertEqual(count,len(calls), '成功项应缓存，失败项不能重置预算')
        return p, output, proofs, calls

    def test_capacity_splits_and_children_reaudited(self):
        p,cs,ps,calls=self.exercise('capacity')
        self.assertEqual([len(x['parents']) for x in calls],[2,1,1])
        self.assertEqual(len(cs),4); self.assertFalse(p.retry_tasks)
        self.assertTrue(all(q['decision']=='pass' for q in ps[2:]))
        self.assertNotEqual(ps[0]['proof_hash'],ps[2]['proof_hash'])

    def test_bad_span_only_retries_bad_parent(self):
        p,cs,_,calls=self.exercise('span')
        self.assertEqual([len(x['parents']) for x in calls],[2,1])
        self.assertIn('span_not_visible',str(calls[1]['validation_feedback']))
        self.assertFalse(p.retry_tasks); self.assertEqual(len(cs),4)

    def test_missing_parent_isolated(self):
        p,cs,_,calls=self.exercise('missing')
        self.assertEqual([len(x['parents']) for x in calls],[2,1])
        self.assertFalse(p.retry_tasks)

    def test_permanent_failure_has_one_task_and_other_success_cached(self):
        p,cs,_,calls=self.exercise('permanent')
        self.assertEqual([len(x['parents']) for x in calls],[2,1,1])
        self.assertEqual(len(p.retry_tasks),1); self.assertEqual(len(cs),3)

    def test_extra_id_fails_closed(self):
        p,cs,_,calls=self.exercise('extra')
        self.assertEqual(len(calls),3); self.assertEqual(len(cs),2)
        self.assertEqual(len(p.retry_tasks),2)


class RepresentativeTests(unittest.TestCase):
    def setUp(self):
        self.base=u.UnifiedTests();self.base.setUp();self.addCleanup(self.base.doCleanups)
        self.h,self.fake=self.base.h,self.base.fake
        self.h.cfg.update(relation_policy='sparse-relations-v2')
        self.h.cfg['retrieval']['batch_top_k']=1

    def run_chain(self, final='equivalent'):
        s=f.session(text='喜欢甲');template=copy.deepcopy(s['turns'][0]);s['turns']=[];items=[]
        for i,name in enumerate(('甲','乙','丙')):
            t=dict(template,evidence_id=f's1:{i+1}',user_text='喜欢'+name,user='喜欢'+name)
            s['turns'].append(t)
            c=g.raw(dict(s,turns=[t]),'喜欢'+name);c['fact'].update(object=name,value=name)
            items.append(c)
        self.fake.items=items
        # 三个点的Top-1只产生A-B、B-C，不人为创建A-C召回。
        with patch.object(self.h.agent.embedding_client,'encode',return_value=[[1.,0.],[.8,.6],[0.,1.]]):
            original=self.fake.ask
            def respond(role,prompt,data):
                if role!='source_relations':return original(role,prompt,data)
                self.fake.calls.append((role,copy.deepcopy(data)))
                return dict(results=[dict(pair_id=t['pair_id'],relation=final if 'representative' in t['purpose'] else 'equivalent',reason='独立核对',basis_refs=[]) for t in data['tasks']])
            with patch.object(self.fake,'ask',side_effect=respond):
                r=self.h.agent.run_session(s)
        return r

    def test_direct_representative_completed_and_verified_at_commit(self):
        r=self.run_chain()
        self.assertEqual(r['commit_status'],'committed')
        self.assertEqual(r['commit']['counts']['reinforce'],2)
        self.assertEqual(r['batch_metrics']['representative_relations_added'],1)
        self.assertFalse(any('缺少与最终代表' in d['reason'] for d in r['maintenance']))

    def test_direct_representative_compatible_not_transitively_merged(self):
        r=self.run_chain('compatible')
        self.assertEqual(r['commit_status'],'committed')
        self.assertEqual(r['commit']['counts']['reinforce'],1)
        self.assertEqual(r['commit']['counts']['coexist'],1)

    def test_failed_supplement_blocks_commit(self):
        r=self.run_chain('invalid')
        self.assertEqual(r['commit_status'],'not_committed')
        self.assertTrue(r['retry_tasks'])

    def test_containment_compatible_does_not_create_source_conflict(self):
        s=f.session(text='先下高速再上高速，途中想躺下休息')
        a=g.raw(s,'先下高速再上高速')
        b=g.raw(s,'先下高速再上高速，途中想躺下休息')
        for c in (a,b):
            c['fact'].update(type='event',value=c['claim'])
            c.update(assertion_mode='temporary',scope='session',retention='event',retention_reason='当次行程')
        self.base.answer='compatible'
        r=self.base.run_session(s,[a,b])
        self.assertEqual(r['commit_status'],'committed')
        self.assertEqual(r['commit']['counts']['coexist'],1)
        self.assertFalse(any('同源提取分歧' in d['reason'] for d in r['maintenance']))


class RetentionTests(unittest.TestCase):
    def test_scoped_attribute_is_event_and_request_not_memory(self):
        c=dict(claim='用户现在在二楼',fact={'type':'attribute'},scope='session:s',retention='memory')
        self.assertEqual(retention(c)[0],'event')
        self.assertEqual(retention(dict(c,retention='session_record'))[0],'session_record')
        c.update(claim='用户必须要吃糖',scope='本次活动',fact={'type':'attribute','predicate':'有需求'})
        self.assertEqual(retention(c)[0],'session_record')
        c.update(claim='用户喜欢唱歌',scope='all_time',fact={'type':'attribute','predicate':'喜欢'})
        self.assertEqual(retention(c)[0],'memory')

    def test_prompt_contract_includes_containment_not_conflict(self):
        from agent_relations import PROMPT
        from agent_prompts import EXTRACT,REVIEW,CLASSIFY
        self.assertIn('包含/细化不等于互斥',PROMPT)
        self.assertIn('不是equivalent或conflict',PROMPT)
        self.assertTrue(all('当次索要' in p and '单次明确' in p for p in (EXTRACT,REVIEW,CLASSIFY)))


if __name__=='__main__':
    unittest.main()
