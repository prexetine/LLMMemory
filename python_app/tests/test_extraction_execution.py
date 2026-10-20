"""预算、失败Skill隔离及逐项批修复；无网络调用。"""
import copy
import unittest
from unittest.mock import patch
import test_generic_flow as g
import test_source_agent_v4 as f
from agent_generic_flow import GenericPipeline
from agent_context import ContextCapacityError, request_metrics, check_request


class ExtractionExecutionTests(unittest.TestCase):
    def setUp(self):
        self.base=g.GenericTests();self.base.setUp();self.addCleanup(self.base.doCleanups)
        self.h=self.base.h;self.fake=self.base.fake
        self.h.cfg.update(extraction_execution='isolated-skill-batch-repair-v1',token_estimator='mixed_chars_v1')

    def test_budget_units_and_guard(self):
        cfg=dict(context_window_tokens=65536,token_estimator='mixed_chars_v1')
        m=request_metrics(cfg,'source_router','中文'*1000,dict(session='abc'*1000))
        self.assertLess(m['estimated_input_tokens'],m['request_utf8_bytes'])
        self.assertEqual(check_request(cfg,'source_router','中文'*1000,dict(session='abc'*1000)),m['estimated_input_tokens'])
        with self.assertRaises(ContextCapacityError):check_request(cfg,'source_router','中'*100000,{})

    def test_repair_context_preserves_steps_time_and_unknown_refs(self):
        from agent_extraction_repair import repair_window
        s=f.session();s['turns']=[dict(s['turns'][0],evidence_id=str(i)) for i in range(10)]
        raw=dict(evidence=[dict(evidence_id='1')],structure=dict(steps=[dict(evidence_id='7')]))
        pack=repair_window(s,[raw])
        self.assertEqual([t['evidence_id'] for t in pack['turns']],['0','1','2','6','7','8'])
        raw['fact']=dict(time_reference=dict(evidence_id='9'))
        self.assertIn('9',[t['evidence_id'] for t in repair_window(s,[raw])['turns']])
        raw['fact']['time_reference']['evidence_id']='unknown'
        self.assertEqual(repair_window(s,[raw]),s)

    def test_skill_capacity_does_not_reroute_or_repeat_success(self):
        s=f.session();s['turns'].append(dict(s['turns'][0],evidence_id='second'))
        p=GenericPipeline(self.h.agent,s);calls=[]
        def router(*args,**kwargs):return dict(skills=['interests_preferences','events_plans'],signals=[],reason='test')
        def extract(w,wid,skill,rules,signals):
            calls.append((skill,len(w['turns'])))
            if skill=='events_plans' and len(w['turns'])>1:raise ContextCapacityError('injected output length')
            return []
        with patch.object(p,'ask',side_effect=router) as route,patch.object(p,'extract_page',side_effect=extract):p.extract_window(s)
        self.assertEqual(route.call_count,1)
        self.assertEqual(calls,[('interests_preferences',2),('events_plans',2),('events_plans',1),('events_plans',1)])

    def run_repair(self,invalid_second=False,fault=None):
        s=f.session(text='我喜欢恐龙和汽车');p=GenericPipeline(self.h.agent,s)
        a=g.raw(s,'喜欢恐龙');b=g.raw(s,'喜欢汽车');b['fact']['object']='汽车'
        originals=[a,b];bad=copy.deepcopy(originals)
        for c in bad:c['subject']=None
        calls=[];original=self.fake.ask
        def respond(role,prompt,data):
            if role.startswith('source_extractor:'):return dict(candidates=copy.deepcopy(bad),notes='')
            if role=='source_repair':
                calls.append(copy.deepcopy(data));rows=[]
                for i,parent in enumerate(data['parents']):
                    raw=next(x for x in originals if x['claim']==parent['candidate']['claim'])
                    value=copy.deepcopy(raw)
                    if invalid_second and len(calls)==1 and i==1:value['subject']=None
                    rows.append(dict(parent_candidate_id=parent['candidate']['candidate_id'],candidates=[value],reason='修复类型'))
                if fault=='extra':rows.append(dict(parent_candidate_id='unknown',candidates=[],reason='bad'))
                if fault=='missing':rows=rows[:1]
                if fault=='duplicate':rows.append(copy.deepcopy(rows[0]))
                if fault=='capacity' and len(rows)>1:raise ContextCapacityError('split repair only')
                return dict(results=rows)
            return original(role,prompt,data)
        with patch.object(self.fake,'ask',side_effect=respond):
            result=p.extract_page(s,'w','interests_preferences',[],[])
            again=p.extract_page(s,'w','interests_preferences',[],[])
        return p,result,again,calls

    def test_bad_rows_batch_and_cache(self):
        p,result,again,calls=self.run_repair()
        self.assertEqual(len(result),2);self.assertEqual(result,again)
        self.assertEqual([len(c['parents']) for c in calls],[2])
        self.assertFalse(p.extraction_issues)

    def test_partial_success_only_retries_failed_row(self):
        p,result,again,calls=self.run_repair(True)
        self.assertEqual(len(result),2);self.assertEqual(result,again)
        self.assertEqual([len(c['parents']) for c in calls],[2,1])

    def test_missing_result_retries_only_missing_parent(self):
        p,result,again,calls=self.run_repair(fault='missing')
        self.assertEqual(len(result),2)
        self.assertEqual([len(c['parents']) for c in calls],[2,1])

    def test_extra_ids_fail_closed_and_retry_is_bounded(self):
        p,result,again,calls=self.run_repair(fault='extra')
        self.assertEqual(result,[]);self.assertEqual(again,[])
        self.assertEqual(len(calls),2);self.assertTrue(p.extraction_issues)

    def test_duplicate_id_not_accepted(self):
        p,result,again,calls=self.run_repair(fault='duplicate')
        self.assertEqual(len(result),1);self.assertTrue(p.extraction_issues)

    def test_capacity_splits_only_repair_group(self):
        p,result,again,calls=self.run_repair(fault='capacity')
        self.assertEqual(len(result),2);self.assertFalse(p.extraction_issues)
        self.assertEqual([len(c['parents']) for c in calls],[2,1,1])

    def test_terminal_skill_capacity_keeps_next_skill(self):
        s=f.session();p=GenericPipeline(self.h.agent,s);self.h.cfg['context_split_retries']=1
        with patch.object(p,'extract_page',side_effect=ContextCapacityError('too large')):
            self.assertEqual(p.extract_skill_window(s,'events_plans',[]),[])
        self.assertTrue(p.retry_tasks)
