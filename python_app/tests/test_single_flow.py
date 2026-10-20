"""单链路契约与恢复测试；合成桩验证程序，不宣称模型语义质量。"""
import copy
import unittest
from unittest.mock import patch

import test_generic_flow as generic
from test_generic_flow import GenericPipeline, raw, f, fact
from agent_flow_data import FLOW
from agent_context import ContextCapacityError
from agent_llm import TransientModelError, JsonClient
from agent_contracts import ModelOutputError


class SingleFlowTests(unittest.TestCase):
    def test_current_config_enables_photos_without_legacy_strategy(self):
        import json
        from pathlib import Path
        from agent_photos import options
        from agent_flow_data import validate_flow
        config=json.loads((Path(__file__).resolve().parents[1]/'config'/'memory_agent_generic_flow.json').read_text(encoding='utf-8'))
        validate_flow(config)
        self.assertNotIn('review_strategy',config)
        self.assertTrue(options(config)['enabled'])

    def setUp(self):
        self.case=generic.GenericTests(); self.case.setUp(); self.addCleanup(self.case.doCleanups)
        self.h=self.case.h; self.fake=self.case.fake

    def test_classification_failure_rule_change_reuses_source(self):
        original=self.fake.ask
        def broken(role,prompt,data):
            if role=='source_classifier': raise TransientModelError('classification offline')
            return original(role,prompt,data)
        with patch.object(self.fake,'ask',side_effect=broken):
            first=self.h.run_session()
        self.assertEqual(first['commit_status'],'not_committed')
        source_calls=sum(role=='source_reviewer' for role,_ in self.fake.calls)
        self.h.registry.dimensions['G16']['rules']+='\n本次分类规则调整'
        self.h.registry.version='changed-product-rules'
        self.h.cfg['source_review_batch_size']=1
        second=self.h.run_session()
        self.assertEqual(second['status'],'completed')
        self.assertEqual(sum(role=='source_reviewer' for role,_ in self.fake.calls),source_calls)
        self.assertEqual(sum(role.startswith('source_extractor:') for role,_ in self.fake.calls),1)

    def test_item_classification_survives_batch_resize(self):
        s=f.session(text='喜欢恐龙和汽车')
        p=GenericPipeline(self.h.agent,s)
        a=p.normalize_candidate(__import__('agent_flow_data').materialize(raw(s),s),'interests_preferences',s,'w')
        b=copy.deepcopy(a); b.update(candidate_id='b',claim='喜欢汽车',fact=fact('喜欢','汽车','汽车'))
        catalog=[dict(dimension_id='G16',name='明确喜欢')]
        result={c['candidate_id']:{'labels':set(),'reasons':[]} for c in (a,b)}
        p.classify_batch([a,b],catalog,result)
        calls=len(self.fake.calls)
        for c in (a,b): p.classify_batch([c],catalog,result)
        self.assertEqual(len(self.fake.calls),calls)

    def test_classification_never_changes_source_proof(self):
        s=f.session(); p=GenericPipeline(self.h.agent,s)
        from agent_flow_data import materialize
        from agent_contracts import verify_source
        c=p.normalize_candidate(materialize(raw(s),s),'interests_preferences',s,'w')
        proof=p.proof(c,f.verdict(c),p.pack([c]),[])
        before=copy.deepcopy(proof)
        p.classify([c],[proof])
        self.assertEqual(proof,before)
        verify_source(proof,c,s)
        c['claim']='changed fact'
        with self.assertRaises(ValueError): verify_source(proof,c,s)

    def test_history_change_does_not_reextract_or_rereview(self):
        self.h.run_session()
        s=f.session('second','我喜欢汽车'); c=raw(s,'喜欢汽车'); c['fact']=fact('喜欢','汽车','汽车')
        self.fake.items=[c]
        original=self.fake.ask
        def broken(role,prompt,data):
            if role=='history_compare': raise TransientModelError('history offline')
            return original(role,prompt,data)
        with patch.object(self.fake,'ask',side_effect=broken): first=self.h.run_session(s)
        self.assertEqual(first['commit_status'],'not_committed')
        counts={role:sum(r==role for r,_ in self.fake.calls) for role in ('source_reviewer','source_extractor:interests_preferences')}
        self.h.run_session(s)
        for role,n in counts.items(): self.assertEqual(sum(r==role for r,_ in self.fake.calls),n)

    def test_relation_cache_survives_batch_change(self):
        from agent_flow_data import materialize
        from agent_data import digest
        s=f.session(text='喜欢恐龙和汽车'); p=GenericPipeline(self.h.agent,s)
        a=p.normalize_candidate(materialize(raw(s),s),'interests_preferences',s,'w')
        b=copy.deepcopy(a); b.update(candidate_id='b',claim='喜欢汽车',fact=fact('喜欢','汽车','汽车'))
        proofs=[p.proof(c,f.verdict(c),p.pack([c]),[]) for c in (a,b)]
        p.reconcile([a,b],proofs); calls=len(self.fake.calls)
        p2=GenericPipeline(self.h.agent,s); self.h.cfg['source_review_batch_size']=1
        p2.reconcile([a,b],copy.deepcopy(proofs))
        self.assertEqual(len(self.fake.calls),calls)
        self.assertEqual(p.consistency_audit,p2.consistency_audit)

    def test_history_item_cache_invalidated_by_revision(self):
        from agent_flow_data import materialize
        self.h.run_session()
        s=f.session('second','我喜欢汽车'); p=GenericPipeline(self.h.agent,s)
        c=raw(s,'喜欢汽车'); c['fact']=fact('喜欢','汽车','汽车')
        c=p.normalize_candidate(materialize(c,s),'interests_preferences',s,'w')
        history=list(self.h.state()['memories'].values())
        data=dict(candidates=[c],history=history,signals=[],group_context=[])
        p.compare_rows(data,{c['candidate_id']:history}); count=len(self.fake.calls)
        p.compare_rows(data,{c['candidate_id']:history})
        self.assertEqual(count,len(self.fake.calls))
        history[0]['revision']+=1
        p.compare_rows(data,{c['candidate_id']:history})
        self.assertGreater(len(self.fake.calls),count)

    def test_network_and_protocol_share_three_calls(self):
        p=GenericPipeline(self.h.agent,f.session())
        seq=[TransientModelError('offline'), {'bad':True}, {'ok':True}]
        def model(*args):
            v=seq.pop(0)
            if isinstance(v,Exception): raise v
            return v
        def check(value):
            if set(value)!={'ok'}: raise ValueError('invalid schema')
        with patch.object(self.fake,'ask',side_effect=model) as call:
            self.assertEqual(p.runtime.ask('test','prompt',{},check),{'ok':True})
            self.assertEqual(call.call_count,3)
        self.assertEqual(next(iter(p.runtime.records.values()))['status'],'completed')

    def test_transport_has_no_hidden_retry(self):
        client=JsonClient.__new__(JsonClient); client.config={'flow_version':FLOW}
        with patch.object(client,'_ask_once',side_effect=TransientModelError('offline')) as call:
            with self.assertRaises(TransientModelError): client.ask('test','prompt',{})
            self.assertEqual(call.call_count,1)

    def test_capacity_does_not_repeat_same_request(self):
        p=GenericPipeline(self.h.agent,f.session())
        with patch.object(self.fake,'ask',side_effect=ContextCapacityError('large')) as call:
            with self.assertRaises(ContextCapacityError): p.runtime.ask('test','prompt',{},lambda r:None)
            self.assertEqual(call.call_count,1)
        key=next(iter(p.runtime.records))
        checkpoint=self.h.store.load_work(key)
        self.assertEqual(checkpoint['status'],'capacity_blocked')
        self.assertEqual(checkpoint['issues'][0]['code'],'context_capacity')

    def test_current_batch_prepare_never_initializes_model(self):
        import json
        from pathlib import Path
        import tempfile
        import memory_agent
        config=Path(__file__).resolve().parents[1]/'config'/'memory_agent_generic_flow.json'
        with tempfile.TemporaryDirectory() as folder, patch('memory_agent.JsonClient',side_effect=AssertionError('no model in prepare')):
            self.assertEqual(memory_agent.run(config,'prepare',folder),0)
            settings=json.loads(config.read_text(encoding='utf-8'))
            device=memory_agent.select_devices(settings)[0]
            expected,_=memory_agent.load_inputs(settings,device)
            manifest=json.loads((Path(folder)/device/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['flow_version'],FLOW)
            self.assertEqual(len(manifest['selected_sessions']),len(expected))

    def test_current_batch_export_failure_never_reports_complete(self):
        import json
        from pathlib import Path
        import tempfile
        import memory_agent
        config=Path(__file__).resolve().parents[1]/'config'/'memory_agent_generic_flow.json'
        device=memory_agent.select_devices(json.loads(config.read_text(encoding='utf-8')))[0]
        row=dict(device_id=device,status='completed',exit_code=0,html=None)
        with tempfile.TemporaryDirectory() as folder, patch('memory_agent.iter_users',return_value=iter([row])), patch('export_memory_html.export_index',side_effect=OSError('disk full')):
            self.assertEqual(memory_agent.run(config,'live',folder),2)
            report=json.loads((Path(folder)/'batch_report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'partial')
            self.assertEqual(report['export_status'],'failed')

    def test_atomic_receipt_recovers_without_model(self):
        first=self.h.run_session()
        self.h.store.db.execute('DELETE FROM context_work'); self.h.store.db.commit()
        with patch.object(self.fake,'ask',side_effect=AssertionError('must not call')):
            second=self.h.run_session()
        self.assertEqual(first['commit'],second['commit'])
        self.assertEqual(len(self.h.state()['memories']),1)

    def test_session_retry_recovers_once(self):
        from memory_agent import execute_session
        original=self.h.agent.run_session
        calls=[]
        def run(s):
            calls.append(s['session_id'])
            if len(calls)==1: raise ValueError('injected gate failure')
            return original(s)
        with patch.object(self.h.agent,'run_session',side_effect=run):
            result=execute_session(self.h.agent,f.session(),self.h.out,self.h.cfg,self.fake)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['session_attempts'],2)
        self.assertEqual(len(result['recovered_errors']),1)

    def test_failed_session_reports_then_next_session_runs(self):
        import json
        from memory_agent import execute_session
        original=self.h.agent.run_session
        calls=[]
        def run(s):
            calls.append(s['session_id'])
            if s['session_id']=='bad': raise ValueError('injected failure')
            return original(s)
        with patch.object(self.h.agent,'run_session',side_effect=run):
            results=[execute_session(self.h.agent,f.session(s),self.h.out,self.h.cfg,self.fake) for s in ('bad','good')]
        self.assertEqual(calls,['bad','bad','good'])
        self.assertEqual(results[0]['status'],'failed')
        self.assertEqual(results[1]['status'],'completed')
        errors=[json.loads(x) for x in (self.h.out/'session_failures.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(errors),1)
        self.assertEqual(len(errors[0]['errors']),2)

    def test_post_commit_exception_recovers_without_duplicate(self):
        from memory_agent import execute_session
        original=self.h.agent.run_session
        calls=[]
        def run(s):
            result=original(s); calls.append(len(self.fake.calls))
            if len(calls)==1: raise RuntimeError('after receipt')
            return result
        with patch.object(self.h.agent,'run_session',side_effect=run):
            result=execute_session(self.h.agent,f.session(),self.h.out,self.h.cfg,self.fake)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(calls[0],calls[1])
        self.assertEqual(len(self.h.state()['memories']),1)

    def test_storage_and_interrupt_are_not_swallowed(self):
        import sqlite3
        from memory_agent import execute_session
        for error in (sqlite3.DatabaseError('corrupt'),OSError('disk full'),KeyboardInterrupt()):
            with patch.object(self.h.agent,'run_session',side_effect=error) as call:
                with self.assertRaises(type(error)):
                    execute_session(self.h.agent,f.session(),self.h.out,self.h.cfg,self.fake)
                self.assertEqual(call.call_count,1)

    def test_photo_binding_error_is_retried_and_reported(self):
        from memory_agent import execute_session
        s=dict(f.session(),source_kind='photo')
        with patch('memory_agent.bind_image',side_effect=ValueError('invalid photo')) as binding, patch.object(self.h.agent,'run_session') as extraction:
            result=execute_session(self.h.agent,s,self.h.out,self.h.cfg,self.fake)
        self.assertEqual(binding.call_count,2)
        extraction.assert_not_called()
        self.assertEqual(result['status'],'failed')
        self.assertTrue((self.h.out/'photo_failures.jsonl').is_file())

    def test_reversed_proof_order_is_not_a_disagreement(self):
        original=GenericPipeline.commit_plan
        def reverse(p,candidates,proofs,plan,signals,coverage):
            return original(p,candidates,list(reversed(proofs)),plan,signals,coverage)
        s=f.session(text='我喜欢恐龙和汽车')
        a=raw(s); b=raw(s,'喜欢汽车'); b['fact']=fact('喜欢','汽车','汽车')
        self.fake.items=[a,b]
        with patch.object(GenericPipeline,'commit_plan',reverse): result=self.h.run_session(s)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(result['approved']),2)

    def test_session_retry_keeps_work_unit_budget(self):
        from memory_agent import execute_session
        from agent_llm import TransientModelError
        def run(s):
            pipeline=GenericPipeline(self.h.agent,s)
            return pipeline.runtime.ask('test','prompt',{},lambda r:None)
        with patch.object(self.h.agent,'run_session',side_effect=run), patch.object(self.fake,'ask',side_effect=TransientModelError('offline')) as model:
            result=execute_session(self.h.agent,f.session(),self.h.out,self.h.cfg,self.fake)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(model.call_count,3)

    def test_runner_continues_and_resume_skips_committed_sessions(self):
        import io,json
        from contextlib import redirect_stdout
        from memory_agent import CentralAgent,run_one
        cfg=dict(self.h.cfg,device_id='test-user'); cfg.pop('user_selection',None)
        sessions=[f.session('bad'),f.session('good')]; out=self.h.out/'isolation-run'
        original=CentralAgent.run_session; attempts=[]
        def run(agent,s):
            attempts.append(s['session_id'])
            if s['session_id']=='bad': raise ValueError('injected commit validation')
            return original(agent,s)
        with patch('memory_agent.JsonClient',return_value=self.fake),patch('memory_agent.load_inputs',return_value=(sessions,{'device_id':'test-user'})),redirect_stdout(io.StringIO()):
            with patch.object(CentralAgent,'run_session',run):
                self.assertEqual(run_one(cfg,'live',str(out)),2)
            report=json.loads((out/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['processed_sessions'],2)
            self.assertEqual(report['committed_sessions'],1)
            self.assertEqual(report['technical_pending_sessions'],1)
            self.assertEqual(report['unvisited_sessions'],0)
            self.assertEqual(attempts,['bad','bad','good'])
            self.assertEqual(run_one(cfg,'live',str(out),resume=True),0)

    def test_explicit_old_config_rejected_before_model(self):
        from memory_agent import CentralAgent
        with self.assertRaises(ValueError):
            CentralAgent(self.fake,self.h.registry,self.h.store,dict(self.h.cfg,flow_version='generic-flow-v1'),self.h.out/'trace.jsonl')
        self.assertEqual(self.fake.calls,[])

    def test_three_attempt_cap_includes_envelope_repair(self):
        calls=[]
        def broken(role,prompt,data):
            if role=='source_router': return dict(skills=['interests_preferences'],signals=[],reason='fixture')
            if role.startswith('source_extractor:'):
                calls.append(data)
                if len(calls)<3: return {'wrong':True}
                c=raw(data['session']); c['evidence'][0]['speaker']='assistant'
                return dict(candidates=[c],notes='')
            raise AssertionError(role)
        with patch.object(self.fake,'ask',side_effect=broken): r=self.h.run_session()
        self.assertEqual(len(calls),3)
        self.assertEqual(r['commit_status'],'not_committed')

    def test_skill_order_does_not_drop_independent_facts(self):
        original=self.fake.ask
        order=['interests_preferences','profile_relations']
        def model(role,prompt,data):
            if role=='source_router': return dict(skills=list(order),signals=[],reason='fixture')
            if role.startswith('source_extractor:'):
                subject='恐龙' if role.endswith('interests_preferences') else '汽车'
                c=raw(data['session'],'喜欢'+subject); c['fact']=fact('喜欢',subject,subject)
                return dict(candidates=[c],notes='')
            return original(role,prompt,data)
        with patch.object(self.fake,'ask',side_effect=model):
            a=self.h.run_session(f.session('order-a','喜欢恐龙和汽车'))
            order.reverse()
            b=self.h.run_session(f.session('order-b','喜欢恐龙和汽车'))
        self.assertEqual({c['claim'] for c in a['candidates']},{c['claim'] for c in b['candidates']})
        self.assertEqual(a['status'],b['status'])

    def test_cross_window_pack_uses_referenced_turn(self):
        from agent_data import digest
        from agent_flow_data import materialize
        s=f.session('cross','你好')
        other=f.session('later','我喜欢恐龙')['turns'][0]
        s['turns'].append(other); s['input_hash']=digest(s['turns'])
        window=dict(s,turns=[other]); p=GenericPipeline(self.h.agent,s)
        c=p.normalize_candidate(materialize(raw(window),window),'interests_preferences',window,'window2')
        proof=p.review([c],[])[0]
        self.assertEqual(proof['decision'],'pass')
        self.assertIn(other['evidence_id'],[t['evidence_id'] for t in proof['evidence_pack']])

    def test_late_control_blocks_early_candidate(self):
        from agent_data import digest
        s=f.session('controls','我喜欢恐龙')
        later=f.session('late','这次聊天不要保存')['turns'][0]
        s['turns'].append(later); s['input_hash']=digest(s['turns'])
        signal=f.control(dict(s,turns=[later]),'这次聊天不要保存')
        self.fake.items=[raw(s)]; self.fake.signals=[signal]
        result=self.h.run_session(s)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['approved'],[])
        self.assertEqual(result['maintenance'][0]['storage_control']['status'],'blocked')

    def test_history_capacity_splits_without_omission(self):
        p=GenericPipeline(self.h.agent,f.session()); seen=[]
        c={'candidate_id':'c'}; page=[{'memory_id':str(i)} for i in range(4)]
        def compare(data,histories):
            if len(data['history'])>1: raise ContextCapacityError('split')
            seen.extend(m['memory_id'] for m in data['history'])
            return {'results':[{'candidate_id':'c','action':'new'}]}
        with patch.object(p,'compare_rows',side_effect=compare):
            result=p.compare_page({'candidates':[c],'history':page},c,page)
        self.assertEqual(seen,['0','1','2','3']); self.assertEqual(len(result),4)

    def test_unknown_usage_not_zero_and_semantic_pending_complete(self):
        from agent_result import summarize_results,summarize_calls
        calls=summarize_calls([{'role':'r','usage':{}},{'role':'r','usage':{'total_tokens':12}}])
        self.assertEqual(calls['r']['usage_missing'],1)
        self.assertEqual(calls['r']['known_tokens'],12)
        statistics=summarize_calls([{'role':'r','attempt':2}], [{'batch_metrics':{'r:item_cached':3}}])
        self.assertEqual(statistics['r']['retries'],1)
        self.assertEqual(statistics['r']['cache_hits'],3)
        row=dict(status='completed',commit={'status':'committed'},reviews=[{'candidate_id':'x','source_decision':'pending'}],maintenance=[])
        result=summarize_results([row],1)
        self.assertEqual(result['technical_pending_sessions'],0)
        self.assertEqual(result['candidate_outcomes'],{'semantic_pending':1})


if __name__=='__main__': unittest.main()
