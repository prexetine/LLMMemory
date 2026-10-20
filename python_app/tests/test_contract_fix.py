"""证据契约修复开发回归；模型桩不等同真实模型质量验收。"""
import copy
import json
import unittest
from unittest.mock import patch

import test_generic_flow as g
from agent_generic_flow import GenericPipeline
from agent_flow_data import materialize
from agent_contracts import verify_source, ModelOutputError, explicit_subject_switch


class ContractFixTests(unittest.TestCase):
    def setUp(self):
        self.case=g.GenericTests(); self.case.setUp(); self.addCleanup(self.case.doCleanups)
        self.h=self.case.h; self.fake=self.case.fake

    def candidate(self,s):
        p=GenericPipeline(self.h.agent,s)
        return p,p.normalize_candidate(materialize(g.raw(s),s),'interests_preferences',s,'w')

    def test_normalized_pending_is_cached_not_model_pass(self):
        s=g.f.session(); p,c=self.candidate(s)
        c.update(subject_role='unresolved',subject_key=None)
        proof=p.review([c],[])[0]
        self.assertEqual(proof['decision'],'pending')
        verify_source(proof,c,s)
        count=len(self.fake.calls)
        self.assertEqual(p.review([c],[])[0]['decision'],'pending')
        self.assertEqual(len(self.fake.calls),count)

    def test_invalid_source_cache_is_bypassed(self):
        s=g.f.session(); p,c=self.candidate(s)
        p.review([c],[])
        for key,data in self.h.store.db.execute('SELECT work_key,data FROM context_work').fetchall():
            value=json.loads(data)
            if 'proof' in value:
                value['proof']['proof_hash']='invalid'
                self.h.store.save_work(key,s,value)
        proof=p.review([c],[])[0]
        verify_source(proof,c,s)
        self.assertEqual(len(p.runtime.invalidations),1)
        p.review([c],[])
        self.assertEqual(len(p.runtime.invalidations),1)

    def test_history_basis_uses_visible_original(self):
        s=g.f.session(); p,c=self.candidate(s)
        response=dict(results=[dict(candidate_id=c['candidate_id'],action='new',targets=[],reason='fixture',related_memory_ids=[],
            basis_refs=[dict(evidence_id=s['turns'][0]['evidence_id'],speaker='user',start=None,end=None)])])
        with patch.object(self.fake,'ask',return_value=response):
            result=p.ask('history_compare','',dict(candidates=[c],history=[],signals=[]),lambda r:None)
        self.assertEqual(result['results'][0]['basis'][0]['quote'],s['turns'][0]['user_text'])
        self.assertNotIn('basis_refs',result['results'][0])

    def test_history_assistant_or_missing_reference_cannot_pass(self):
        s=g.f.session(); p,c=self.candidate(s)
        for ref in (dict(evidence_id=s['turns'][0]['evidence_id'],speaker='assistant',start=None,end=None),
                    dict(evidence_id='missing',speaker='user',start=None,end=None)):
            response=dict(results=[dict(candidate_id=c['candidate_id'],action='new',targets=[],reason='fixture',related_memory_ids=[],basis_refs=[ref])])
            with patch.object(self.fake,'ask',return_value=response):
                with self.assertRaises(ModelOutputError):
                    p.ask('history_compare','',dict(candidates=[c],history=[],signals=[],test_reference=ref),lambda r:None)

    def test_invalid_stage_cache_is_revalidated(self):
        s=g.f.session(); p,c=self.candidate(s)
        def check(r):
            if r!={'ok':True}: raise ValueError('bad schema')
        with patch.object(self.fake,'ask',return_value={'ok':True}) as model:
            p.runtime.ask('test','',{},check)
            key=next(iter(p.runtime.records))
            self.h.store.save_work(key,s,dict(status='completed',output={'bad':True}))
            p.runtime.ask('test','',{},check)
        self.assertEqual(model.call_count,2)
        self.assertEqual(len(p.runtime.invalidations),1)

    def test_third_person_is_not_speaker_switch(self):
        for text in ('那你给我一个讲一个漂亮的小故事，他没来。','故事里的爸爸来了','我是小明'):
            signal=dict(switch_basis='handoff',switch_evidence=dict(quote=text),target_text='他')
            self.assertFalse(explicit_subject_switch(signal))
        self.assertTrue(explicit_subject_switch(dict(switch_basis='handoff',switch_evidence=dict(quote='现在换爸爸来说'),target_text='爸爸')))
        self.assertTrue(explicit_subject_switch(dict(switch_basis='self_identification',switch_evidence=dict(quote='现在我是小明'),target_text='小明')))

    def test_real_unrelated_pronoun_does_not_block_request(self):
        s=g.f.session(text='那你给我一个讲一个漂亮的小故事，他没来。')
        raw=g.raw(s,'请求讲一个漂亮的小故事',assertion_mode='request')
        raw['fact']=g.fact(kind='request',predicate='请求',obj='故事',value='讲故事')
        raw['retention']='session_record'; raw['retention_reason']='当次请求'
        self.fake.items=[raw]
        self.fake.signals=[dict(kind='subject_switch',subject_key=None,attribute_key=None,scope='session',target_type='unresolved',target_text='他',
            evidence=[dict(evidence_id=s['turns'][0]['evidence_id'],quote=s['turns'][0]['user_text'])],switch_basis='handoff',
            switch_reference=dict(evidence_id=s['turns'][0]['evidence_id'],speaker='user',start=None,end=None))]
        result=self.h.run_session(s)
        self.assertEqual(result['commit']['status'],'committed')
        self.assertEqual(result['signals'],[])
        self.assertEqual(result['approved'][0]['subject_role'],'current_user')
        self.assertEqual(result['approved'][0]['retention'],'session_record')

    def test_history_requires_real_targets_and_change_basis(self):
        from agent_contracts import validate_comparison
        s=g.f.session(); _,c=self.candidate(s)
        for action in ('reinforce','coexist','conflict','supersede'):
            row=dict(candidate_id=c['candidate_id'],action=action,targets=[],basis=[],reason='fixture')
            with self.assertRaises(ValueError): validate_comparison(dict(results=[row]),[c],[],s)
        row.update(targets=[dict(memory_id='old',revision=1)])
        with self.assertRaisesRegex(ValueError,'更新必须'):
            validate_comparison(dict(results=[row]),[c],[dict(memory_id='old',revision=1)],s)

    def test_explicit_second_speaker_is_isolated(self):
        s=g.f.session(text='现在换爸爸来说，我喜欢汽车。')
        self.fake.items=[g.raw(s,'喜欢汽车')]
        self.fake.signals=[dict(kind='subject_switch',subject_key='爸爸',attribute_key=None,scope='session',target_type='item',target_text='爸爸',
            evidence=[dict(evidence_id=s['turns'][0]['evidence_id'],quote='现在换爸爸来说')],switch_basis='handoff',
            switch_reference=dict(evidence_id=s['turns'][0]['evidence_id'],speaker='user',start=0,end=7))]
        result=self.h.run_session(s)
        self.assertEqual(result['approved'][0]['subject_role'],'other')
        self.assertNotEqual(result['approved'][0]['subject_key'],'account:'+s['device_id']+':self')

    def test_time_must_not_be_assistant_or_photo_timestamp(self):
        s=g.f.session(text='今天去了公园'); s['turns'][0]['assistant_text']='2026年9月7日'
        raw=g.raw(s)
        raw['fact']['time_reference']=dict(evidence_id=s['turns'][0]['evidence_id'],speaker='assistant',start=0,end=9)
        with self.assertRaises(ValueError): materialize(raw,s)
        raw['fact']['time_reference'].update(speaker='user',start=0,end=2)
        with self.assertRaises(ValueError): materialize(raw,dict(s,source_kind='photo'))

    def test_empty_fact_is_protocol_failure_not_silent_success(self):
        s=g.f.session(); p=GenericPipeline(self.h.agent,s); raw=g.raw(s)
        raw['fact']['value']=''
        with self.assertRaisesRegex(ValueError,'事实值'):
            p.normalize_candidate(materialize(raw,s),'interests_preferences',s,'w')

    def test_final_diagnostic_uses_final_exception(self):
        from agent_generic_flow import failure_details
        from agent_flow_data import EvidenceReferenceError
        try:
            try: raise EvidenceReferenceError('wrong_speaker','e1','final role failure')
            except ValueError as error: raise ModelOutputError(str(error)) from error
        except ModelOutputError as error:
            detail=failure_details(error)
        self.assertEqual(detail['code'],'wrong_speaker');self.assertIn('final role failure',detail['reason']);self.assertEqual(detail['evidence_ids'],['e1'])

    def test_time_only_from_located_user_expression(self):
        for text, phrase, expected in [('我在2026年9月7日去了水上乐园','2026年9月7日','2026-09-07'),('昨天去了水上乐园','昨天',None)]:
            s=g.f.session(text=text); p=GenericPipeline(self.h.agent,s); raw=g.raw(s,'去了水上乐园')
            raw['fact']=g.fact(kind='event',predicate='去',obj='水上乐园')
            start=text.index(phrase)
            raw['fact']['time_reference']=dict(evidence_id=s['turns'][0]['evidence_id'],speaker='user',start=start,end=start+len(phrase))
            candidate=p.normalize_candidate(materialize(raw,s),'events_plans',s,'w')
            self.assertEqual(candidate['fact']['time_text'],phrase)
            self.assertEqual(candidate['fact']['time_value'],expected)
        s=g.f.session(); _,candidate=self.candidate(s)
        self.assertEqual(candidate['fact']['time_text'],'')
        self.assertIsNone(candidate['fact']['time_value'])
        raw=g.raw(s); raw['valid_from']='2026-09-07'
        with self.assertRaises(ValueError): materialize(raw,s)

    def test_assistant_context_not_fact_evidence(self):
        s=g.f.session(); s['turns'][0]['assistant_text']='从前有一只小恐龙。'
        raw=g.raw(s); raw['structure']['steps']=[dict(kind='context',reference=dict(evidence_id=s['turns'][0]['evidence_id'],speaker='assistant',start=None,end=None))]
        c=materialize(raw,s)
        self.assertEqual(c['structure']['steps'][0]['speaker'],'assistant')
        self.assertEqual(c['evidence'][0]['quote'],s['turns'][0]['user_text'])
        raw['structure']['steps'][0]['kind']='event'
        with self.assertRaises(ValueError): materialize(raw,s)

    def test_item_cache_validation(self):
        s=g.f.session(); p,_=self.candidate(s)
        def validate(row):
            if row!={'ok':True}: raise ValueError('wrong cached item')
        key,_=p.runtime.load_item('source_classifier','',{},validate)
        p.runtime.save_item(key,'source_classifier',{},dict(ok=False))
        self.assertIsNone(p.runtime.load_item('source_classifier','',{},validate)[1])
        self.assertEqual(len(p.runtime.invalidations),1)
        p.runtime.save_item(key,'source_classifier',{},dict(ok=True))
        self.assertEqual(p.runtime.load_item('source_classifier','',{},validate)[1],{'ok':True})

    def test_mixed_versions_do_not_relabel_old_commits(self):
        from agent_result import summarize_results
        from agent_flow_data import PROCESSING_REVISION
        first=self.h.run_session()
        self.assertEqual(first['processing_revision'],PROCESSING_REVISION)
        old=copy.deepcopy(first);old.pop('processing_revision')
        summary=summarize_results([old,first],2)
        self.assertTrue(summary['mixed_versions'])
        self.assertEqual(summary['processing_versions']['pre-contract-fix'],1)


if __name__=='__main__':
    unittest.main()
