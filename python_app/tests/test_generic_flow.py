"""跨用户/跨内容的通用流程回归；全部Mock，禁止外部模型调用。"""
import copy
import json
import unittest
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
import test_source_agent_v4 as f
from test_chain_facts import fact as stored_fact
from test_memory_management import structure
from agent_flow_data import FLOW, materialize, compact_fact, source_pairs, verify_consistency
from agent_flow_data import validate_flow, EvidenceReferenceError
from agent_generic_flow import GenericPipeline
from agent_llm import TransientModelError
from agent_contracts import verify_source
from agent_schema import response_schema
from export_memory_html import export_user
from memory_agent import CentralAgent, run_one
import test_memory_management as management


def fact(*args, **kwargs):
    value=stored_fact(*args,**kwargs)
    value.pop('time_text'); value.pop('time_value')
    value['time_reference']=None
    return value


def raw(s, claim='喜欢恐龙', **kw):
    c=f.raw(s,claim,dimension_id='UNCLASSIFIED',fact=fact(),structure=structure(topic=''),**kw)
    c['evidence']=[dict(evidence_id=s['turns'][0]['evidence_id'],speaker='user',start=None,end=None)]
    return c


class GenericFake(f.Fake):
    def __init__(self):
        super().__init__(); self.labels=['G16']; self.relation='distinct'; self.fail_review=False
    def ask(self,role,prompt,data):
        if role.startswith('source_extractor:'):
            self.calls.append((role,copy.deepcopy(data)))
            return dict(candidates=copy.deepcopy(self.items if self.items is not None else [raw(data['session'])]),notes='')
        if role=='source_classifier':
            self.calls.append((role,copy.deepcopy(data)))
            return dict(results=[dict(candidate_id=c['candidate_id'],dimension_ids=[d for d in self.labels if d in {x['dimension_id'] for x in data['catalog']}],reason='分类不改变事实') for c in data['candidates']])
        if role=='source_consistency':
            self.calls.append((role,copy.deepcopy(data)))
            return dict(results=[dict(pair_id=p['pair_id'],relation=self.relation,reason='按来源含义比较') for p in data['pairs']])
        if role=='source_reviewer' and self.fail_review:
            raise TransientModelError('offline timeout')
        result=super().ask(role,prompt,data)
        def references(items):
            turns=data.get('session',{}).get('turns',data.get('evidence_pack',[]))
            refs=[]
            for item in items:
                text=next(t['user_text'] for t in turns if t['evidence_id']==item['evidence_id'])
                start=text.index(item['quote'])
                refs.append(dict(evidence_id=item['evidence_id'],speaker='user',start=start,end=start+len(item['quote'])))
            return refs
        if role=='history_compare':
            for row in result['results']: row['basis_refs']=references(row.pop('basis'))
        if role=='source_router':
            for signal in result['signals']:
                signal['evidence']=references(signal['evidence'])
                signal.setdefault('switch_basis','none'); signal.setdefault('switch_reference',None)
        return result


class GenericTests(unittest.TestCase):
    def setUp(self):
        self.h=f.V4Tests(); self.h.setUp(); self.addCleanup(self.h.doCleanups)
        self.h.cfg['flow_version']=FLOW
        for k in ('review_strategy','max_agent_steps','max_audit_rounds','history_limit','network_retries','model_output_attempts'):
            self.h.cfg.pop(k,None)
        self.fake=GenericFake()
        self.h.agent=CentralAgent(self.fake,self.h.registry,self.h.store,self.h.cfg,self.h.out/'trace.jsonl')

    def test_flow_boundary_is_explicit(self):
        with self.assertRaises(ValueError): validate_flow({})
        self.assertEqual(validate_flow(self.h.cfg, {'flow_version':FLOW}), FLOW)
        for cfg in ({'flow_version':'typo'}, {'flow_version':'generic-flow-v1'},
                    {'flow_version':FLOW,'review_strategy':'source_v3'}):
            with self.assertRaises(ValueError): validate_flow(cfg)
        with self.assertRaises(ValueError): validate_flow(self.h.cfg, {})
        with self.assertRaises(ValueError): validate_flow({}, {'flow_version':FLOW})

    def test_joint_contract_is_rejected_before_inputs(self):
        with patch('memory_agent.load_inputs') as load:
            with self.assertRaisesRegex(ValueError, '联合提取配置'):
                run_one(dict(self.h.cfg, extraction_contract='compact-v1'), 'live', str(self.h.out/'disabled'))
            load.assert_not_called()

    def test_router_dispatches_each_selected_skill_independently(self):
        selected = ['interests_preferences', 'events_plans', 'product_interaction']
        original = self.fake.ask
        def respond(role, prompt, data):
            if role == 'source_router':
                self.fake.calls.append((role, copy.deepcopy(data)))
                return dict(skills=selected, signals=[], reason='独立Skill回归')
            return original(role, prompt, data)
        self.fake.items = []
        with patch.object(self.fake, 'ask', side_effect=respond):
            candidates, signals = GenericPipeline(self.h.agent, f.session()).extract_window(f.session())
        roles = [role for role, _ in self.fake.calls]
        self.assertEqual(roles.count('source_router'), 1)
        self.assertEqual([r for r in roles if r.startswith('source_extractor:')],
                         ['source_extractor:' + skill for skill in selected])
        self.assertEqual(candidates, [])
        self.assertEqual(signals, [])

    def test_invalid_flow_rejected_before_loading_inputs(self):
        from memory_agent import run_one
        with patch('memory_agent.load_inputs') as load:
            with self.assertRaisesRegex(ValueError, '旧控制参数'):
                run_one(dict(self.h.cfg,review_strategy='source_v3'), 'live', str(self.h.out/'invalid-flow'))
            load.assert_not_called()

    def test_reference_failures_have_distinct_reasons(self):
        s=f.session(text='我喜欢恐龙')
        for change, code in [({'evidence_id':'missing'},'id_not_in_context'),
                             ({'speaker':'assistant'},'wrong_speaker'),
                             ({'start':0,'end':1000},'span_not_visible'),
                             ({'start':True,'end':2},'invalid_span')]:
            c=raw(s); c['evidence'][0].update(change)
            with self.assertRaises(EvidenceReferenceError) as ctx: materialize(c,s)
            self.assertEqual(ctx.exception.code,code)
            self.assertIn(c['evidence'][0]['evidence_id'],str(ctx.exception))

    def test_split_reference_not_silently_replaced(self):
        s=f.session(text='恐龙')
        s['turns'][0]['source_spans']={'user_text':[3,5]}
        other=copy.deepcopy(s['turns'][0]); other.update(user_text='汽车',source_spans={'user_text':[8,10]})
        s['turns'].append(other)
        c=raw(s)
        with self.assertRaises(EvidenceReferenceError) as ctx: materialize(c,s)
        self.assertEqual(ctx.exception.code,'ambiguous_fragment')
        c['evidence'][0].update(start=8,end=10)
        self.assertEqual(materialize(c,s)['evidence'][0]['quote'],'汽车')

    def test_reference_roles_ranges_and_extra_keys(self):
        s=f.session(text='我喜欢恐龙，也喜欢汽车')
        c=raw(s); c['evidence'][0].update(start=3,end=5)
        self.assertEqual(materialize(c,s)['evidence'][0]['quote'],'恐龙')
        for change in ({'speaker':'assistant'},{'start':-1,'end':2},{'start':0,'end':999},{'extra':True}):
            bad=copy.deepcopy(c); bad['evidence'][0].update(change)
            with self.assertRaises(ValueError): materialize(bad,s)

    def test_unclassified_content_commits_and_exports(self):
        self.fake.labels=[]
        r=self.h.run_session()
        self.assertEqual(r['commit']['counts']['new'],1)
        self.assertEqual(r['approved'][0]['dimension_id'],'UNCLASSIFIED')
        self.assertEqual(self.h.run_session()['commit'],r['commit'])
        case=management.ManagementTests(); case.h=self.h; case.write_report([r])
        self.h.store.export_book(self.h.out/'book.md',self.h.registry)
        self.assertIn('喜欢恐龙',(self.h.out/'book.md').read_text(encoding='utf-8'))
        export_user(self.h.out,self.h.out/'index.html')
        for name in ('index.html','index_report.html'):
            self.assertIn('喜欢恐龙',(self.h.out/name).read_text(encoding='utf-8'))

    def test_reference_pipeline_and_labels_do_not_change_source(self):
        r=self.h.run_session(); c=r['candidates'][0]; p=r['reviews'][0]
        self.assertEqual(c['dimension_ids'],['G16'])
        self.assertEqual(c['evidence'][0]['quote'],'我喜欢恐龙呀。')
        bad=copy.deepcopy(c); bad['claim']='喜欢火箭'
        with self.assertRaises(ValueError): verify_source(p,bad,f.session(text='我喜欢恐龙呀。'))
        calls=[x for x in self.fake.calls if x[0].startswith('source_extractor:')]
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1]['skill_rules'],[])

    def test_failed_review_resume_does_not_repeat_extraction(self):
        self.fake.fail_review=True
        r=self.h.run_session()
        self.assertEqual(r['status'],'partial')
        self.assertFalse(self.h.state()['memories'])
        before=len([x for x in self.fake.calls if x[0].startswith('source_extractor:')])
        self.fake.fail_review=False
        r=self.h.run_session()
        self.assertEqual(r['status'],'completed')
        self.assertEqual(len([x for x in self.fake.calls if x[0].startswith('source_extractor:')]),before)

    def test_same_evidence_independent_facts_not_blocked(self):
        s=f.session(text='我喜欢恐龙也喜欢汽车')
        a=raw(s); b=raw(s,'喜欢汽车'); b['fact']=fact('喜欢','汽车','汽车')
        self.fake.items=[a,b]
        r=self.h.run_session(s)
        self.assertEqual(len(r['approved']),2)
        self.assertEqual(len(r['consistency_audit']),1)

    def test_incompatible_source_is_pending_not_label_escape(self):
        s=f.session(text='今天可以一起玩')
        a=raw(s,'可以一起玩'); b=raw(s,'希望每天一起玩'); b['fact']=fact('希望','玩','每天')
        self.fake.items=[a,b]; self.fake.relation='incompatible'
        r=self.h.run_session(s)
        self.assertEqual(r['approved'],[])
        self.assertTrue(all(p['source_decision']=='pending' for p in r['reviews']))

    def test_compact_history_keeps_meaning_not_long_steps(self):
        c=dict(fact=fact(),claim='具体结论',subject_key='p',scope='学校',structure=dict(context='story',steps=[dict(text='长故事'*20000)]),evidence=[dict(evidence_id='e',quote='长原文'*20000)])
        projected=compact_fact(c)
        self.assertLess(len(json.dumps(projected)),1000)
        self.assertEqual(projected['fact'],c['fact'])
        self.assertEqual(projected['context'],'story')
        self.assertEqual(projected['scope'],'学校')

    def test_schema_references_no_quote_and_strict(self):
        s=response_schema('source_extractor:interests_preferences',dict(policy='source-maintenance-v4',flow_version=FLOW,skill_rules=[]))
        p=s['properties']['candidates']['items']['properties']['evidence']['items']
        self.assertNotIn('quote',p['properties'])
        self.assertFalse(p['additionalProperties'])

    def test_users_do_not_share_source_pairs(self):
        a=dict(device_id='a',subject_key='same',origin_session_id='s',evidence=[dict(evidence_id='e')])
        self.assertEqual(source_pairs([a,dict(a,device_id='b')]),[])

    def test_complete_photo_chain_unclassified_not_personal_trait(self):
        from test_agent_photos import fixture,pixels,config
        from agent_photos import bind_image
        root=self.h.out/'photos';root.mkdir(); s=fixture(root)
        with patch('agent_photos.urllib.request.build_opener',return_value=SimpleNamespace(open=lambda *a,**kw:io.BytesIO(pixels()))):
            bind_image(s,root,config(),self.fake)
        c=raw(s,'画面有红色形状',kind='observation',assertion_mode='observation',subject_role='other',subject='照片画面')
        c['fact']=fact('可见','形状','红色','observation');c['structure']=structure('visual','photo','形状')
        self.fake.items=[c];self.fake.labels=[]
        r=self.h.run_session(s)
        self.assertEqual(r['approved'][0]['retention'],'event')
        self.assertEqual(r['approved'][0]['subject_key'],s['turns'][0]['evidence_id'])

    def test_runner_resume_after_capacity_change_preserves_source_cache(self):
        cfg=dict(self.h.cfg,device_id='test-user');cfg.pop('user_selection',None)
        s=f.session(text='我喜欢恐龙呀。');out=self.h.out/'run'
        self.fake.fail_review=True
        with patch('memory_agent.JsonClient',return_value=self.fake),patch('memory_agent.load_inputs',return_value=([s],{'device_id':'test-user'})),redirect_stdout(io.StringIO()):
            self.assertEqual(run_one(cfg,'live',str(out)),2)
            self.fake.fail_review=False; cfg['history_context_tokens']=2048
            self.assertEqual(run_one(cfg,'live',str(out),resume=True),0)
        self.assertEqual(len([x for x in self.fake.calls if x[0].startswith('source_extractor:')]),1)

    def test_one_bad_candidate_retry_keeps_good_source_proof(self):
        s=f.session(text='我喜欢恐龙也喜欢汽车')
        a=raw(s);b=raw(s,'喜欢汽车');b['fact']=fact('喜欢','汽车','汽车')
        broken=copy.deepcopy(b);broken['evidence'][0]['speaker']='assistant'
        original=self.fake.ask; self.fix=False
        def ask(role,prompt,data):
            if role.startswith('source_extractor:'):
                self.fake.calls.append((role,copy.deepcopy(data)))
                return dict(candidates=[b if self.fix else broken] if 'candidate_to_fix' in data else [a,broken],notes='')
            return original(role,prompt,data)
        with patch.object(self.fake,'ask',side_effect=ask):
            first=self.h.run_session(s);self.assertEqual(first['status'],'partial')
            self.fix=True;second=self.h.run_session(s)
        self.assertEqual(second['status'],'completed')
        reviewed=[c['claim'] for role,data in self.fake.calls if role=='source_reviewer' for c in data['candidates']]
        self.assertEqual(reviewed.count('喜欢恐龙'),1)
        self.assertEqual(len(second['approved']),2)

    def test_source_pair_ids_strict_and_transaction_gate(self):
        s=f.session(); p=GenericPipeline(self.h.agent,s)
        a=p.normalize_candidate(materialize(raw(s),s),'interests_preferences',s,'w')
        b=p.normalize_candidate(materialize(raw(s,'另一事实'),s),'interests_preferences',s,'w')
        ps=[p.proof(c,f.verdict(c),p.pack([c]),[]) for c in (a,b)]
        with self.assertRaises(ValueError): verify_consistency([a,b],ps,[])

    def test_history_full_episode_is_not_sent(self):
        self.h.run_session()
        s=f.session('second','我喜欢汽车')
        c=raw(s,'喜欢汽车');c['fact']=fact('喜欢','汽车','汽车');self.fake.items=[c]
        r=self.h.run_session(s)
        calls=[data for role,data in self.fake.calls if role=='history_compare']
        self.assertTrue(calls)
        self.assertTrue(all('structure' not in m and 'evidence' not in m for data in calls for m in data['history']))
        self.assertEqual(r['status'],'completed')


if __name__=='__main__': unittest.main()
