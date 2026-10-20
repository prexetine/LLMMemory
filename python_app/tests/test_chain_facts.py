"""2026-10-08真实错误类型回归；网络与语义模型均模拟，不计线上质量。"""
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import test_source_agent_v4 as fixtures
from agent_fact_policy import relation_guard, normalize_fact, same_fact
from agent_contracts import source_result, FORBIDDEN
from agent_v4_schema import schema
from agent_llm import JsonClient, TransientModelError
from agent_maintenance import apply_decision
from legacy_memory_agent import run_one
from export_memory_html import export_user, build_data
from memory_view_data import build_report_user


def fact(predicate="喜好", obj="恐龙", value="喜欢", kind="attribute", polarity="positive", time_text=""):
    return dict(type=kind, predicate=predicate, object=obj, value=value, polarity=polarity,
                cardinality="multiple", time_text=time_text, time_value=None)


class ChainFactsTests(unittest.TestCase):
    def setUp(self):
        self.h = fixtures.V4Tests()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)

    def test_real_compatible_examples_never_conflict(self):
        pairs = [(fact("位置", "四川", "身处"), fact("位置", "医院", "身处")),
                 (fact("成长", "本人", "长大"), fact("睡眠", "本人", "刚睡醒")),
                 (fact("关注", "叶子", "关注"), fact("关注", "雨势", "关注")),
                 (fact("目标", "公园", "找叶子"), fact("目标", "公园", "找虫子")),
                 (fact("收听", "故事", "斑点龙", "request"), fact("收听", "故事", "兔一一", "request"))]
        for a,b in pairs:
            c = dict(fact=a, subject_key="u", scope="session:s", origin_session_id="s")
            m = dict(c, fact=b, memory_id="m", revision=1)
            d = dict(action="conflict",targets=[dict(memory_id="m",revision=1)],basis=[],reason="同维度互斥")
            self.assertEqual(relation_guard(c,d,[m])["action"],"coexist")

    def test_normal_event_progress_and_simultaneous_events(self):
        s = fixtures.session(text="我在水上乐园玩。我出来了，我在休息呢。")
        self.h.fake.items = [fixtures.raw(s,"在水上乐园玩",assertion_mode="temporary",fact=fact("出游", "水上乐园", "玩耍", "event")),
                             fixtures.raw(s,"从水上乐园出来休息",assertion_mode="temporary",fact=fact("出游", "水上乐园", "出来休息", "event"))]
        self.h.fake.action = "conflict"
        r = self.h.run_session(s)
        self.assertEqual(len(r['approved']),2)
        self.assertEqual([d['action'] for d in r['maintenance']],['new','coexist'])
        self.assertFalse(self.h.state()['groups'])
        for c in r['approved']:
            self.assertEqual(c['fact']['type'],'event')
            self.assertIsNone(c['event']['time'])

    def test_absolute_and_relative_event_time(self):
        for text, expected in [('2026年9月7日去了水上乐园','2026-09-07'),('昨天去了水上乐园',None),('去了水上乐园',None)]:
            s=fixtures.session(text=text)
            time_text='2026年9月7日' if expected else '昨天' if text.startswith('昨天') else ''
            c=self.h.normalized(s,fact=fact('出游','水上乐园','游玩','event',time_text=time_text))
            self.assertEqual(c['event']['time'],expected)
        s=fixtures.session(text='去了水上乐园')
        with self.assertRaises(ValueError): self.h.normalized(s,fact=fact(time_text='昨天'))

    def test_source_unclassified_and_inconsistent_rows_do_not_retry_whole_batch(self):
        s=fixtures.session(text='我喜欢恐龙')
        c=self.h.normalized(s,dimension_id='UNCLASSIFIED')
        r={'results':[fixtures.verdict(c)]}
        source_result(r,[c],{s['turns'][0]['evidence_id']},normalize=True)
        self.assertEqual(r['results'][0]['decision'],'pass')
        c=self.h.normalized(s,subject_role='unresolved')
        r={'results':[fixtures.verdict(c)]}
        source_result(r,[c],{s['turns'][0]['evidence_id']},normalize=True)
        self.assertEqual(r['results'][0]['decision'],'pending')
        self.assertIn('主体',r['results'][0]['reason'])

    def test_unsupported_rules_not_sent_or_in_schema(self):
        self.h.run_session()
        for role,data in self.h.fake.calls:
            if role.startswith('source_extractor'):
                self.assertFalse({r['dimension_id'] for r in data['skill_rules']} & FORBIDDEN)
                enums=schema(role,data)['properties']['candidates']['items']['properties']['dimension_id']['enum']
                self.assertFalse(set(enums)&FORBIDDEN)

    def test_unclassified_can_finish_but_not_admit(self):
        s=fixtures.session(text='我喜欢恐龙')
        self.h.fake.items=[fixtures.raw(s,dimension_id='UNCLASSIFIED')]
        r=self.h.run_session(s)
        self.assertEqual(r['status'],'completed')
        self.assertEqual(r['reviews'][0]['source_decision'],'pass')
        self.assertFalse(r['approved'])
        self.assertIn('未分类',r['maintenance'][0]['reason'])

    def test_same_fact_cross_dimension_reinforces_once(self):
        s=fixtures.session(text='我喜欢恐龙')
        a=self.h.normalized(s,fact=fact())
        b=copy.deepcopy(a);b['dimension_id']='G17';b['candidate_id']='other-dimension'
        from legacy_source_v4 import SourcePipeline
        p=SourcePipeline(self.h.agent,s)
        proofs=[p.proof(c,fixtures.verdict(c),s['turns'],[]) for c in [a,b]]
        receipt,_=p.maintain([a,b],proofs,[],{})
        self.assertEqual(receipt['counts']['reinforce'],1)
        m=next(iter(self.h.state()['memories'].values()))
        self.assertEqual(m['dimension_ids'],['G16','G17'])
        self.assertEqual(m['support_count'],1)

    def test_events_from_different_sessions_not_merged(self):
        for sid in ('s1','s2'):
            s=fixtures.session(sid,'在水上乐园玩')
            self.h.fake.items=[fixtures.raw(s,'在水上乐园玩',fact=fact('出游','水上乐园','玩','event'))]
            self.h.run_session(s)
        self.assertEqual(len(self.h.state()['memories']),2)

    def test_false_withdrawal_does_not_abort_router(self):
        s=fixtures.session(text='暂停，我喜欢恐龙')
        signal=dict(kind='withdrawal',subject_key=None,attribute_key=None,scope='session',target_type='session',target_text='',evidence=[dict(evidence_id=s['turns'][0]['evidence_id'],quote='暂停')])
        self.h.fake.signals=[signal]
        r=self.h.run_session(s)
        self.assertEqual(r['signals'],[])
        self.assertEqual(len(r['approved']),1)
        self.assertEqual(sum(k=='source_router' for k,_ in self.h.fake.calls),1)

    def test_transient_request_failure_continues_and_resume(self):
        cfg=dict(self.h.cfg,device_id='test-user');cfg.pop('user_selection',None)
        sessions=[fixtures.session('bad','我喜欢恐龙'),fixtures.session('good','我喜欢恐龙')]
        fake=fixtures.Fake();original=fake.ask
        def ask(role,prompt,data):
            if data.get('session',{}).get('session_id')=='bad': raise TransientModelError('模拟重试耗尽')
            return original(role,prompt,data)
        out=self.h.out/'run'
        with patch.object(fake,'ask',side_effect=ask),patch('legacy_memory_agent.JsonClient',return_value=fake),patch('legacy_memory_agent.load_inputs',return_value=(sessions,{'device_id':'test-user'})),redirect_stdout(io.StringIO()):
            self.assertEqual(run_one(cfg,'live',str(out)),2)
        report=json.loads((out/'report.json').read_text(encoding='utf-8'))
        self.assertEqual([r['status'] for r in report['results']],['failed','completed'])
        fresh=fixtures.Fake()
        with patch('legacy_memory_agent.JsonClient',return_value=fresh),patch('legacy_memory_agent.load_inputs',return_value=(sessions,{'device_id':'test-user'})),redirect_stdout(io.StringIO()):
            self.assertEqual(run_one(cfg,'live',str(out),True),0)
        self.assertFalse(any(d.get('session',{}).get('session_id')=='good' for _,d in fresh.calls))

    def test_client_timeout_bounded_retry(self):
        client=object.__new__(JsonClient);client.config={'network_retries':1}
        with patch.object(client,'_ask_once',side_effect=TimeoutError('read timeout')) as call:
            with self.assertRaises(TransientModelError):client.ask('x','x',{})
            self.assertEqual(call.call_count,2)
        with patch.object(client,'_ask_once',side_effect=RuntimeError('HTTP 401')) as call:
            with self.assertRaises(RuntimeError):client.ask('x','x',{})
            self.assertEqual(call.call_count,1)

    def test_http_500_429_retry_and_401_stop(self):
        from unittest.mock import Mock
        for code, expected in [(500,2),(429,2),(401,1)]:
            client=object.__new__(JsonClient)
            client.config=dict(provider='Deepseek',max_calls=None,max_total_input_chars=None,max_output_tokens=100,timeout_seconds=1,network_retries=1)
            client.calls=client.input_chars=client.total_tokens=0;client.elapsed=0
            client.url,client.key,client.model='https://example.invalid','test','test'
            client.log_path=self.h.out/('http-'+str(code)+'.jsonl')
            client.opener=Mock()
            client.opener.open.side_effect=lambda *a,**kw: (_ for _ in ()).throw(urllib.error.HTTPError(client.url,code,'test',{},io.BytesIO(b'{}')))
            with self.assertRaises(TransientModelError if code!=401 else RuntimeError):client.ask('test','',{})
            self.assertEqual(client.opener.open.call_count,expected)
            rows=[json.loads(s) for s in client.log_path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows),expected)
            self.assertTrue(all(r['error']=='HTTP '+str(code) for r in rows))

    def test_same_time_does_not_merge_different_facts(self):
        s=fixtures.session(text='我喜欢恐龙和汽车')
        a=self.h.normalized(s,fact=fact(obj='恐龙'))
        b=self.h.normalized(s,claim='喜欢汽车',fact=fact(obj='汽车'))
        self.assertFalse(same_fact(a,b))

    def test_repeated_event_same_session_not_assumed_same_occurrence(self):
        s=fixtures.session(text='去了水上乐园')
        a=self.h.normalized(s,fact=fact('出游','水上乐园','玩','event'))
        b=copy.deepcopy(a);b['evidence']=[dict(evidence_id='another-turn',quote='又去了水上乐园')]
        self.assertFalse(same_fact(a,b))

    def test_source_batch_completeness_not_relaxed(self):
        s=fixtures.session();c=self.h.normalized(s)
        for rows in [[],[fixtures.verdict(c),fixtures.verdict(c)],[dict(fixtures.verdict(c),candidate_id='extra')]]:
            with self.assertRaises(ValueError):source_result({'results':rows},[c],{s['turns'][0]['evidence_id']},normalize=True)

    def test_latest_version_export_does_not_mutate_database(self):
        self.h.fake.repair=True
        s=fixtures.session(text='我喜欢恐龙和汽车')
        self.h.fake.items=[fixtures.raw(s,'喜欢恐龙和汽车')]
        r=self.h.run_session(s)
        folder=self.h.out
        (folder/'report.json').write_text(json.dumps(dict(device_id='test-user',status='completed',selected_sessions=1,results=[r],review_version='source-maintenance-v4')),encoding='utf-8')
        (folder/'manifest.json').write_text(json.dumps(dict(review_version='source-maintenance-v4')),encoding='utf-8')
        before=self.h.store.db.total_changes
        export_user(folder,folder/'index.html')
        formal=build_data(folder,fixtures.BASE/self.h.cfg['rules_file'])
        cards=build_report_user(folder,formal)['cards']
        self.assertEqual(len(cards),2)
        self.assertFalse(any(c['claim']=='喜欢恐龙和汽车' for c in cards))
        self.assertEqual(self.h.store.db.total_changes,before)


if __name__=='__main__': unittest.main()
