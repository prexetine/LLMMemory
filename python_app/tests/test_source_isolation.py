"""真实故障类型的离线回归，不调用模型。"""
import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import test_source_agent_v4 as f
import test_memory_management as management
from test_memory_management import structure
from test_chain_facts import fact
from agent_context import windows
from legacy_source_v4 import SourcePipeline
from agent_contracts import validate_candidate
from agent_source_consistency import align_source_proofs, base_preference_support
from agent_fact_policy import compatible_scope, relation_guard
from export_memory_html import export_user


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.h = f.V4Tests(); self.h.setUp(); self.addCleanup(self.h.doCleanups)

    def test_actual_photo_window_review_and_commit(self):
        from test_agent_photos import fixture, pixels, config
        from agent_photos import bind_image
        root = self.h.out / 'photos'; root.mkdir()
        s = fixture(root)
        with patch('agent_photos.urllib.request.build_opener', return_value=SimpleNamespace(open=lambda *a, **k: io.BytesIO(pixels()))):
            bind_image(s, root, config(), self.h.fake)
        raw = f.raw(s, '画面有红色物体', kind='observation', assertion_mode='observation',
                    subject='照片画面', subject_role='other', subject_key=s['turns'][0]['evidence_id'], scope='仅此照片', fact=fact('呈现', '物体', '红色', 'observation'),
                    structure=structure('visual', 'photo', '物体'))
        self.h.fake.items = [raw]
        self.assertEqual(next(windows(s, 8192))['source_kind'], 'photo')
        r = self.h.run_session(s)
        self.assertEqual(r['commit']['counts']['new'], 1)
        self.assertEqual(r['approved'][0]['structure']['type'], 'visual')

    def test_steps_evidence_and_dialogue_observation_event(self):
        case = management.ManagementTests(); s, raw = case.learning()
        raw['assertion_mode'] = 'observation'
        raw['fact']['type'] = 'attribute'
        raw['evidence'] = raw['evidence'][:1]
        c = validate_candidate(raw, self.h.registry.dimensions['E44']['owner_skill_id'], self.h.registry, s, 'w')
        self.assertEqual(c['fact']['type'], 'event')
        self.assertEqual(len(c['evidence']), 2)
        self.assertEqual(c['retention'], 'event')

    def test_one_bad_row_does_not_lose_good_row_and_both_html(self):
        s = f.session(text='我喜欢恐龙，也喜欢汽车')
        good, bad = f.raw(s), f.raw(s, '喜欢汽车', subject=None)
        original = self.h.fake.ask
        def ask(role, prompt, data):
            if role.startswith('source_extractor:'):
                return dict(candidates=[data['candidate_to_fix']] if 'candidate_to_fix' in data else [good, bad], notes='')
            return original(role, prompt, data)
        with patch.object(self.h.fake, 'ask', side_effect=ask):
            r = self.h.run_session(s)
            self.assertEqual(len(r['approved']), 1)
            self.assertEqual(len(r['extraction_issues']), 1)
            self.assertTrue(r['needs_attention'])
            self.assertEqual(self.h.run_session(s), r)
        case = management.ManagementTests(); case.h = self.h; case.write_report([r])
        export_user(self.h.out, self.h.out / 'index.html')
        for name in ('index.html', 'index_report.html'):
            html = (self.h.out / name).read_text(encoding='utf-8')
            self.assertIn('提取待整理', html)
            self.assertIn('喜欢汽车', html)

    def test_same_source_semantic_disagreement_not_vote(self):
        s = f.session(text='每天我们都可以')
        a = self.h.normalized(s, claim='希望每天和助手聊天', fact=fact('愿望', '每天和助手聊天', '每天'))
        b = self.h.normalized(s, claim='希望每天都能聊天', fact=fact('愿望', '和助手聊天', '每天'))
        a['dimension_id'] = b['dimension_id'] = 'G19'
        p = SourcePipeline(self.h.agent, s)
        v = f.verdict(b, 'pending'); v['checks']['semantics'] = 'uncertain'; v['reason'] = '我们指代不清'
        proofs = [p.proof(a, f.verdict(a), p.pack([a]), []), p.proof(b, v, p.pack([b]), [])]
        aligned = align_source_proofs([a,b], proofs)
        self.assertEqual([x['decision'] for x in aligned], ['pending', 'pending'])
        c = copy.deepcopy(b); c['fact']['object'] = '恐龙'
        self.assertEqual(align_source_proofs([a,c], proofs)[0]['decision'], 'pass')

    def test_scope_and_degree_guard_boundaries(self):
        s = f.session(text='我喜欢和助手聊天')
        a = self.h.normalized(s, scope='all_time', fact=fact('喜欢','和助手聊天','喜欢'))
        b = dict(a, scope='session:s1', memory_id='m', revision=1, state='reviewed', fact=fact('喜欢','和智能助手一起呆着聊天','最喜欢'))
        self.assertTrue(compatible_scope(a,b))
        self.assertFalse(compatible_scope(a,dict(b,origin_session_id='other')))
        self.assertFalse(compatible_scope(a,dict(b,scope='和爸爸在家')))
        d = dict(action='reinforce',targets=[dict(memory_id='m',revision=1)],basis=[],reason='基础支持',support_kind='base_only')
        self.assertEqual(relation_guard(a,d,[b])['action'], 'reinforce')
        self.assertFalse(base_preference_support(b,a))
        self.assertFalse(base_preference_support(a,dict(b,valid_from='2020-01-01')))

    def test_base_support_keeps_degree_and_independent_count(self):
        first = f.session('strong', '我最喜欢和助手聊天')
        self.h.fake.items = [f.raw(first, '最喜欢和助手聊天', scope='all_time', fact=fact('喜欢','和助手聊天','最喜欢'))]
        self.h.run_session(first)
        second = f.session('weak', '我喜欢和助手聊天')
        self.h.fake.items = [f.raw(second, '喜欢和助手聊天', scope='all_time', fact=fact('喜欢','和助手聊天','喜欢'))]
        r = self.h.run_session(second)
        self.assertEqual(r['commit']['counts']['reinforce'], 1)
        memory = next(iter(self.h.state()['memories'].values()))
        self.assertEqual(memory['fact']['value'], '最喜欢')
        self.assertEqual(memory['support_count'], 2)
        self.assertEqual(memory['support_annotations'][-1]['support_kind'], 'base_only')
        self.h.run_session(second)
        self.assertEqual(next(iter(self.h.state()['memories'].values()))['support_count'], 2)


if __name__ == '__main__':
    unittest.main()
