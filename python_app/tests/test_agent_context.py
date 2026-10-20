"""有界上下文与完整覆盖回归；模型网络全部使用合成桩。"""
import copy
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from test_memory_agent import REGISTRY, FakeClient, candidate, session, review_result, audit
from legacy_memory_agent import CentralAgent, EXTRACTOR, validate_candidate, BASE, run
from agent_context import (BudgetClient, ContextCapacityError, windows, size, check_request,
                           bounded_history, related_history, input_budget, DEFAULTS)
from legacy_execution import extract_pages, audit_pages
from agent_review import ItemReviewer
from agent_store import MemoryStore
from agent_llm import JsonClient


def multi_session(count=20, width=150):
    s = session()
    s['turns'] = []
    for i in range(count):
        t = session(f't{i}', text='我喜欢恐龙' + '甲' * width)['turns'][0]
        t['assistant_text'] = '乙' * width
        s['turns'].append(t)
    from agent_data import digest
    s['input_hash'] = digest(s['turns'])
    return s


class RecordingClient(FakeClient):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.requests = []

    def ask(self, role, prompt, data):
        check_request(self.config, role, prompt, data)
        self.requests.append((role, copy.deepcopy(data)))
        if role == 'planner':
            return {'action': 'finish', 'skill_id': '', 'query': '', 'reason': 'done'}
        if role.startswith('extractor:'):
            found = next((t for t in data['session']['turns'] if '我喜欢恐龙' in t['user_text']), None)
            if found:
                c = candidate(dict(data['session'], turns=[found]))
                return {'candidates': [c], 'notes': '仅测试'}
            return {'candidates': [], 'notes': '无证据'}
        return super().ask(role, prompt, data)


class ContextTests(unittest.TestCase):
    def assert_coverage(self, original, chunks):
        for t in original['turns']:
            for field in ('user_text', 'assistant_text'):
                intervals = []
                for chunk in chunks:
                    for fragment in chunk['turns']:
                        if fragment['evidence_id'] != t['evidence_id']:
                            continue
                        if field in fragment['source_spans']:
                            start, end = fragment['source_spans'][field]
                            self.assertEqual(fragment[field], t[field][start:end])
                            intervals.append((start, end))
                cursor = 0
                for start, end in sorted(intervals):
                    self.assertLessEqual(start, cursor)
                    cursor = max(cursor, end)
                self.assertEqual(cursor, len(t[field]))

    def test_long_single_turn_unicode_and_assistant_are_not_lost(self):
        s = session(text='长😀文' * 3000)
        s['turns'][0]['assistant_text'] = '回答𠮷' * 1000
        chunks = list(windows(s, 1800))
        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(size(c) <= 1800 for c in chunks))
        self.assert_coverage(s, chunks)

    def test_multiturn_windows_preserve_source_intervals(self):
        s = multi_session()
        chunks = list(windows(s, 2200))
        self.assert_coverage(s, chunks)
        self.assertEqual(chunks, list(windows(s, 2200)))

    def test_schema_and_output_reserve_count_towards_budget(self):
        with self.assertRaises(ContextCapacityError):
            check_request({'context_window_tokens': 22000}, 'auditor', '', {'catalog': REGISTRY.catalog()})
        with self.assertRaises(ValueError):
            input_budget({'context_window_tokens': 12000})

    def test_bounded_history_never_accumulates_and_required_target_not_dropped(self):
        items = [{'memory_id': str(i), 'claim': '话' * 100} for i in range(100)]
        chosen = bounded_history(items + items, 1500, 3)
        self.assertLessEqual(size(chosen), 1500)
        self.assertLessEqual(len(chosen), 3)
        required = dict(items[0], dimension_id='G16', evidence=[])
        c = dict(candidate(), target_memory_id='0')
        with self.assertRaises(ContextCapacityError):
            related_history([required], c, 10)

    def test_long_session_end_to_end_keeps_requests_bounded_and_one_commit(self):
        config = {'session_chunk_tokens': 1800, 'max_agent_steps': 4, 'max_audit_rounds': 2, 'history_limit': 5}
        client = RecordingClient(config)
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / 'db')
            agent = CentralAgent(client, REGISTRY, store, config, Path(tmp) / 'trace')
            s = multi_session(12, 80)
            result = agent.run_session(s)
            self.assertEqual(result['status'], 'completed')
            self.assertGreater(len(result['windows']), 1)
            self.assertGreater(len(result['approved']), 0)
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0], 1)
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM evidence').fetchone()[0], 12)
            for role, data in client.requests:
                self.assertNotIn('actions', data)
                self.assertNotIn('issues', data)
                if role == 'planner':
                    self.assertNotIn('candidates', data)
                    self.assertNotIn('history', data)
            # 正式提交幂等且复用Review凭证，不再次请求模型。
            count = len(client.requests)
            second = agent.run_session(s)
            self.assertEqual(len(client.requests), count)
            self.assertEqual(second['receipt']['status'], 'already_committed')
            store.close()

    def test_later_window_negation_prevents_earlier_memory_commit(self):
        class NegationClient(RecordingClient):
            def ask(self, role, prompt, data):
                if role == 'reviewer' and any('不要保存' in t['user_text'] for t in data['session']['turns']):
                    return dict(review_result(data['candidate']), decision='reject', reason='后续撤回')
                return super().ask(role, prompt, data)
        config = {'session_chunk_tokens': 1600, 'max_agent_steps': 4, 'max_audit_rounds': 2, 'history_limit': 5}
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / 'db')
            s = multi_session(6, 60)
            s['turns'][-1]['user_text'] = '刚才说的是故事，不要保存这些喜好'
            result = CentralAgent(NegationClient(config), REGISTRY, store, config, Path(tmp) / 'trace').run_session(s)
            self.assertEqual(result['approved'], [])
            self.assertTrue(any(r['decision'] == 'reject' for r in result['reviews']))
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0], 0)
            store.close()

    def test_runtime_capacity_error_replans_smaller_windows(self):
        class SmallProvider(RecordingClient):
            def ask(self, role, prompt, data):
                if role == 'router' and size(data['session']) > 1800:
                    raise ContextCapacityError('synthetic smaller provider window')
                return super().ask(role, prompt, data)
        config = {'session_chunk_tokens': 4000, 'max_agent_steps': 4, 'max_audit_rounds': 2, 'history_limit': 5}
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / 'db')
            agent = CentralAgent(SmallProvider(config), REGISTRY, store, config, Path(tmp) / 'trace')
            result = agent.run_session(multi_session(6, 80))
            self.assertEqual(result['status'], 'completed')
            self.assertIn('window_replan', (Path(tmp) / 'trace').read_text(encoding='utf-8'))
            store.close()

    def test_interruption_reuses_windows_without_early_formal_memory(self):
        class InterruptClient(RecordingClient):
            def ask(self, role, prompt, data):
                if role == 'router' and sum(r == 'router' for r, _ in self.requests) == 1:
                    raise KeyboardInterrupt()
                return super().ask(role, prompt, data)
        config = {'session_chunk_tokens': 1600, 'max_agent_steps': 4, 'max_audit_rounds': 2, 'history_limit': 5}
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / 'db')
            s = multi_session(6, 60)
            with self.assertRaises(KeyboardInterrupt):
                CentralAgent(InterruptClient(config), REGISTRY, store, config, Path(tmp) / 'trace').run_session(s)
            self.assertGreater(store.db.execute('SELECT COUNT(*) FROM context_work').fetchone()[0], 0)
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0], 0)
            client = RecordingClient(config)
            result = CentralAgent(client, REGISTRY, store, config, Path(tmp) / 'trace').run_session(s)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(sum(r == 'router' for r, _ in client.requests), len(result['windows']) - 1)
            store.close()

    def test_revision_routes_only_selected_rules_and_rereviews(self):
        class Client(RecordingClient):
            def ask(self, role, prompt, data):
                if role == 'reviewer' and data['candidate']['claim'] == candidate()['claim']:
                    return dict(review_result(data['candidate']), decision='revise', reason='措辞调整')
                if role == 'revision_router':
                    self.requests.append((role, data))
                    return {'dimension_ids': ['G16']}
                if role == 'extractor:review_revision':
                    self.requests.append((role, data))
                    c = candidate()
                    c['claim'] = '发言者自述喜欢恐龙'
                    return {'candidates': [c], 'notes': '修订'}
                return super().ask(role, prompt, data)
        client = Client()
        c = validate_candidate(candidate(), 'interests_preferences', REGISTRY, session(), [])
        accepted, _ = ItemReviewer(client, REGISTRY, validate_candidate, EXTRACTOR).review([c], session(), [], lambda r: None)
        self.assertEqual(len(accepted), 1)
        repair = next(d for role, d in client.requests if role == 'extractor:review_revision')
        self.assertEqual([d['dimension_id'] for d in repair['skill_rules']], ['G16'])

    def test_provider_context_error_classified_and_logged(self):
        client = JsonClient.__new__(JsonClient)
        client.config = {'provider': 'Deepseek', 'max_calls': None, 'max_total_input_chars': None,
                         'max_output_tokens': 100, 'timeout_seconds': 1, 'network_retries': 0}
        client.calls = client.input_chars = client.total_tokens = 0
        client.elapsed = 0
        client.url, client.key, client.model = 'https://example.invalid', 'test', 'test'
        from unittest.mock import Mock
        client.opener = Mock()
        client.opener.open.side_effect = urllib.error.HTTPError(client.url, 400, 'bad', {},
            io.BytesIO(b'{"error":{"code":"context_length_exceeded"}}'))
        with tempfile.TemporaryDirectory() as tmp:
            client.log_path = Path(tmp) / 'calls'
            with self.assertRaises(ContextCapacityError):
                client.ask('test', '', {})
            record = json.loads(client.log_path.read_text(encoding='utf-8'))
            self.assertEqual(record['error'], 'provider_context_capacity')
            self.assertIn('estimated_input_tokens', record)

    def test_many_candidates_audited_in_pages_without_dropping_ids(self):
        from types import SimpleNamespace
        from agent_context import BudgetClient
        s = session()
        candidates = []
        for i in range(60):
            c = candidate()
            c['claim'] += str(i)
            candidates.append(validate_candidate(c, 'interests_preferences', REGISTRY, s, []))
        client = RecordingClient()
        agent = SimpleNamespace(client=BudgetClient(client, {}), config={}, registry=REGISTRY)
        result = audit_pages(agent, s, candidates, [])
        self.assertEqual(set(result['approved_candidate_ids']), {c['candidate_id'] for c in candidates})
        self.assertGreater(sum(role == 'auditor' for role, _ in client.requests), 1)

    def test_context_replan_exhaustion_never_commits(self):
        class RejectClient(RecordingClient):
            def ask(self, role, prompt, data):
                raise ContextCapacityError('provider cannot accept fixed prompt')
        config = {'context_split_retries': 1, 'max_agent_steps': 4, 'max_audit_rounds': 2, 'history_limit': 5}
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / 'db')
            with self.assertRaises(ContextCapacityError):
                CentralAgent(RejectClient(), REGISTRY, store, config, Path(tmp) / 'trace').run_session(multi_session(4))
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0], 0)
            store.close()

    def test_compact_history_does_not_send_old_assistant_or_full_data(self):
        from test_memory_agent import commit_with_review
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / 'db')
            s = session()
            s['turns'][0]['assistant_text'] = 'private assistant text' * 10000
            c = validate_candidate(candidate(), 'interests_preferences', REGISTRY, s, [])
            commit_with_review(store, s, [c], audit([c]), REGISTRY.version, 'completed', [])
            compact = store.history(s['device_id'], '2027', 'G16', 30, compact=True)
            self.assertEqual(len(compact), 1)
            self.assertNotIn('assistant_text', json.dumps(compact))
            self.assertNotIn('data', compact[0])
            self.assertLess(size(compact), 2000)
            store.close()

    def test_selected_users_real_data_window_plan_without_model(self):
        import time
        import tracemalloc
        from agent_data import load_sessions
        from legacy_memory_agent import select_devices, ROUTER, AUDITOR
        from agent_context import request_size
        config = json.loads((BASE / 'python_app/config/memory_agent.json').read_text(encoding='utf-8'))
        started = time.perf_counter()
        tracemalloc.start()
        sessions_total = windows_total = max_original = max_input = 0
        users = select_devices(config)[:2]
        try:
            for device in users:
                sessions, _ = load_sessions(BASE / config['data_root'], device, config['gap_minutes'])
                for s in sessions:
                    sessions_total += 1
                    max_original = max(max_original, size(s))
                    chunks = list(windows(s, config.get('session_chunk_tokens', DEFAULTS['session_chunk_tokens'])))
                    self.assert_coverage(s, chunks)
                    for chunk in chunks:
                        windows_total += 1
                        # 实数据原文窗口的路由与空候选覆盖复核；候选/历史分页另有合成测试。
                        check_request(config, 'router', ROUTER, {'session': chunk, 'catalog': REGISTRY.catalog()})
                        data = {'session': chunk, 'catalog': REGISTRY.catalog(), 'candidates': [], 'candidate_rules': []}
                        estimated = check_request(config, 'auditor', AUDITOR, data)
                        max_input = max(max_input, estimated)
                del sessions
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        print(json.dumps({'context_plan_only': True, 'users': len(users), 'sessions': sessions_total,
                          'windows': windows_total, 'max_original_utf8_bytes': max_original,
                          'max_checked_request_estimate': max_input,
                          'input_budget': input_budget(config), 'peak_python_bytes': peak,
                          'elapsed_seconds': round(time.perf_counter() - started, 3)}, ensure_ascii=False))

    def test_partial_window_checkpoint_is_retryable_and_not_formal(self):
        class PartialClient(RecordingClient):
            def ask(self, role, prompt, data):
                if role == 'auditor':
                    raise ValueError('synthetic invalid audit')
                return super().ask(role, prompt, data)
        config = {'max_agent_steps': 3, 'max_audit_rounds': 2, 'history_limit': 5}
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / 'db')
            first = CentralAgent(PartialClient(), REGISTRY, store, config, Path(tmp) / 'trace').run_session(session())
            self.assertEqual(first['status'], 'partial')
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0], 0)
            second = CentralAgent(RecordingClient(), REGISTRY, store, config, Path(tmp) / 'trace').run_session(session())
            self.assertEqual(second['status'], 'completed')
            self.assertGreater(len(second['approved']), 0)
            store.close()

    def test_history_multiple_quotes_same_evidence_id_do_not_overwrite(self):
        s = session()
        history = [dict(memory_id=str(i), dimension_id='G16', evidence=[{'evidence_id': 'old:1',
                    'data': {'user_text': text}}]) for i, text in enumerate(('喜欢猫', '喜欢狗'))]
        c = candidate()
        c['evidence'].append({'evidence_id': 'old:1', 'quote': '喜欢猫'})
        checked = validate_candidate(c, 'interests_preferences', REGISTRY, s, history)
        self.assertEqual(checked['evidence'], c['evidence'])

    def test_old_review_run_cannot_be_resumed_as_bounded_version(self):
        config = json.loads((BASE / 'python_app/config/memory_agent.json').read_text(encoding='utf-8'))
        config.pop('user_selection')
        config.update(device_id='test-user', max_sessions=1)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            out = folder / 'out'
            out.mkdir()
            cp = folder / 'config.json'
            cp.write_text(json.dumps(config), encoding='utf-8')
            (out / 'manifest.json').write_text(json.dumps({'review_version': 'item-review-v1'}), encoding='utf-8')
            (out / 'report.json').write_text('{}', encoding='utf-8')
            with patch('legacy_memory_agent.load_sessions', return_value=([session()], {})), patch('legacy_memory_agent.JsonClient') as client:
                with self.assertRaisesRegex(ValueError, '旧运行'):
                    run(cp, 'live', str(out), True)
                client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
