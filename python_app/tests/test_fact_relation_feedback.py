"""Per-item relationship feedback regressions; synthetic inputs, no external calls."""
from collections import Counter
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python_app/src'))
sys.path.insert(0, str(ROOT / 'python_app/tests'))
from agent_context import ContextCapacityError
from agent_data import digest
from agent_fact_flow import FactDrivenPipeline
import test_fact_runner as runner

SOURCE = Path(os.environ.get('LLMMEMORY_RELATION_SOURCE', ROOT / 'python_app/src/agent_fact_relations.py'))
spec = importlib.util.spec_from_file_location('relationship_under_test', SOURCE)
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)


def turn(eid='e1', text='我喜欢恐龙和汽车'):
    return dict(evidence_id=eid, user_text=text, assistant_text='仅作为上下文')


def endpoint(cid, evidence, obj='恐龙'):
    return dict(candidate_id=cid, device_id='u1', subject='用户', subject_key='account:u1:self',
                scope='all_time', fact=dict(type='attribute', predicate='喜欢', object=obj,
                value=obj, polarity='positive', time_text=''), evidence=[dict(evidence_id=evidence, quote='合成证据')])


def relation(pid, eid='e1', end=None, kind='related_but_independent', direction='none'):
    return dict(pair_id=pid, relation=kind, reason='合成验收响应', direction=direction,
                basis_refs=[dict(evidence_id=eid, speaker='user', start=0 if end is not None else None, end=end)])


class LocalRuntime:
    """Small cache double; CLI tests below exercise the actual StageRuntime."""
    def __init__(self):
        self.attempts = Counter()
        self.cache = {}
        self.discarded = []

    def fingerprint(self, role, prompt, data):
        return digest([role, prompt, data])

    def load_item(self, role, prompt, data, validate):
        key = self.fingerprint(role + ':item', prompt, data)
        value = self.cache.get(key)
        if value is not None:
            try:
                validate(copy.deepcopy(value))
            except (ValueError, TypeError, KeyError):
                self.cache.pop(key)
            else:
                return key, copy.deepcopy(value)
        return key, None

    def save_item(self, key, role, data, row):
        self.cache[key] = copy.deepcopy(row)

    def discard_unit_cache(self, unit, reason):
        self.discarded.append((unit, reason))


class LocalPipeline:
    _validate_relation_envelope = staticmethod(FactDrivenPipeline._validate_relation_envelope)

    def __init__(self, turns, respond):
        from types import SimpleNamespace
        self.s = dict(device_id='u1', session_id='s1', input_hash='synthetic', turns=copy.deepcopy(turns))
        self.a = SimpleNamespace(config=dict(relation_batch_size=16))
        self.runtime = LocalRuntime()
        self.flow = 'fact-driven-v1'
        self.requests, self.tasks = [], []
        self.respond = respond

    def ask(self, role, prompt, data, validate, **kwargs):
        self.requests.append(copy.deepcopy(data))
        value = self.respond(data, len(self.requests))
        validate(value)
        return copy.deepcopy(value)

    def _task(self, stage, candidate_ids, reason, unit_id=None):
        value = dict(stage=stage, candidate_ids=candidate_ids, reason=reason, pair_id=unit_id)
        if not any(t['pair_id'] == unit_id for t in self.tasks):
            self.tasks.append(value)


def pair(pid='p1', eid='e1'):
    return {pid: (endpoint(pid + '-left', eid), endpoint(pid + '-right', eid, '汽车'))}


class RelationFeedbackTests(unittest.TestCase):
    def test_bad_span_retry_receives_precise_feedback(self):
        p = LocalPipeline([turn()], lambda data, n: dict(results=[relation('p1', end=999 if n == 1 else None)]))
        result = implementation.judge(p, pair(), [], 'unchanged-prompt')
        self.assertIn('p1', result)
        self.assertNotIn('validation_feedback', p.requests[0]['tasks'][0])
        feedback = p.requests[1]['tasks'][0]['validation_feedback']
        self.assertEqual(feedback['code'], 'span_not_visible')
        self.assertEqual(feedback['evidence_id'], 'e1')
        self.assertIn('引用超出', feedback['message'])
        self.assertEqual(feedback['visible_user_spans'], [dict(evidence_id='e1', speaker='user', start=0, end=len(turn()['user_text']))])
        self.assertFalse(p.tasks)

    def test_partial_batch_retries_only_failed_pair_without_cross_pair_evidence(self):
        pairs = dict(pair('good', 'e1'), **pair('bad', 'e2'))
        def respond(data, n):
            return dict(results=[relation(t['pair_id'], 'e1' if t['pair_id'] == 'good' else 'e2',
                                          end=999 if n == 1 and t['pair_id'] == 'bad' else None) for t in data['tasks']])
        p = LocalPipeline([turn(), turn('e2', '我想听故事')], respond)
        result = implementation.judge(p, pairs, [], 'unchanged-prompt')
        self.assertEqual(set(result), {'good', 'bad'})
        self.assertEqual([t['pair_id'] for t in p.requests[1]['tasks']], ['bad'])
        self.assertEqual({r['evidence_id'] for r in p.requests[1]['tasks'][0]['validation_feedback']['visible_user_spans']}, {'e2'})
        count = len(p.requests)
        implementation.judge(p, pairs, [], 'unchanged-prompt')
        self.assertEqual(len(p.requests), count)

    def test_missing_id_feedback_does_not_repeat_successful_pair(self):
        p = LocalPipeline([turn()], lambda data, n: dict(results=[relation(t['pair_id']) for t in data['tasks'] if n > 1 or t['pair_id'] == 'good']))
        result = implementation.judge(p, dict(pair('good'), **pair('bad')), [], 'prompt')
        self.assertEqual(set(result), {'good', 'bad'})
        self.assertEqual([t['pair_id'] for t in p.requests[1]['tasks']], ['bad'])
        self.assertIn('缺失或重复', p.requests[1]['tasks'][0]['validation_feedback']['message'])

    def test_duplicate_id_feedback_is_local_to_failed_pair(self):
        def respond(data, n):
            rows = [relation(t['pair_id']) for t in data['tasks']]
            return dict(results=rows + ([relation('bad')] if n == 1 else []))
        p = LocalPipeline([turn()], respond)
        implementation.judge(p, dict(pair('good'), **pair('bad')), [], 'prompt')
        self.assertEqual([t['pair_id'] for t in p.requests[1]['tasks']], ['bad'])
        self.assertIn('重复', p.requests[1]['tasks'][0]['validation_feedback']['message'])

    def test_extra_id_does_not_become_a_new_task(self):
        p = LocalPipeline([turn()], lambda data, n: dict(results=[relation('p1')] + ([relation('foreign')] if n == 1 else [])))
        implementation.judge(p, pair(), [], 'prompt')
        self.assertEqual([t['pair_id'] for t in p.requests[1]['tasks']], ['p1'])
        self.assertIn('额外', p.requests[1]['tasks'][0]['validation_feedback']['message'])
        self.assertNotIn('foreign', str(p.requests[1]))

    def test_cached_invalid_row_supplies_feedback_to_first_request(self):
        p = LocalPipeline([turn()], lambda data, n: dict(results=[relation('p1')]))
        # Learn the canonical key via a successful request, then inject an invalid cached row.
        implementation.judge(p, pair(), [], 'prompt')
        key = next(iter(p.runtime.cache))
        p.runtime.cache[key] = relation('p1', end=999)
        p.requests.clear()
        implementation.judge(p, pair(), [], 'prompt')
        self.assertEqual(p.requests[0]['tasks'][0]['validation_feedback']['code'], 'span_not_visible')
        self.assertEqual(set(p.runtime.cache), {key}, 'transient feedback must not change valid per-item cache identity')

    def test_visible_span_uses_original_fragment_offsets(self):
        fragment = turn(text='恐龙和汽车')
        fragment['source_spans'] = {'user_text': [10, 15]}
        def respond(data, n):
            row = relation('p1', end=999)
            if n > 1:
                row['basis_refs'][0].update(start=10, end=15)
            return dict(results=[row])
        p = LocalPipeline([fragment], respond)
        implementation.judge(p, pair(), [], 'prompt')
        self.assertEqual(p.requests[1]['tasks'][0]['validation_feedback']['visible_user_spans'][0]['start'], 10)
        self.assertEqual(p.requests[1]['tasks'][0]['validation_feedback']['visible_user_spans'][0]['end'], 15)

    def test_permanent_invalid_span_remains_pending_with_two_attempts(self):
        p = LocalPipeline([turn()], lambda data, n: dict(results=[relation('p1', end=999)]))
        self.assertEqual(implementation.judge(p, pair(), [], 'prompt'), {})
        self.assertEqual(len(p.requests), 2)
        self.assertEqual(len(p.tasks), 1)
        self.assertIn('validation_feedback', p.requests[1]['tasks'][0])
        implementation.judge(p, pair(), [], 'prompt')
        self.assertEqual(len(p.requests), 2, 'outer recovery must not reset per-pair attempt budget')

    def test_update_without_current_correction_is_not_accepted(self):
        p = LocalPipeline([turn()], lambda data, n: dict(results=[relation('p1', kind='changed_with_direction', direction='left_replaces_right')]))
        self.assertEqual(implementation.judge(p, pair(), [], 'prompt'), {})
        self.assertIn('更正或变化证据', p.requests[1]['tasks'][0]['validation_feedback']['message'])

    def test_update_with_current_correction_can_pass(self):
        p = LocalPipeline([turn()], lambda data, n: dict(results=[relation('p1', kind='changed_with_direction', direction='left_replaces_right')]))
        signals = [dict(kind='correction', evidence=[dict(evidence_id='e1', quote=turn()['user_text'])])]
        self.assertEqual(implementation.judge(p, pair(), signals, 'prompt')['p1']['direction'], 'left_replaces_right')
        self.assertEqual(len(p.requests), 1)

    def test_capacity_split_does_not_spend_semantic_attempts(self):
        def respond(data, n):
            if len(data['tasks']) > 1:
                raise ContextCapacityError('synthetic capacity')
            return dict(results=[relation(data['tasks'][0]['pair_id'])])
        p = LocalPipeline([turn()], respond)
        result = implementation.judge(p, dict(pair('a'), **pair('b')), [], 'prompt')
        self.assertEqual(set(result), {'a', 'b'})
        self.assertTrue(all(n == 1 for n in p.runtime.attempts.values()))
        self.assertTrue(all('validation_feedback' not in t for request in p.requests for t in request['tasks']))

    def test_feedback_does_not_mutate_pair_or_session_inputs(self):
        pairs = pair()
        p = LocalPipeline([turn()], lambda data, n: dict(results=[relation('p1', end=999 if n == 1 else None)]))
        old_pairs, old_session = copy.deepcopy(pairs), copy.deepcopy(p.s)
        implementation.judge(p, pairs, [], 'prompt')
        self.assertEqual(pairs, old_pairs)
        self.assertEqual(p.s, old_session)


class RelationFeedbackCliTests(unittest.TestCase):
    def setUp(self):
        self.h = runner.FactRunnerTests()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.h.write_dialogue('我喜欢恐龙，也喜欢汽车')
        class ScriptedModel(runner.OfflineModel):
            def __init__(self):
                super().__init__()
                self.relationship_requests = []
                self.permanent = False

            def ask(self, role, prompt, data):
                if role == 'source_relations':
                    self.calls.append(role)
                    self.relationship_requests.append(copy.deepcopy(data))
                    rows = []
                    for task in data['tasks']:
                        row = relation(task['pair_id'], task['evidence_pack'][0]['evidence_id'],
                                       end=None if task.get('validation_feedback') and not self.permanent else 999)
                        rows.append(row)
                    return dict(results=rows)
                result = super().ask(role, prompt, data)
                if role.startswith('source_extractor:'):
                    other = copy.deepcopy(result['candidates'][0])
                    other['claim'] = '我喜欢汽车'
                    other['fact'].update(object='汽车', value='汽车')
                    other['keywords'] = ['喜欢', '汽车']
                    result['candidates'].append(other)
                return result
        self.h.model = ScriptedModel()

    def run_cli(self, **kwargs):
        with patch('agent_fact_relations.judge', implementation.judge), \
                patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('Network is blocked in offline regressions')):
            return self.h.run_batch(**kwargs)

    def test_feedback_reaches_cli_database_report_html_and_resume(self):
        self.assertEqual(self.run_cli(), 0)
        self.h.assert_completed()
        self.assertEqual(self.h.query("SELECT COUNT(*) AS n FROM fact_assertions WHERE state='accepted'"), [dict(n=2)])
        self.assertEqual(self.h.query('SELECT * FROM fact_aliases'), [])
        self.assertEqual(self.h.query('PRAGMA foreign_key_check'), [])
        requests = self.h.model.relationship_requests
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1]['tasks'][0]['validation_feedback']['code'], 'span_not_visible')
        calls = list(self.h.model.calls)
        self.assertEqual(self.run_cli(resume=True), 0)
        self.assertEqual(self.h.model.calls, calls)
        self.assertEqual(self.h.query("SELECT COUNT(*) AS n FROM fact_assertions WHERE state='accepted'"), [dict(n=2)])

    def test_unfixed_scripted_response_stays_pending_and_exports_reason(self):
        self.h.model.permanent = True
        self.assertEqual(self.run_cli(), 2)
        report = self.h.report()
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(self.h.query("SELECT COUNT(*) AS n FROM fact_assertions WHERE state='accepted'"), [dict(n=0)])
        self.assertEqual(len(self.h.model.relationship_requests), 2)
        self.assertIn('validation_feedback', self.h.model.relationship_requests[1]['tasks'][0])
        self.assertIn('span_not_visible', (self.h.user_output / 'index.html').read_text(encoding='utf-8'))
        self.assertTrue((self.h.output / 'index_report.html').is_file())
        self.assertEqual(self.h.query('PRAGMA foreign_key_check'), [])


if __name__ == '__main__':
    unittest.main()
