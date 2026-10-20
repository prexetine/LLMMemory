"""不调用外网的Agent契约、历史与存储测试。"""
import json
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_data import Registry, digest, exact_keys
from agent_store import MemoryStore
from agent_llm import JsonClient
from agent_schema import response_schema
from agent_review import CHECKS, REVIEW_VERSION, make_record
from validate_agent_run import validate as validate_run_artifacts
from legacy_memory_agent import BASE, CentralAgent, validate_candidate, validate_audit, run

REGISTRY = Registry(BASE / "J20核心用户记忆维度与提取存储规划.md")


def session(sid="s1", text="我喜欢恐龙", start="2026-01-01T10:00:00+08:00"):
    turn = {"evidence_id": sid + ":1", "user_text": text, "assistant_text": "你叫小明",
            "occurred_at": start, "source_file": "test", "source_row": 1, "sheet": "test"}
    return {"device_id": "test-user", "session_id": sid, "input_hash": digest(turn),
            "start": start, "end": start, "turns": [turn]}


def candidate(s=None):
    s = s or session()
    return {"dimension_id": "G16", "subject": "当前发言者", "claim": "用户自述喜欢恐龙",
            "scope": "自述偏好", "kind": "self_report", "keywords": ["恐龙", "喜欢"],
            "evidence": [{"evidence_id": s["turns"][0]["evidence_id"], "quote": "我喜欢恐龙"}],
            "operation": "new", "target_memory_id": ""}


def audit(candidates):
    dims = {c["dimension_id"] for c in candidates}
    return {"approved_candidate_ids": [c["candidate_id"] for c in candidates], "rejected": [],
            "followup_skills": [], "dimension_statuses": {
                "with_candidate": sorted(dims), "no_evidence": sorted(set(REGISTRY.dimensions) - dims),
                "insufficient": [], "missing_source": [], "not_assessed": []}, "notes": "测试桩，不证明真实效果"}


def review_result(c):
    return {"decision": "approve", "checks": dict.fromkeys(CHECKS, "pass"),
            "reason": "合成审核通过，不证明真实模型质量",
            "evidence_ids": [e["evidence_id"] for e in c["evidence"]]}


def commit_with_review(store, s, candidates, coverage, version, status, history):
    records = [make_record(c, s, history, version, review_result(c), "synthetic_test")
               for c in candidates]
    return store.commit(s, candidates, coverage, version, status, history, records)


class FakeClient:
    def __init__(self):
        self.n = 0

    def ask(self, role, instruction, data):
        if role == "reviewer":
            return review_result(data["candidate"])
        if role == "router":
            return {"skills": ["interests_preferences"], "reason": "合成偏好样例"}
        if role == "planner":
            self.n += 1
            return {"action": "execute_skill" if self.n == 1 else "finish",
                    "skill_id": "interests_preferences" if self.n == 1 else "",
                    "query": "", "reason": "测试"}
        if role.startswith("extractor:"):
            return {"candidates": [candidate(data["session"])], "notes": "测试"}
        return audit(data["candidates"])


class AgentTests(unittest.TestCase):
    def test_fixed_dimension_map_normalizes_without_inventing_status(self):
        a = audit([])
        a['dimension_statuses'] = {d: 'no_evidence' for d in REGISTRY.dimensions}
        validate_audit(a, [], REGISTRY)
        self.assertEqual(set(a['dimension_statuses']['no_evidence']), set(REGISTRY.dimensions))
        schema = response_schema('auditor', {'catalog': REGISTRY.catalog()})
        self.assertEqual(len(schema['properties']['dimension_statuses']['required']), 106)
        self.assertFalse(schema['properties']['dimension_statuses']['additionalProperties'])

    def test_resume_skips_completed_sessions_and_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = json.loads((BASE / 'python_app/config/memory_agent.json').read_text(encoding='utf-8'))
            config.update(device_id='test-user', max_sessions=1)
            config.pop('user_selection', None)  # 旧单用户配置兼容测试
            cp = folder / 'config.json'
            cp.write_text(json.dumps(config), encoding='utf-8')
            s = session()
            with patch('legacy_memory_agent.load_sessions', return_value=([s], {'device_id': 'test-user'})), patch('legacy_memory_agent.JsonClient', return_value=FakeClient()), redirect_stdout(io.StringIO()):
                self.assertEqual(run(cp, 'live', str(folder / 'out')), 0)
            # 已完成会话不能再次调用模型；object没有ask方法，若误调用测试会失败。
            with patch('legacy_memory_agent.load_sessions', return_value=([s], {'device_id': 'test-user'})), patch('legacy_memory_agent.JsonClient', return_value=object()), redirect_stdout(io.StringIO()):
                self.assertEqual(run(cp, 'live', str(folder / 'out'), True), 0)
            s['input_hash'] = 'changed'
            with patch('legacy_memory_agent.load_sessions', return_value=([s], {'device_id': 'test-user'})):
                with self.assertRaises(ValueError):
                    run(cp, 'live', str(folder / 'out'), True)

    def test_independent_artifact_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            store = MemoryStore(folder / 'memory.sqlite3')
            s1 = session()
            c = validate_candidate(candidate(), 'interests_preferences', REGISTRY, s1, [])
            receipt = commit_with_review(store, s1, [c], audit([c]), REGISTRY.version, 'completed', [])
            s2 = session('s2', start='2026-01-02T10:00:00+08:00')
            commit_with_review(store, s2, [], audit([]), REGISTRY.version, 'completed', [])
            store.export_book(folder / '记忆书页.md', REGISTRY)
            store.close()
            files = {
                'report.json': {'status': 'completed'},
                'manifest.json': {'device_id': 'test-user', 'selected_sessions': [s1, s2],
                                  'review_version': REVIEW_VERSION},
                'trace.jsonl': {'session_id': 's2', 'history_bootstrap': {
                    'count': 1, 'memory_ids': [receipt['mapping'][0]['memory_id']]}},
                'model_calls.jsonl': {'provider': 'Doubao', 'synthetic_fixture': True},
            }
            for name, value in files.items():
                (folder / name).write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')
            result = validate_run_artifacts(folder)
            self.assertTrue(result['chain_passed'])
            self.assertEqual(result['review_gate'], 'checked')
            self.assertEqual(result['sessions_reading_prior_memory'], 1)

    def test_remote_schema_closes_objects_and_limits_dimensions(self):
        schema = response_schema('extractor:interests_preferences',
                                 {'skill_rules': REGISTRY.load('interests_preferences')})
        self.assertFalse(schema['additionalProperties'])
        item = schema['properties']['candidates']['items']
        self.assertFalse(item['additionalProperties'])
        self.assertIn('G16', item['properties']['dimension_id']['enum'])
        self.assertNotIn('G09', item['properties']['dimension_id']['enum'])
        self.assertFalse(item['properties']['evidence']['items']['additionalProperties'])

    def test_pending_skill_blocks_premature_finish(self):
        class PrematureClient(FakeClient):
            def ask(self, role, instruction, data):
                if role == 'planner':
                    return {'action': 'finish', 'skill_id': '', 'query': '', 'reason': '历史已回答'}
                return super().ask(role, instruction, data)
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / 'test.sqlite3')
            agent = CentralAgent(PrematureClient(), REGISTRY, store,
                                 {'max_agent_steps': 2, 'max_audit_rounds': 2, 'history_limit': 10},
                                 Path(tmp) / 'trace.jsonl')
            result = agent.run_session(session())
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(len(result['approved']), 1)
            self.assertEqual(result['actions'][0]['action'], 'execute_skill')
            store.close()

    def test_control_character_repair_preserves_schema_validation(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return json.dumps({'choices': [{'message': {'content': '{"notes":"line1\nline2"}'},
                                                'finish_reason': 'stop'}], 'usage': {}}).encode('utf-8')
        class Opener:
            def open(self, *args, **kwargs): return Response()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.yaml'
            p.write_text('Doubao:\n  api_key: synthetic-test-key\n  base_url: https://ark.cn-beijing.volces.com/api/v3\n  model_map:\n    Doubao-Test: test\n', encoding='utf-8')
            cfg = {'provider': 'Doubao', 'model_alias': 'Doubao-Test', 'max_calls': 1,
                   'max_request_chars': 100, 'max_total_input_chars': 100, 'max_output_tokens': 100,
                   'timeout_seconds': 1}
            log = Path(tmp) / 'log'
            client = JsonClient(cfg, p, log)
            client.opener = Opener()
            result = client.ask('test', 'test', {})
            self.assertEqual(result['notes'], 'line1\nline2')
            record = json.loads(log.read_text(encoding='utf-8'))
            self.assertEqual(record['json_status'], 'valid')
            self.assertNotIn('error', record)
            with self.assertRaises(ValueError):
                exact_keys(result, {'notes', 'candidates'})

    def test_model_provider_and_alias_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.yaml'
            p.write_text('Doubao:\n  api_key: synthetic-test-key\n  base_url: https://ark.cn-beijing.volces.com/api/v3\n  model_map:\n    GLM5.2: forbidden\n', encoding='utf-8')
            for provider, alias in [('Other', 'Doubao-Seed'), ('Doubao', 'GLM5.2')]:
                with self.assertRaises(ValueError):
                    JsonClient({'provider': provider, 'model_alias': alias}, p, Path(tmp) / 'log')

    def test_budget_blocks_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.yaml'
            p.write_text('Doubao:\n  api_key: synthetic-test-key\n  base_url: https://ark.cn-beijing.volces.com/api/v3\n  model_map:\n    Doubao-Test: test\n', encoding='utf-8')
            config = {'provider': 'Doubao', 'model_alias': 'Doubao-Test', 'max_calls': 0, 'max_request_chars': 10}
            client = JsonClient(config, p, Path(tmp) / 'log')
            with self.assertRaises(RuntimeError):
                client.ask('test', 'test', {})

    def test_legacy_char_limit_ignored_but_context_budget_enforced(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return json.dumps({'choices': [{'message': {'content': '{"ok":true}'},
                                                'finish_reason': 'stop'}], 'usage': {}}).encode('utf-8')
        class Opener:
            def open(self, *args, **kwargs): return Response()
        with tempfile.TemporaryDirectory() as tmp:
            client = JsonClient.__new__(JsonClient)
            client.config = {'provider': 'Doubao', 'max_calls': None,
                             'max_total_input_chars': None, 'max_request_chars': 100,
                             'max_output_tokens': 100, 'timeout_seconds': 1}
            client.calls, client.input_chars, client.total_tokens = 9999, 99999999, 0
            client.elapsed = 0
            client.url, client.key, client.model = 'https://example.invalid/chat', 'test', 'test'
            client.log_path, client.opener = Path(tmp) / 'log', Opener()
            self.assertEqual(client.ask('test', 'test', {}), {'ok': True})
            self.assertEqual(client.calls, 10000)
            from unittest.mock import patch
            for legacy_value in (100, None, 'absent'):
                if legacy_value == 'absent':
                    client.config.pop('max_request_chars', None)
                else:
                    client.config['max_request_chars'] = legacy_value
                data = {'text': '长' * 1001}
                with patch.object(client.opener, 'open', wraps=client.opener.open) as opened:
                    self.assertEqual(client.ask('test', 'test', data), {'ok': True})
                    payload = json.loads(opened.call_args.args[0].data.decode('utf-8'))
                    self.assertEqual(json.loads(payload['messages'][1]['content']), data)
                record = json.loads(client.log_path.read_text(encoding='utf-8').splitlines()[-1])
                self.assertGreater(record['input_chars'], 100)
            from agent_context import ContextCapacityError
            with patch.object(client.opener, 'open') as opened:
                with self.assertRaises(ContextCapacityError):
                    client.ask('test', 'test', {'text': '长' * 100001})
                opened.assert_not_called()
            client.config['max_total_input_chars'] = client.input_chars
            with self.assertRaisesRegex(RuntimeError, '累计输入预算不足'):
                client.ask('test', 'test', {})

    def test_revision_refuses_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / 'test.sqlite3')
            s = session()
            commit_with_review(store, s, [], audit([]), REGISTRY.version, 'completed', [])
            s['input_hash'] = 'changed'
            with self.assertRaises(ValueError):
                commit_with_review(store, s, [], audit([]), REGISTRY.version, 'completed', [])
            store.close()

    def test_partial_never_submits_unreviewed_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / 'test.sqlite3')
            agent = CentralAgent(FakeClient(), REGISTRY, store,
                                 {'max_agent_steps': 1, 'max_audit_rounds': 2, 'history_limit': 10},
                                 Path(tmp) / 'trace.jsonl')
            result = agent.run_session(session())
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(len(result['candidates']), 1)
            self.assertEqual(result['approved'], [])
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0], 0)
            store.close()

    def test_registry_unique_106(self):
        self.assertEqual(len(REGISTRY.dimensions), 106)
        self.assertEqual(sum(len(x["dimensions"]) for x in REGISTRY.catalog()), 106)
        self.assertEqual(REGISTRY.dimensions["E24"]["owner_skill_id"], "memory_control")

    def test_reject_extra_dynamic_key(self):
        c = candidate()
        c["invented"] = True
        with self.assertRaises(ValueError):
            validate_candidate(c, "interests_preferences", REGISTRY, session(), [])

    def test_reject_assistant_as_evidence(self):
        c = candidate()
        c["evidence"][0]["quote"] = "你叫小明"
        with self.assertRaises(ValueError):
            validate_candidate(c, "interests_preferences", REGISTRY, session(), [])

    def test_reject_cross_user_evidence(self):
        c = candidate()
        c["evidence"][0]["evidence_id"] = "other-user:1"
        with self.assertRaises(ValueError):
            validate_candidate(c, "interests_preferences", REGISTRY, session(), [])

    def test_reject_wrong_skill(self):
        with self.assertRaises(ValueError):
            validate_candidate(candidate(), "events_plans", REGISTRY, session(), [])

    def test_audit_requires_full_coverage(self):
        a = audit([])
        a["dimension_statuses"]["no_evidence"].pop()
        with self.assertRaises(ValueError):
            validate_audit(a, [], REGISTRY)

    def test_store_idempotency_history_isolation_and_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "test.sqlite3")
            s = session()
            c = validate_candidate(candidate(), "interests_preferences", REGISTRY, s, [])
            r = commit_with_review(store, s, [c], audit([c]), REGISTRY.version, "completed", [])
            again = commit_with_review(store, s, [c], audit([c]), REGISTRY.version, "completed", [])
            self.assertEqual(again["status"], "already_committed")
            self.assertEqual(store.history("other-user", "2027", "", 10), [])
            self.assertEqual(store.history("test-user", s["start"], "", 10), [])
            self.assertEqual(len(store.history("test-user", "2027", "恐龙", 10)), 1)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0], 1)
            store.export_book(Path(tmp) / "book.md", REGISTRY)
            self.assertEqual((Path(tmp) / "book.md").read_text(encoding="utf-8").count('<a id="'), 106)
            store.close()

    def test_reinforcement_and_conflict_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "test.sqlite3")
            s = session()
            c = validate_candidate(candidate(), "interests_preferences", REGISTRY, s, [])
            commit_with_review(store, s, [c], audit([c]), REGISTRY.version, "completed", [])
            history = store.history("test-user", "2027", "", 10)
            s2 = session("s2", start="2026-01-02T10:00:00+08:00")
            c2 = candidate(s2)
            c2.update(operation="reinforce", target_memory_id=history[0]["memory_id"])
            c2 = validate_candidate(c2, "interests_preferences", REGISTRY, s2, history)
            commit_with_review(store, s2, [c2], audit([c2]), REGISTRY.version, "completed", history)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 1)
            s3 = session("s3", start="2026-01-03T10:00:00+08:00")
            with self.assertRaises(ValueError):
                commit_with_review(store, s3, [c2], audit([c2]), REGISTRY.version, "completed", history)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 2)
            store.close()

    def test_control_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "test.sqlite3")
            agent = CentralAgent(FakeClient(), REGISTRY, store,
                                 {"max_agent_steps": 6, "max_audit_rounds": 2, "history_limit": 10},
                                 Path(tmp) / "trace.jsonl")
            result = agent.run_session(session())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["approved"]), 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
