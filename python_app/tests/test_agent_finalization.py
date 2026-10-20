"""有界收尾审查与重复调度回归，禁止通过清空待办伪造完成。"""
from pathlib import Path
import tempfile
import unittest

from test_memory_agent import FakeClient, REGISTRY, session, audit
from legacy_memory_agent import CentralAgent
from agent_context import compact_feedback
from agent_store import MemoryStore


class FollowupClient(FakeClient):
    def __init__(self, persistent=False):
        super().__init__()
        self.audits = 0
        self.persistent = persistent
        self.final_feedback = None

    def ask(self, role, instruction, data):
        if role == 'planner':
            return {'action': 'finish', 'skill_id': '', 'query': '', 'reason': 'done'}
        if role == 'auditor':
            self.audits += 1
            self.final_feedback = data.get('execution_feedback')
            result = audit(data['candidates'])
            if self.audits == 1 or self.persistent:
                result['followup_skills'] = ['interests_preferences']
            return result
        return super().ask(role, instruction, data)


class FinalizationTests(unittest.TestCase):
    def run_agent(self, client):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store = MemoryStore(Path(temp.name) / 'memory.sqlite3')
        self.addCleanup(store.close)
        agent = CentralAgent(client, REGISTRY, store,
                             {'max_agent_steps': 4, 'max_audit_rounds': 1, 'history_limit': 5},
                             Path(temp.name) / 'trace.jsonl')
        return agent.run_session(session()), store

    def test_last_followup_gets_final_audit_and_item_review(self):
        client = FollowupClient()
        result, store = self.run_agent(client)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(client.audits, 2)
        self.assertTrue(client.final_feedback['finalization'])
        self.assertTrue(result['reviews'])
        self.assertEqual(store.db.execute('select count(*) from sessions').fetchone()[0], 1)

    def test_unresolved_followup_remains_partial(self):
        result, store = self.run_agent(FollowupClient(persistent=True))
        self.assertEqual(result['status'], 'partial')
        self.assertFalse(result['approved'])
        self.assertEqual(store.db.execute('select count(*) from sessions').fetchone()[0], 0)

    def test_feedback_reasons_bounded(self):
        data = audit([])
        data['rejected'] = [{'candidate_id': str(i), 'reason': '错' * 1000} for i in range(20)]
        result = compact_feedback(data)
        self.assertEqual(len(result['rejected']), 8)
        self.assertTrue(all(len(r['reason']) == 240 for r in result['rejected']))

    def test_repeated_planner_skill_transitions_to_audit(self):
        class Repeating(FakeClient):
            def ask(self, role, instruction, data):
                if role == 'planner':
                    return {'action': 'execute_skill', 'skill_id': 'interests_preferences',
                            'query': '', 'reason': 'repeat'}
                return super().ask(role, instruction, data)
        result, _ = self.run_agent(Repeating())
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['reviews'])

    def test_invalid_final_audit_never_commits(self):
        class InvalidFinal(FollowupClient):
            def ask(self, role, instruction, data):
                result = super().ask(role, instruction, data)
                if role == 'auditor' and self.audits >= 2:
                    result['unexpected'] = True
                return result
        result, store = self.run_agent(InvalidFinal())
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(store.db.execute('select count(*) from sessions').fetchone()[0], 0)

    def test_invalid_final_audit_can_be_corrected_by_model(self):
        class Corrected(FollowupClient):
            def ask(self, role, instruction, data):
                result = super().ask(role, instruction, data)
                if role == 'auditor' and self.audits == 2:
                    result['unexpected'] = True
                if role == 'auditor' and self.audits == 3:
                    assert 'validation_feedback' in data
                return result
        client = Corrected()
        result, _ = self.run_agent(client)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(client.audits, 3)
        self.assertTrue(result['reviews'])


if __name__ == '__main__':
    unittest.main()
