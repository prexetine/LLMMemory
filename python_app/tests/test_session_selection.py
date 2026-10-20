"""精确会话测试：名单错误不得退回用户全量。"""
import unittest
from memory_agent import configured_sessions


class SessionSelectionTests(unittest.TestCase):
    def test_exact_selection_preserves_input_order(self):
        ss=[dict(session_id=x) for x in ('a','b','c')]
        cfg=dict(max_sessions='all',session_ids_by_device={'u':['c','a']})
        self.assertEqual(configured_sessions(ss,cfg,'u'),[ss[0],ss[2]])

    def test_missing_user_or_session_fails_closed(self):
        for selection in ({'v':['a']},{'u':['missing']},{'u':[]},{'u':['a','a']},{'u':'a'}):
            with self.subTest(selection=selection),self.assertRaises(ValueError):
                configured_sessions([dict(session_id='a')],dict(max_sessions='all',session_ids_by_device=selection),'u')

    def test_existing_selection_unchanged(self):
        ss=[dict(session_id=x) for x in ('a','b')]
        self.assertEqual(configured_sessions(ss,dict(max_sessions='all'),'u'),ss)
        self.assertEqual(configured_sessions(ss,dict(max_sessions=1),'u'),ss[:1])
