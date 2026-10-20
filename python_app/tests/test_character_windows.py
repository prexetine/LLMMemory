"""字符分窗不混用UTF-8字节，保留完整正文及完整请求门禁。"""
import unittest
from agent_context import windows, check_request, ContextCapacityError
from agent_flow_data import validate_flow


class CharacterWindowTests(unittest.TestCase):
    def session(self, text):
        return dict(device_id='u',session_id='s',start='',end='',input_hash='x',
                    turns=[dict(evidence_id='e',user_text=text,assistant_text='')])

    def test_chinese_characters_not_bytes(self):
        s=self.session('中'*100)
        self.assertEqual(len(list(windows(s,100,unit='chars'))),1)
        self.assertGreater(len(list(windows(s,500))),1)

    def test_oversized_turn_preserves_text(self):
        text='甲乙丙丁'*30
        ws=list(windows(self.session(text),32,unit='chars'))
        self.assertEqual(''.join(t['user_text'] for w in ws for t in w['turns']),text)
        self.assertTrue(all(sum(len(t['user_text'])+len(t['assistant_text']) for t in w['turns'])<=32 for w in ws))

    def test_invalid_budget(self):
        for value in (0,-1,True,'32768',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_flow(dict(session_chunk_chars=value))

    def test_request_guard_still_applies(self):
        with self.assertRaises(ContextCapacityError):
            check_request(dict(context_window_tokens=30000), 'source_router', '中'*20000, {})
