"""输出上限可省略；输入容量防护不取消。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from agent_llm import JsonClient
from agent_context import input_budget


class OptionalOutputLimitTests(unittest.TestCase):
    def request_payload(self, extra):
        with tempfile.TemporaryDirectory() as folder:
            client = JsonClient.__new__(JsonClient)
            client.config = dict(provider='Doubao', max_calls=None, max_total_input_chars=None,
                                 timeout_seconds=60, **extra)
            client.calls = client.input_chars = client.total_tokens = 0
            client.elapsed = 0.0
            client.model, client.url, client.key = 'test', 'https://example.invalid', 'test'
            client.log_path = Path(folder) / 'calls.jsonl'
            response = Mock()
            response.read.return_value = json.dumps({'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}).encode('utf-8')
            client.opener = Mock()
            client.opener.open.return_value.__enter__ = Mock(return_value=response)
            client.opener.open.return_value.__exit__ = Mock(return_value=False)
            client._ask_once('test', 'test', {})
            return json.loads(client.opener.open.call_args.args[0].data)

    def test_absent_omitted_from_request(self):
        self.assertNotIn('max_tokens', self.request_payload({}))

    def test_explicit_limit_preserved(self):
        self.assertEqual(self.request_payload({'max_output_tokens': 12000})['max_tokens'], 12000)

    def test_input_reserve_unchanged(self):
        self.assertEqual(input_budget({}), input_budget({'max_output_tokens': 12000}))


if __name__ == '__main__':
    unittest.main()
