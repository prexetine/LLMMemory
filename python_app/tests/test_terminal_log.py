"""终端日志追加、即时落盘和嵌套用户隔离，不请求模型。"""
import io
import logging
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_terminal_log import terminal_log


class TerminalLogTests(unittest.TestCase):
    def test_stdout_stderr_logging_immediate_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / 'logs/terminal.log'
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                for n in range(2):
                    with terminal_log(folder, f'用户=小明 resume={bool(n)}'):
                        print('中文报告', n)
                        sys.stderr.write('错误详情\n')
                        logging.info('测试模型响应token=42')
                        text = path.read_text(encoding='utf-8')
                        self.assertIn('测试模型响应token=42', text)
                        self.assertIn('中文报告 ' + str(n), text)
            text = path.read_text(encoding='utf-8')
            self.assertEqual(text.count('=== START'), 2)
            self.assertEqual(text.count('测试模型响应token=42'), 2)
            self.assertIn('错误详情', text)

    def test_exception_trace_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = (sys.stdout, sys.stderr, list(logging.getLogger().handlers), logging.getLogger().level)
            with self.assertRaisesRegex(ValueError, '故障注入'):
                with terminal_log(Path(tmp), 'exception-test'):
                    raise ValueError('故障注入')
            self.assertEqual((sys.stdout, sys.stderr, list(logging.getLogger().handlers), logging.getLogger().level), before)
            text = (Path(tmp) / 'logs/terminal.log').read_text(encoding='utf-8')
            self.assertIn('Traceback', text)
            self.assertIn('ValueError: 故障注入', text)

    def test_nested_users_no_cross_user_leak_or_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(tmp)
            with terminal_log(root, 'batch'):
                for user in ['user-a', 'user-b']:
                    with terminal_log(root / user, user):
                        with terminal_log(root / user, 'same-path'):
                            logging.info('独立内容 %s', user)
                            print('报告', user)
            for user, other in [('user-a', 'user-b'), ('user-b', 'user-a')]:
                text = (root / user / 'logs/terminal.log').read_text(encoding='utf-8')
                self.assertEqual(text.count('独立内容 ' + user), 1)
                self.assertNotIn(other, text)


if __name__ == '__main__':
    unittest.main()
