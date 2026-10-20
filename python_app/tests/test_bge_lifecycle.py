"""服务生命周期离线回归，不加载权重或调用远端模型。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import bge_lifecycle as life


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'service.json'
        self.cfg = dict(port=8766, model_path=self.temp.name, python='unused', token='test')
        self.path.write_text(json.dumps(self.cfg), encoding='utf-8')
        self.options = dict(port=8766, service_config=str(self.path))
        self.running = True
        self.client = Mock(info=dict(status='ready', model_path=self.temp.name, chunk_tokens=512, dimension=1024))
        self.client.request.side_effect = self.shutdown
        self.addCleanup(patch.stopall)
        self.probe = patch.object(life, 'probe', side_effect=lambda _: self.client if self.running else None).start()
        self.launch = patch.object(life, 'launch', side_effect=self.start).start()
        patch.object(life.time, 'sleep').start()

    def shutdown(self, *args):
        self.running = False

    def start(self, *args):
        self.running = True
        return Mock(poll=Mock(return_value=None))

    def test_reuse_and_stop(self):
        with life.managed_encoder(self.options):
            self.assertTrue(self.running)
        self.assertFalse(self.running)
        self.launch.assert_not_called()

    def test_auto_start(self):
        self.running = False
        with life.managed_encoder(self.options):
            self.assertTrue(self.running)
        self.launch.assert_called_once()
        self.assertFalse(self.running)

    def test_shared_lease(self):
        with life.managed_encoder(self.options):
            with life.managed_encoder(self.options):
                pass
            self.assertTrue(self.running)
            self.client.request.assert_not_called()
        self.client.request.assert_called_once_with('/shutdown', {})

    def test_exception_and_interrupt_cleanup(self):
        for error in (ValueError('failure'), KeyboardInterrupt()):
            self.running = True
            with self.assertRaises(type(error)):
                with life.managed_encoder(self.options):
                    raise error
            self.assertFalse(self.running)

    def test_start_failure_never_enters_batch(self):
        self.running = False
        self.launch.side_effect = None
        self.launch.return_value = Mock(poll=Mock(return_value=1))
        with self.assertRaisesRegex(RuntimeError, '退出'):
            with life.managed_encoder(self.options):
                self.fail('must not enter batch')

    def test_mismatch_does_not_stop_unrelated_model(self):
        self.client.info['chunk_tokens'] = 256
        with self.assertLogs(level='ERROR'):
            with self.assertRaisesRegex(RuntimeError, '不匹配'):
                with life.managed_encoder(self.options):
                    self.fail('must not enter batch')
        self.client.request.assert_not_called()

    def test_stale_lease_removed(self):
        folder = self.path.parent / 'service-leases'
        folder.mkdir()
        stale = folder / 'old.lease'
        stale.write_text('0', encoding='utf-8')
        with life.managed_encoder(self.options):
            self.assertFalse(stale.exists())
        self.assertFalse(self.running)

    def test_entry_wraps_partial_result_and_prepare_skips_service(self):
        import memory_agent
        cfg = dict(flow_version='generic-flow-v3-relations', retrieval=self.options)
        self.path.write_text(json.dumps(cfg), encoding='utf-8')
        from contextlib import contextmanager
        events = []
        @contextmanager
        def managed(_):
            events.append('enter')
            try:
                yield
            finally:
                events.append('exit')
        with patch.object(life, 'managed_encoder', managed), patch.object(memory_agent, 'run_dispatch', return_value=2):
            self.assertEqual(memory_agent.run(self.path, 'live'), 2)
            self.assertEqual(events, ['enter', 'exit'])
            memory_agent.run(self.path, 'prepare')
            self.assertEqual(events, ['enter', 'exit'])


if __name__ == '__main__':
    unittest.main()
