"""增量索引离线回归；编码器为确定性Mock，无本地模型或网络请求。"""
import copy
import sqlite3
import unittest
from collections import Counter

from agent_retrieval import Retriever, RetrievalError, text_for
import test_unified_relations as u
import test_source_agent_v4 as f


class Encoder:
    signature='offline-v1'
    dimension=2
    def __init__(self):
        self.calls=[]
        self.fail=False
    def encode(self,texts):
        self.calls.append(list(texts))
        if self.fail:raise RetrievalError('injected unavailable')
        return [[1.0,0.0] for _ in texts]


def memory(mid,device='u'):
    return dict(memory_id=mid,revision=1,state='reviewed',device_id=device,subject_key='self',
                claim=mid,subject='我',scope='all_time',fact={'object':mid},evidence=[],origin_session_id='s')


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(':memory:');self.addCleanup(self.db.close)
        self.db.execute('CREATE TABLE vector_queue(memory_id TEXT PRIMARY KEY,revision INTEGER,text_hash TEXT,status TEXT)')
        self.encoder=Encoder();self.stats=Counter()
        self.r=Retriever(self.db,{'top_k':20},self.stats,self.encoder)
        self.state=dict(memories={str(i):memory(str(i)) for i in range(3)},groups={})
        self.c=dict(memory('query'),candidate_id='c')

    def recall(self):
        return self.r.recall([self.c],self.state)

    def test_warm_unchanged_reads_no_vectors_and_writes_no_index(self):
        expected=self.recall();self.encoder.calls.clear();before=self.stats.copy();sql=[]
        self.db.set_trace_callback(sql.append)
        actual=self.recall()
        self.assertEqual(actual,expected);self.assertFalse(self.encoder.calls)
        self.assertEqual(self.stats['embedding_cache_reads'],before['embedding_cache_reads'])
        self.assertEqual(self.stats['index_upserts'],before['index_upserts'])
        self.assertFalse(any('DELETE FROM dense_index' in q or 'INSERT INTO dense_index' in q or 'UPDATE vector_queue' in q for q in sql))

    def test_new_memory_only_encodes_one(self):
        self.recall();self.encoder.calls.clear();before=self.stats.copy()
        self.state['memories']['new']=memory('new');self.recall()
        self.assertEqual(sum(map(len,self.encoder.calls)),1)
        self.assertEqual(self.stats['index_upserts']-before['index_upserts'],1)

    def test_revision_only_reuses_vector(self):
        self.recall();self.encoder.calls.clear()
        self.state['memories']['1']['revision']=2
        self.recall();self.assertFalse(self.encoder.calls)
        self.assertEqual(self.db.execute("SELECT revision FROM dense_index WHERE memory_id='1'").fetchone()[0],2)

    def test_content_change_even_without_revision_is_detected(self):
        self.recall();self.encoder.calls.clear()
        self.state['memories']['1']['claim']='changed'
        self.recall();self.assertEqual(sum(map(len,self.encoder.calls)),1)

    def test_inactive_removed_without_rebuilding_others(self):
        self.recall();before=self.stats.copy();self.encoder.calls.clear()
        self.state['memories']['1']['state']='superseded'
        rows=self.recall()['c']
        self.assertNotIn('1',[m['memory_id'] for m in rows]);self.assertFalse(self.encoder.calls)
        self.assertEqual(self.stats['index_upserts'],before['index_upserts'])
        self.assertEqual(self.stats['index_removed']-before['index_removed'],1)
        self.assertEqual(len(self.r._resident),3)

    def test_corrupt_warm_disk_cache_invalidates_resident(self):
        self.recall();self.encoder.calls.clear()
        key=self.db.execute("SELECT cache_key FROM dense_index WHERE memory_id='1'").fetchone()[0]
        self.db.execute('UPDATE dense_cache SET data=? WHERE cache_key=?',('[0,0]',key));self.db.commit()
        self.recall();self.assertEqual(sum(map(len,self.encoder.calls)),1)
        self.assertEqual(self.stats['embedding_cache_invalidated'],1)

    def test_restart_loads_once_without_rebuilding_or_encoding(self):
        self.recall();self.encoder.calls.clear();before=self.stats['index_upserts']
        self.r=Retriever(self.db,{'top_k':20},self.stats,self.encoder)
        self.recall();self.assertFalse(self.encoder.calls)
        self.assertEqual(self.stats['index_upserts'],before)
        reads=self.stats['embedding_cache_reads'];self.recall()
        self.assertEqual(self.stats['embedding_cache_reads'],reads)

    def test_encoder_failure_keeps_previous_index_and_retry_recovers(self):
        self.recall();before=self.db.execute('SELECT * FROM dense_index ORDER BY memory_id').fetchall()
        self.state['memories']['new']=memory('new');self.encoder.fail=True
        with self.assertRaises(RetrievalError):self.recall()
        self.assertEqual(before,self.db.execute('SELECT * FROM dense_index ORDER BY memory_id').fetchall())
        self.encoder.fail=False;self.recall()
        self.assertEqual(self.db.execute('SELECT count(*) FROM dense_index').fetchone()[0],4)

    def test_model_signature_change_reencodes_once(self):
        self.recall();self.encoder.calls.clear();self.encoder.signature='offline-v2'
        self.recall();self.assertEqual(sum(map(len,self.encoder.calls)),4)
        self.encoder.calls.clear();self.recall();self.assertFalse(self.encoder.calls)

    def test_index_write_failure_rolls_back_and_reuses_encoded_vectors(self):
        self.recall();before=self.db.execute('SELECT * FROM dense_index ORDER BY memory_id').fetchall()
        self.state['memories'].update(a=memory('a'),b=memory('b'))
        self.db.execute("CREATE TRIGGER fail_index BEFORE INSERT ON dense_index WHEN NEW.memory_id='b' BEGIN SELECT RAISE(ABORT,'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.recall()
        self.assertEqual(before,self.db.execute('SELECT * FROM dense_index ORDER BY memory_id').fetchall())
        self.db.execute('DROP TRIGGER fail_index');self.encoder.calls.clear()
        self.recall();self.assertFalse(self.encoder.calls)
        self.assertEqual(self.db.execute('SELECT count(*) FROM dense_index').fetchone()[0],5)

    def test_other_device_index_preserved(self):
        self.db.execute("INSERT INTO dense_index VALUES('other','other-id',1,'other-key')");self.db.commit()
        self.state['memories']['other-id']=memory('other-id','other')
        self.assertNotIn('other-id',[m['memory_id'] for m in self.recall()['c']])
        self.assertEqual(self.db.execute("SELECT cache_key FROM dense_index WHERE device='other'").fetchone()[0],'other-key')


class PipelineIncrementalTests(unittest.TestCase):
    def test_retriever_lives_across_sessions_and_queue_not_reset(self):
        base=u.UnifiedTests();base.setUp();self.addCleanup(base.doCleanups)
        base.run_session(f.session('s1',text='我喜欢恐龙'))
        first=base.h.agent.retriever
        s=f.session('s2',text='我喜欢汽车')
        c=u.g.raw(s,'喜欢汽车');c['fact'].update(object='汽车',value='汽车')
        base.run_session(s,[c])
        self.assertIs(base.h.agent.retriever,first)
        state=base.h.store.db.execute('SELECT status FROM vector_queue').fetchall()
        self.assertTrue(any(s[0]=='indexed' for s in state))


def benchmark():
    """固定256条/1024维合成向量，比较冷加载和驻留增量；不代表BGE推理耗时。"""
    import time
    import tracemalloc
    import statistics
    import json
    case=IncrementalTests();case.setUp()
    try:
        case.state['memories']={str(i):memory(str(i)) for i in range(256)}
        case.encoder.dimension=1024
        case.encoder.encode=lambda texts:[[1.0]+[0.0]*1023 for _ in texts]
        case.recall()
        case.r=Retriever(case.db,{'top_k':20},case.stats,case.encoder)
        tracemalloc.start();started=time.perf_counter()
        case.recall();cold=time.perf_counter()-started
        case.stats.clear();times=[]
        for _ in range(5):
            started=time.perf_counter();case.recall();times.append(time.perf_counter()-started)
        peak=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
        print(json.dumps(dict(records=256,dimension=1024,samples=5,cold_seconds=cold,
            warm_mean_seconds=statistics.mean(times),warm_min_seconds=min(times),warm_max_seconds=max(times),
            peak_python_bytes=peak,warm_cache_reads=case.stats['embedding_cache_reads'],
            warm_index_upserts=case.stats['index_upserts'],warm_encoder_calls=case.stats['embedding_requests']),ensure_ascii=False))
    finally:case.doCleanups()


if __name__=='__main__':
    import sys
    if '--benchmark' in sys.argv:benchmark()
    else:unittest.main()
