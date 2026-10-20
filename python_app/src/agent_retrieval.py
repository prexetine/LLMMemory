"""可重建的本地稠密向量缓存及隔离召回；SQLite记忆仍为真源。"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler
from agent_data import digest

TEMPLATE = 'memory-dense-v1'


class RetrievalError(ValueError):
    code = 'retrieval_unavailable'


def text_for(record: dict) -> str:
    fields = ('subject', 'subject_role', 'attribute_key', 'claim', 'scope', 'kind', 'assertion_mode', 'valid_from', 'valid_to', 'fact')
    return json.dumps({k: record.get(k) for k in fields}, ensure_ascii=False, sort_keys=True)


def compatible(a: dict, b: dict) -> bool:
    if not a.get('subject_key') or a.get('subject_key') != b.get('subject_key') or a.get('device_id') != b.get('device_id'):
        return False
    contexts = {x.get('structure', {}).get('context', 'real') for x in (a, b)}
    if len(contexts) > 1:
        # 同一原话的同一结论可能被不同Skill标成现实/游戏；允许关系模型核对，
        # 不直接合并，不放开故事/图片或不同来源的隔离。
        refs = [{(e['evidence_id'], e.get('quote')) for e in x.get('evidence', [])} for x in (a, b)]
        if not (contexts == {'real', 'game'} and refs[0] and refs[0] == refs[1]
                and a.get('origin_session_id') == b.get('origin_session_id')
                and a.get('claim') and a.get('claim') == b.get('claim')
                and a.get('fact') and a.get('fact') == b.get('fact')
                and all(a.get(k) == b.get(k) for k in ('assertion_mode', 'valid_from', 'valid_to', 'retention'))):
            return False
    # 当次请求、画面观察和暂时状态不跨来源泛化；事件可跨Session召回再判断日期。
    if any(x.get('assertion_mode') in ('request', 'temporary', 'observation') for x in (a, b)):
        return a.get('origin_session_id') == b.get('origin_session_id')
    return True


def structured(a: dict, b: dict) -> bool:
    shared = {e['evidence_id'] for e in a.get('evidence', [])} & {e['evidence_id'] for e in b.get('evidence', [])}
    fa, fb = a.get('fact', {}), b.get('fact', {})
    return bool(shared or (fa.get('object') and fa.get('object') == fb.get('object')
                          and a.get('attribute_key') and a.get('attribute_key') == b.get('attribute_key'))
                or (a.get('event_id') and a.get('event_id') == b.get('event_id')))


def validate_vector(vector, dimension):
    if not isinstance(vector, list) or len(vector) != dimension or any(type(v) not in (float, int) or not math.isfinite(v) for v in vector):
        raise RetrievalError('向量维度或数值非法')
    norm = math.sqrt(sum(v*v for v in vector))
    if abs(norm-1) > 0.001:
        raise RetrievalError('向量未归一或损坏')


class LocalEncoder:
    def __init__(self, config):
        self.config = config
        self.base = 'http://127.0.0.1:' + str(config.get('port', 8766))
        self.opener = build_opener(ProxyHandler({}))
        self.info = self.request('/health')
        if self.info.get('status') != 'ready' or self.info.get('dimension') != 1024:
            raise RetrievalError('BGE健康检查失败')
        self.signature = self.info['model_signature']
        self.dimension = self.info['dimension']

    def request(self, path, payload=None):
        try:
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None
            headers = {'Content-Type': 'application/json'}
            if body is not None:
                cfg = json.loads(Path(self.config['service_config']).read_text(encoding='utf-8'))
                headers['Authorization'] = 'Bearer ' + cfg['token']
            with self.opener.open(Request(self.base+path, data=body, headers=headers), timeout=self.config.get('timeout_seconds', 120)) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as exc:
            raise RetrievalError('本地编码服务不可用：' + type(exc).__name__) from exc

    def encode(self, texts):
        result = self.request('/encode', {'texts': texts})
        if result.get('model_signature') != self.signature or len(result.get('vectors', [])) != len(texts):
            raise RetrievalError('编码服务版本或响应数量变化')
        for v in result['vectors']:
            validate_vector(v, self.dimension)
        return result['vectors']


class Retriever:
    def __init__(self, db, config, stats, encoder=None):
        self.db, self.config, self.stats = db, config, stats
        self.encoder = encoder or LocalEncoder(config)
        db.execute('CREATE TABLE IF NOT EXISTS dense_cache(cache_key TEXT PRIMARY KEY, data TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS dense_index(device TEXT, memory_id TEXT, revision INTEGER, cache_key TEXT, PRIMARY KEY(device,memory_id))')
        # 更新/删除缓存是异常失效路径；正常新增不清空已驻留的历史向量。
        db.execute('CREATE TABLE IF NOT EXISTS dense_cache_epoch(id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL)')
        db.execute('INSERT OR IGNORE INTO dense_cache_epoch VALUES(1,0)')
        for event in ('UPDATE', 'DELETE'):
            db.execute(f'CREATE TRIGGER IF NOT EXISTS dense_cache_{event.lower()}_epoch AFTER {event} ON dense_cache BEGIN UPDATE dense_cache_epoch SET revision=revision+1 WHERE id=1; END')
        db.execute('CREATE TRIGGER IF NOT EXISTS dense_cache_replace_epoch BEFORE INSERT ON dense_cache WHEN EXISTS(SELECT 1 FROM dense_cache WHERE cache_key=NEW.cache_key) BEGIN UPDATE dense_cache_epoch SET revision=revision+1 WHERE id=1; END')
        db.commit()
        self._resident = {}
        self._identity = None

    def _check_cache_generation(self):
        identity = (TEMPLATE, self.encoder.signature, self.encoder.dimension,
                    self.db.execute('SELECT revision FROM dense_cache_epoch WHERE id=1').fetchone()[0])
        if identity != self._identity:
            self._resident.clear()
            self._identity = identity

    def vectors(self, records):
        self._check_cache_generation()
        keys = [digest([TEMPLATE, self.encoder.signature, r.get('device_id'), text_for(r)]) for r in records]
        values, missing = {}, {}
        for key, record in zip(keys, records):
            if key in self._resident:
                values[key] = self._resident[key]
                self.stats['embedding_resident_hits'] += 1
                continue
            if key in values or key in missing:
                continue
            self.stats['embedding_cache_reads'] += 1
            row = self.db.execute('SELECT data FROM dense_cache WHERE cache_key=?', (key,)).fetchone()
            if row:
                try:
                    v = json.loads(row[0]); validate_vector(v, self.encoder.dimension)
                except (ValueError, TypeError):
                    self.stats['embedding_cache_invalidated'] += 1
                else:
                    values[key] = v; self.stats['embedding_cache_hits'] += 1
                    continue
            missing[key] = text_for(record)
        pending = list(missing)
        for start in range(0, len(pending), 32):
            batch = pending[start:start+32]
            before = time.perf_counter()
            rows = self.encoder.encode([missing[k] for k in batch])
            if len(rows) != len(batch):
                raise RetrievalError('编码数量不足')
            self.stats['embedding_seconds'] += time.perf_counter()-before
            self.stats['embedding_requests'] += 1
            self.stats['embedding_texts'] += len(batch)
            for v in rows:
                validate_vector(v, self.encoder.dimension)
            with self.db:
                for k, v in zip(batch, rows):
                    self.db.execute('INSERT OR REPLACE INTO dense_cache VALUES(?,?)', (k, json.dumps(v)))
                    values[k] = v
        self._check_cache_generation()
        self._resident.update(values)
        return keys, [values[k] for k in keys]

    def recall(self, candidates, state):
        started = time.perf_counter()
        devices={c['device_id'] for c in candidates}
        history = sorted((m for m in state['memories'].values() if m['device_id'] in devices and m['state'] in ('reviewed', 'conflicted')), key=lambda m: m['memory_id'])
        keys, vectors = self.vectors(history+candidates)
        desired={(m['device_id'],m['memory_id']):(m['revision'],key) for m,key in zip(history,keys)}
        # 只读取轻量映射元数据用于恢复/对账，不读取未变化向量正文，不重建索引。
        existing={}
        for device in devices:
            existing.update({(device,mid):(rev,key) for mid,rev,key in self.db.execute('SELECT memory_id,revision,cache_key FROM dense_index WHERE device=?',(device,))})
        pending_queue=dict(self.db.execute("SELECT memory_id,revision FROM vector_queue WHERE status!='indexed'"))
        removed,upserts=0,0
        with self.db:
            for device,mid in existing.keys()-desired.keys():
                self.db.execute('DELETE FROM dense_index WHERE device=? AND memory_id=?',(device,mid))
                removed += 1
            for (device,mid),(revision,key) in desired.items():
                if existing.get((device,mid)) != (revision,key):
                    self.db.execute('INSERT INTO dense_index VALUES(?,?,?,?) ON CONFLICT(device,memory_id) DO UPDATE SET revision=excluded.revision,cache_key=excluded.cache_key',(device,mid,revision,key))
                    upserts += 1
                if pending_queue.get(mid)==revision:
                    self.db.execute("UPDATE vector_queue SET status='indexed' WHERE memory_id=? AND revision=? AND status!='indexed'",(mid,revision))
        self.stats['index_removed'] += removed
        self.stats['index_upserts'] += upserts
        # 仅驻留当前有效历史及本批候选，避免未入库候选长期积累。
        self._resident={k:self._resident[k] for k in set(keys)}
        result = {}
        for i, candidate in enumerate(candidates):
            query = vectors[len(history)+i]
            pool = [(m, sum(a*b for a,b in zip(query, vectors[j]))) for j,m in enumerate(history) if compatible(candidate,m)]
            pool.sort(key=lambda x: (-x[1], x[0]['memory_id']))
            ids = {m['memory_id'] for m,_ in pool[:self.config.get('top_k',20)]}
            ids.update(m['memory_id'] for m,_ in pool if structured(candidate,m))
            while True:
                old = set(ids)
                for group in state['groups'].values():
                    members = {t['memory_id'] for t in group['members']}
                    if group['status']=='open' and members & ids:
                        ids.update(mid for mid in members if mid in state['memories'] and compatible(candidate,state['memories'][mid]))
                if old == ids: break
            result[candidate['candidate_id']] = [m for m in history if m['memory_id'] in ids]
        self.stats['retrieval_seconds'] += time.perf_counter()-started
        self.stats['retrieved_relations'] += sum(map(len,result.values()))
        return result
