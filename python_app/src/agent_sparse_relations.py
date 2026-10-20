"""本批稀疏召回与可重验清单，不用相似度直接决定事实关系。"""
from __future__ import annotations
import json
from agent_data import digest
from agent_retrieval import TEMPLATE, text_for, compatible, structured, validate_vector, RetrievalError

POLICY = 'sparse-relations-v2'


def select_pairs(candidates, vectors, top_k):
    if type(top_k) is not int or top_k < 1:
        raise ValueError('batch_top_k必须是正整数')
    if len(candidates) != len(vectors):
        raise RetrievalError('本批向量数量不完整')
    selected = set()
    for i, c in enumerate(candidates):
        pool = [(j, sum(a*b for a,b in zip(vectors[i], vectors[j])))
                for j,b in enumerate(candidates) if i != j and compatible(c,b)]
        pool.sort(key=lambda row: (-row[1], candidates[row[0]]['candidate_id']))
        neighbors = {j for j,_ in pool[:top_k]}
        neighbors.update(j for j,_ in pool if structured(c,candidates[j]))
        for j in neighbors:
            selected.add(tuple(sorted((c['candidate_id'], candidates[j]['candidate_id']))))
    return [list(pair) for pair in sorted(selected)]


def batch_manifest(retriever, candidates, top_k):
    keys, vectors = retriever.vectors(candidates)
    return dict(policy=POLICY, top_k=top_k, model_signature=retriever.encoder.signature,
                dimension=retriever.encoder.dimension,
                candidates=[dict(candidate_id=c['candidate_id'], content_hash=digest(text_for(c)), cache_key=k)
                            for c,k in zip(candidates,keys)],
                pairs=select_pairs(candidates,vectors,top_k))


def verify_manifest(db, manifest, candidates):
    if db is None or manifest.get('policy') != POLICY:
        raise ValueError('缺少可重验的本批召回清单')
    if len(manifest['candidates']) != len(candidates):
        raise ValueError('本批召回候选不完整')
    vectors=[]
    for c,ref in zip(candidates,manifest['candidates']):
        key=digest([TEMPLATE,manifest['model_signature'],c.get('device_id'),text_for(c)])
        if ref != dict(candidate_id=c['candidate_id'],content_hash=digest(text_for(c)),cache_key=key):
            raise ValueError('本批召回候选内容或顺序变化')
        row=db.execute('SELECT data FROM dense_cache WHERE cache_key=?',(key,)).fetchone()
        if row is None: raise ValueError('本批召回向量缓存缺失')
        vector=json.loads(row[0]); validate_vector(vector,manifest['dimension']); vectors.append(vector)
    if manifest['pairs'] != select_pairs(candidates,vectors,manifest['top_k']):
        raise ValueError('本批召回任务遗漏或变化')


def relevant_signals(signals, records):
    """同主体的全局/未归一线索保守保留；明确其他属性的线索不重复发送。"""
    ids={e['evidence_id'] for c in records for e in c.get('evidence',[])}
    result=[]
    for s in signals:
        if ids & {e['evidence_id'] for e in s.get('evidence',[])}:
            result.append(s); continue
        if any((not s.get('subject_key') or s.get('subject_key')==c.get('subject_key')) and
               (not s.get('attribute_key') or s.get('attribute_key')==c.get('attribute_key')) for c in records):
            result.append(s)
    return result
