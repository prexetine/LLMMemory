"""真实本地编码＋已有数据库副本召回检查；不调用生成式模型。"""
import argparse
import copy
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
import tracemalloc
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent_retrieval import LocalEncoder, Retriever, text_for


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def run(args):
    source=Path(args.database).resolve(strict=True)
    before=file_hash(source); started=time.perf_counter(); tracemalloc.start()
    config=dict(service_config=str(Path(args.service_config).resolve()),port=8766,top_k=20,timeout_seconds=240)
    client=LocalEncoder(config)
    with tempfile.TemporaryDirectory() as folder:
        original=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
        db=sqlite3.connect(str(Path(folder)/'copy.sqlite3')); original.backup(db); original.close()
        state=json.loads(db.execute('SELECT data FROM v3_state LIMIT 1').fetchone()[0])
        memories=[m for m in state['memories'].values() if m['state'] in ('reviewed','conflicted')][:40]
        state=dict(memories={m['memory_id']:m for m in memories},groups={})
        candidates=[]
        for i,m in enumerate(memories[:5]):
            c=copy.deepcopy(m); c['candidate_id']='recall-check-'+str(i); candidates.append(c)
        stats=Counter(); retriever=Retriever(db,config,stats,client)
        t=time.perf_counter(); first=retriever.recall(candidates,state); first_seconds=time.perf_counter()-t
        misses=[m['memory_id'] for m,c in zip(memories,candidates) if m['memory_id'] not in {x['memory_id'] for x in first[c['candidate_id']]}]
        requests=stats['embedding_requests']; t=time.perf_counter(); second=retriever.recall(candidates,state); cached_seconds=time.perf_counter()-t
        assert not misses,misses
        assert requests==stats['embedding_requests']
        assert first==second
        # 独立构造的开发语义案例，不作为真实儿童标注数据。
        sentences=['我喜欢恐龙','我不喜欢恐龙','我对恐龙很感兴趣','爸爸喜欢汽车','今天不想听恐龙故事']
        t=time.perf_counter(); vectors=client.encode(sentences); sample_seconds=time.perf_counter()-t
        scores=[[round(sum(a*b for a,b in zip(v,w)),5) for w in vectors] for v in vectors]
        # 语义召回必须包含同义、反义和事件相关候选；不是判定它们重复。
        base=dict(device_id='development',subject_key='self',subject='当前用户',attribute_key='a',origin_session_id='h',
                  assertion_mode='self_report',scope='all_time',evidence=[],structure={'context':'real'},revision=1,state='reviewed')
        required=['我喜欢恐龙','我不喜欢恐龙','我昨天去了水上乐园','我喜欢和爸爸搭积木']
        distractors=['我喜欢'+x for x in ('苹果','香蕉','草莓','西瓜','葡萄','牛奶','面包','饼干','米饭','面条','蛋糕','酸奶','橙子','桃子','梨','芒果','菠萝','柠檬','樱桃','蓝莓','足球','篮球','游泳','跑步','跳舞','唱歌','绘画','阅读','滑冰','骑车')]
        synthetic=[dict(base,memory_id='d'+str(i),claim=t,fact={}) for i,t in enumerate(required+distractors)]
        queries=['我对恐龙很感兴趣','我讨厌恐龙','昨天到水上乐园玩了','最开心的是跟爸爸一起拼积木']
        cs=[dict(base,candidate_id='q'+str(i),claim=t,fact={}) for i,t in enumerate(queries)]
        recalled=retriever.recall(cs,dict(memories={m['memory_id']:m for m in synthetic},groups={}))
        semantic_misses=[i for i,c in enumerate(cs) if 'd'+str(i) not in {m['memory_id'] for m in recalled[c['candidate_id']]}]
        assert not semantic_misses,semantic_misses
        db.close()
    assert before==file_hash(source),'原数据库变化'
    report=dict(kind='development_regression_not_independent_acceptance',database=str(source),database_sha256=before,
                source_unchanged=True,memories=len(memories),exact_recall_cases=len(candidates),misses=misses,
                semantic_recall_cases=4,semantic_recall_misses=semantic_misses,semantic_case_source='本次人工设计的开发样例，非独立标注验收',
                first_recall_seconds=first_seconds,cached_recall_seconds=cached_seconds,metrics=dict(stats),
                model=client.request('/health'),semantic_examples=sentences,similarities=scores,sample_encoding_seconds=sample_seconds,
                elapsed_seconds=time.perf_counter()-started,client_python_peak_bytes=tracemalloc.get_traced_memory()[1],
                flops='未实测；不把编码耗时当作生成式模型耗时',llm_calls=0)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--database',required=True); p.add_argument('--service-config',required=True); p.add_argument('--output',required=True)
    run(p.parse_args())
