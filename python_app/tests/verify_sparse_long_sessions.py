"""两个真实长会话的缓存回放；网络和模型调用被显式禁止。"""
import copy
import json
import sqlite3
import tempfile
import time
from contextlib import closing
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from agent_data import Registry, digest
from agent_store import MemoryStore
from agent_llm import TransientModelError
from agent_generic_flow import GenericPipeline
from memory_agent import BASE, CentralAgent, load_inputs
from agent_retrieval import Retriever
from agent_relations import tasks_for, eligible, dependency
from agent_sparse_relations import batch_manifest
from agent_context import size


class Captured(BaseException):
    pass


class OfflineClient:
    model = 'ep-20260924145835-gsgvb'
    calls = 0
    total_tokens = 0
    elapsed = 0.0
    image_assets = {}
    def __init__(self): self.missing=Counter()
    def ask(self, role, prompt, data):
        self.missing[role]+=1
        raise TransientModelError('offline_cache_missing: no model call permitted')


class OfflineEncoder:
    signature='bge-dense-cls-chunk-v1:382ece65c5e44f6902387a1aca5d4f9ca7aaf8af42699892fca930f6c74e1dbf:512'
    dimension=1024
    def encode(self,texts): raise RuntimeError('缺少已有向量；禁止自动编码')


def measure(facts,tasks,proofs,session,signals,compact):
    deps={t['pair_id']:dependency(t,facts,proofs,session,signals) for t in tasks}
    def payload(group):
        ids={r for t in group for r in (t['left'],t['right'])}
        from agent_flow_data import compact_fact
        pack={t['evidence_id']:t for task in group for t in deps[task['pair_id']]['evidence_pack']}
        ss=list({digest(s):s for t in group for s in deps[t['pair_id']]['signals']}.values()) if compact else signals
        ps={r for t in group if 'source' in t['purpose'] for r in (t['left'],t['right'])} if compact else ids
        return dict(flow_version='generic-flow-v3-relations',tasks=group,facts=[dict(ref=r,fact=compact_fact(facts[r])) for r in sorted(ids)],
                    evidence_pack=list(pack.values()),signals=[dict(s,evidence=[{'evidence_id':e['evidence_id']} for e in s['evidence']]) for s in ss],
                    source_reviews=[{k:p[k] for k in ('candidate_id','decision','checks','reason')} for p in proofs if 'c:'+p['candidate_id'] in ps])
    batches=[]
    if compact:
        group=[]
        for t in tasks:
            if group and (len(group)==16 or size(payload(group+[t]))>24000): batches.append(group);group=[]
            group.append(t)
        if group:batches.append(group)
    else:
        def split(group):
            if len(group)>1 and size(payload(group))>24000:
                n=len(group)//2;split(group[:n]);split(group[n:])
            else:batches.append(group)
        for i in range(0,len(tasks),16):split(tasks[i:i+16])
    return dict(tasks=len(tasks),batches=len(batches),dynamic_bytes=sum(size(payload(b)) for b in batches),
                purposes=dict(Counter(p for t in tasks for p in t['purpose'])))


def main():
    reports=[]
    cfg=json.loads((BASE/'python_app/config/memory_agent_core_rank26_30_0907_bge_relations.json').read_text(encoding='utf-8'))
    out=BASE/'output/sparse_relations_validation';out.mkdir(exist_ok=True)
    for user,sid in [('12354838B667950A','chat-731909'),('12354838B6679432','chat-2169125')]:
        started=time.perf_counter()
        origin=BASE/cfg['output_dir']/user/'memory.sqlite3'
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'memory.sqlite3'
            with closing(sqlite3.connect(origin.as_uri()+'?mode=ro',uri=True)) as src,closing(sqlite3.connect(target)) as dest:src.backup(dest)
            store=MemoryStore(target);client=OfflineClient();agent=CentralAgent(client,Registry(BASE/cfg['rules_file']),store,cfg,Path(tmp)/'trace.jsonl')
            session=next(s for s in load_inputs(cfg,user)[0] if s['session_id']==sid)
            captured={}
            def capture(p,candidates,proofs,signals,coverage):
                captured.update(candidates=copy.deepcopy(candidates),proofs=copy.deepcopy(proofs),signals=copy.deepcopy(signals))
                raise Captured()
            with patch.object(GenericPipeline,'maintain',capture):
                try:agent.run_session(session)
                except Captured:pass
            if not captured:raise RuntimeError('未捕获到未提交Session，不能用已提交回执代替本次验证')
            (out/(user+'_inputs.json')).write_text(json.dumps(dict(captured,session=session),ensure_ascii=False),encoding='utf-8')
            cs,ps,ss=(captured[k] for k in ('candidates','proofs','signals'))
            from agent_maintenance import V4Store
            state=V4Store(store).snapshot(user)
            proofmap={p['candidate_id']:p for p in ps}
            active=[c for c in cs if eligible(c,proofmap[c['candidate_id']],ss)]
            retriever=Retriever(store.db,cfg['retrieval'],Counter(),OfflineEncoder())
            histories=retriever.recall(active,state)
            oldfacts,oldtasks=tasks_for(cs,ps,histories,ss)
            observed={}
            observed_rows={}
            for text, in store.db.execute('SELECT data FROM context_work WHERE session_id=?',(sid,)):
                d=json.loads(text)
                if 'pair_id' in d and 'relation' in d:
                    observed[d['pair_id']]=d['relation'];observed_rows[d['pair_id']]=d
            oldids={t['pair_id'] for t in oldtasks}
            all_required={pid for pid,r in observed.items() if r in ('equivalent','conflict','left_updates_right','right_updates_left','uncertain')}
            required=all_required & oldids
            variants=[]
            for k in (8,16,20,32):
                selection=batch_manifest(retriever,active,k)
                facts,tasks=tasks_for(cs,ps,histories,ss,selection)
                ids={t['pair_id'] for t in tasks}
                misses=sorted(required-ids)
                variants.append(dict(top_k=k,required_observed=len(required),unmatched_old_required=sorted(all_required-oldids),missing_required=misses,**measure(facts,tasks,ps,session,ss,True)))
                if not misses:break
            oldmap={t['pair_id']:t for t in oldtasks}
            differences=[]
            for pid in variants[0]['missing_required']:
                task=oldmap[pid]
                differences.append(dict(pair_id=pid,left=oldfacts[task['left']]['claim'],right=oldfacts[task['right']]['claim'],
                                        old_relation=observed[pid],old_reason=observed_rows[pid]['reason']))
            report=dict(user=user,session=sid,candidates=len(cs),active=len(active),offline_cache_misses=dict(client.missing),
                        old=measure(oldfacts,oldtasks,ps,session,ss,False),variants=variants,elapsed_seconds=time.perf_counter()-started,
                        remote_calls=0,embedding_calls=0,original_database_read_only=True,
                        coverage_status='旧模型标签非真值；存在未召回旧标签，不能据此宣称召回质量验收通过',
                        differences=differences,
                        scope='开发回归；任务/批次为同输入结构回放，未扣缓存及精确命中；不是实际调用/token/耗时收益')
            store.close();reports.append(report)
            print(json.dumps({**report,'differences':len(differences),'variants':[{**v,'missing_required':len(v['missing_required']),'unmatched_old_required':len(v['unmatched_old_required'])} for v in variants]},ensure_ascii=False),flush=True)
        (out/'report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
