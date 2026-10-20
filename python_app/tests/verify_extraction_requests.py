"""只读真实会话，重建请求容量；不创建模型客户端、不编码、不联网。"""
import json
import time
import tracemalloc
from pathlib import Path
from agent_context import windows, request_metrics, input_budget, bisect_window
from agent_prompts import ROUTER, EXTRACT
from agent_data import SKILL_NAMES, digest
from agent_contracts import VERSION
from memory_agent import BASE, load_inputs


def main():
    started=time.perf_counter();tracemalloc.start()
    config=json.loads((BASE/'python_app/config/memory_agent_950A_extraction_fix.json').read_text(encoding='utf-8'))
    results=[]
    for user,sid in [('12354838B667950A','chat-731909'),('12354838B6679432','chat-2169125')]:
        session=next(s for s in load_inputs(config,user)[0] if s['session_id']==sid)
        ws=list(windows(session,config['session_chunk_chars'],unit='chars'))
        rows=[]
        for w in ws:
            data=dict(session=w,catalog=[dict(skill_id=k,name=v) for k,v in SKILL_NAMES.items()],flow_version=config['flow_version'],policy=VERSION)
            old=request_metrics(dict(config,token_estimator='utf8_bytes_legacy'),'source_router',ROUTER,data)
            new=request_metrics(config,'source_router',ROUTER,data)
            extraction=dict(session=w,skill_rules=[],strategy='events_plans',signals=[],shared_facts=[],flow_version=config['flow_version'],policy=VERSION)
            rows.append(dict(old_router=old,new_router=new,initial_extractor=request_metrics(config,'source_extractor:events_plans',EXTRACT,extraction)))
        def leaves(w,mode):
            d=dict(session=w,catalog=[dict(skill_id=k,name=v) for k,v in SKILL_NAMES.items()],flow_version=config['flow_version'],policy=VERSION)
            if request_metrics(dict(config,token_estimator=mode),'source_router',ROUTER,d)['estimated_input_tokens']<=input_budget(config):return 1
            a,b=bisect_window(w);return leaves(a,mode)+leaves(b,mode)
        results.append(dict(user=user,session=sid,input_hash=session['input_hash'],qa=len(session['turns']),initial_windows=len(ws),
            old_router_windows=sum(leaves(w,'utf8_bytes_legacy') for w in ws),new_router_windows=sum(leaves(w,'mixed_chars_v1') for w in ws),requests=rows))
    elapsed=time.perf_counter()-started;peak=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
    report=dict(config_hash=digest(config),results=results,remote_model_calls=0,embedding_calls=0,
                offline_elapsed_seconds=elapsed,python_peak_bytes=peak,
                limits='开发容量回归；本地估算非供应商tokenizer。未包括未知路由线索及前序Skill候选；后续Skill仍可能需独立拆分。无真实模型质量、端到端耗时或FLOPs结论。')
    out=BASE/'output/extraction_fix_validation';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
