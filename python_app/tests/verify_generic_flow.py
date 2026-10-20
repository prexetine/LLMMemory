"""离线端到端导出与真实历史投影性能检查；不调用模型、不修改旧数据。"""
from pathlib import Path
import json
import hashlib
import sqlite3
import time
import tracemalloc
from test_generic_flow import GenericTests, raw, f, fact
import test_memory_management as management
from agent_flow_data import compact_fact, FLOW
from agent_context import size
from export_memory_html import export_user


def main():
    base=Path(__file__).resolve().parents[2]
    out=base/'output/single_flow_offline_validation';out.mkdir(exist_ok=True)
    started=time.perf_counter();tracemalloc.start()
    case=GenericTests();case.setUp()
    try:
        case.fake.labels=[]
        first=case.h.run_session()
        case.fake.fail_review=True
        second=case.h.run_session(f.session('pending','我喜欢航天'))
        helper=management.ManagementTests();helper.h=case.h;helper.write_report([first,second])
        report=json.loads((case.h.out/'report.json').read_text(encoding='utf-8'))
        report.update(status='partial',flow_version=FLOW)
        (case.h.out/'report.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8')
        export_user(case.h.out,case.h.out/'index.html')
        for name in ('index.html','index_report.html'):
            html=(case.h.out/name).read_text(encoding='utf-8').replace('<body>','<body><aside style="padding:12px;background:#fff1cc">通用流程离线模拟验收，不是真实用户重跑结果。</aside>',1)
            (out/name).write_text(html,encoding='utf-8',newline='\n')
        root=base/'output/memory_agent_seed_8EAF_0907_source_isolation/12354838B6678EAF'
        files=[root/n for n in ('report.json','memory.sqlite3','model_calls.jsonl')]
        hashes=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        before=hashes()
        db=sqlite3.connect(files[1].as_uri()+'?mode=ro',uri=True)
        try: records=[json.loads(r[0]) for r in db.execute('SELECT data FROM memories')]
        finally: db.close()
        old_keys=('memory_id','revision','device_id','dimension_id','subject_key','attribute_key','claim','scope','kind','valid_from','valid_to','last_seen','state','fact','origin_session_id','source_order','event','evidence','retention','assertion_mode','structure')
        old=[{k:m.get(k) for k in old_keys} for m in records]
        new=[compact_fact(m) for m in records]
        assert before==hashes()
        result=dict(mode='offline_mock_and_readonly_projection',real_model_calls=0,records=len(records),
                    old_projection_bytes=sum(map(size,old)),new_projection_bytes=sum(map(size,new)),
                    old_items_over_4096=sum(size(m)>4096 for m in old),new_items_over_4096=sum(size(m)>4096 for m in new),
                    old_files_unchanged=True,sha256=before,elapsed_seconds=round(time.perf_counter()-started,3),
                    python_peak_bytes=tracemalloc.get_traced_memory()[1],model_parameters_flops='远端模型，本次未调用，不适用',
                    limitation='投影保留事实语义字段，不含原文/步骤。字节量不是模型token或准确率，未对旧记忆重新审核。')
        (out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps(result,ensure_ascii=False))
    finally:
        case.doCleanups();tracemalloc.stop()


if __name__=='__main__':main()
