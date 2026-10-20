"""相同合成事实的新旧调度对比；只Mock关系，不评估模型质量。"""
import json
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from test_unified_relations import UnifiedTests
import test_generic_flow as g
import test_source_agent_v4 as f


def measure(version):
    case=UnifiedTests(); case.setUp()
    try:
        case.h.cfg['flow_version']=version
        for i in range(8):
            s=f.session('history-'+str(i),text='我喜欢项目'+str(i))
            c=g.raw(s,'喜欢项目'+str(i)); c['fact'].update(object='项目'+str(i),value='项目'+str(i))
            result=case.run_session(s,[c]); assert result['commit_status']=='committed'
        case.fake.calls.clear()
        s=f.session('target',text='我喜欢绘画、游泳、唱歌、跳舞、阅读和拼图')
        items=[]
        for name in ('绘画','游泳','唱歌','跳舞','阅读','拼图'):
            c=g.raw(s,'喜欢'+name); c['fact'].update(object=name,value=name); items.append(c)
        start=time.perf_counter(); result=case.run_session(s,items)
        assert result['commit_status']=='committed'
        calls=[(r,d) for r,d in case.fake.calls if r in ('source_consistency','history_compare','source_relations')]
        return dict(calls=dict(Counter(r for r,_ in calls)),request_bytes=sum(len(json.dumps(d,ensure_ascii=False).encode('utf-8')) for _,d in calls),
                    total_relation_calls=len(calls),seconds=time.perf_counter()-start,approved=len(result['approved']))
    finally: case.doCleanups()


if __name__=='__main__':
    tracemalloc.start(); start=time.perf_counter()
    report=dict(kind='mock_scheduling_only_not_model_quality',history_facts=8,new_facts=6,
                old=measure('generic-flow-v2'),new=measure('generic-flow-v3-relations'))
    report.update(elapsed_seconds=time.perf_counter()-start,peak_python_bytes=tracemalloc.get_traced_memory()[1],actual_llm_calls=0)
    path=Path(__file__).resolve().parents[2]/'output/bge_relations_validation/call_comparison.json'
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
