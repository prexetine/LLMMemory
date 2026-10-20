"""内部开发验证：只读真实基线，在临时SQLite副本恢复回执；禁止调用模型。"""
from pathlib import Path
import collections
import hashlib
import json
import sqlite3
import tempfile
import time
import tracemalloc

import test_generic_flow  # 设置项目src路径，不发起测试或调用
from agent_data import Registry, digest
from agent_store import MemoryStore
from agent_maintenance import V4Store
from memory_agent import CentralAgent, load_inputs
from export_memory_html import export_user


class NoModel:
    def ask(self, *args, **kwargs):
        raise AssertionError('离线恢复禁止模型调用')


REGRESSIONS = {
    'time_reference':'test_contract_fix.ContractFixTests.test_time_only_from_located_user_expression',
    'history_basis':'test_contract_fix.ContractFixTests.test_history_basis_uses_visible_original',
    'history_targets':'test_contract_fix.ContractFixTests.test_history_requires_real_targets_and_change_basis',
    'normalized_proof':'test_contract_fix.ContractFixTests.test_normalized_pending_is_cached_not_model_pass',
    'assistant_context':'test_contract_fix.ContractFixTests.test_assistant_context_not_fact_evidence',
    'speaker_reference':'test_contract_fix.ContractFixTests.test_history_assistant_or_missing_reference_cannot_pass',
    'bounded_retry':'test_single_flow.SingleFlowTests.test_session_retry_keeps_work_unit_budget',
    'empty_fact':'test_contract_fix.ContractFixTests.test_empty_fact_is_protocol_failure_not_silent_success',
}


def category(task):
    reason=task['reason']
    if '时间' in reason or 'time_text' in reason: return 'time_reference'
    if '助手步骤' in reason: return 'assistant_context'
    if '说话角色' in reason or 'wrong_speaker' in reason: return 'speaker_reference'
    if '谓词' in reason or '事实值' in reason or '属性和事实值' in reason: return 'empty_fact'
    if '目标' in reason and task['stage']=='history_compare': return 'history_targets'
    if '历史' in reason or task['stage']=='history_compare': return 'history_basis'
    if '主体指代' in reason or '覆盖' in reason: return 'normalized_proof'
    if '预算' in reason: return 'bounded_retry'
    return 'inspect'


def main():
    started=time.perf_counter(); tracemalloc.start()
    base=Path(__file__).resolve().parents[2]
    root=base/'output/memory_agent_seed_8E41_0907_single_flow/12354838B6678E41'
    out=base/'output/contract_fix_offline_validation'; out.mkdir(exist_ok=True)
    paths=[root/n for n in ('report.json','memory.sqlite3','model_calls.jsonl','manifest.json')]
    hashes=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before=hashes(); report=json.loads(paths[0].read_text(encoding='utf-8'))
    tasks=[dict(t,session_id=r['session_id']) for r in report['results'] for t in r.get('retry_tasks',[])]
    for t in tasks:
        t['regression_category']=category(t)
        t['development_test']=REGRESSIONS[t['regression_category']]
    cfg=json.loads((base/'python_app/config/memory_agent_generic_flow.json').read_text(encoding='utf-8'))
    sessions,_=load_inputs(cfg,report['device_id']); byid={s['session_id']:s for s in sessions}
    recovered=0
    with tempfile.TemporaryDirectory(prefix='memory_contract_') as tmp:
        tmp=Path(tmp); target=tmp/'memory.sqlite3'
        source=sqlite3.connect(paths[1].as_uri()+'?mode=ro',uri=True)
        dest=sqlite3.connect(target)
        try: source.backup(dest)
        finally: dest.close(); source.close()
        store=MemoryStore(target)
        try:
            agent=CentralAgent(NoModel(),Registry(base/cfg['rules_file']),store,cfg,tmp/'trace.jsonl')
            # 不含可重建的检查点：正式事实、证据、版本、回执、支持计数必须完全相同。
            tables=[r[0] for r in store.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'") if r[0]!='context_work']
            fingerprint=lambda:digest({t:[str(tuple(r)) for r in store.db.execute('SELECT * FROM "'+t+'" ORDER BY rowid')] for t in tables})
            original=fingerprint()
            committed=store.db.execute('SELECT session_id,input_hash FROM sessions').fetchall()
            for sid,input_hash in committed:
                session=dict(byid[sid],input_hash=input_hash)
                # 图片已提交回执绑定的是原快照输入指纹；此处不重新下载图片。
                expected=V4Store(store).receipt(session)
                for _ in range(2):
                    result=agent.run_session(session)
                    assert result['commit']==expected
                    assert result.get('processing_revision','pre-contract-fix')=='pre-contract-fix'
                recovered+=1
            assert fingerprint()==original, '副本正式库被恢复操作改变'
            (tmp/'report.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8')
            (tmp/'manifest.json').write_text(paths[3].read_text(encoding='utf-8'),encoding='utf-8')
            # 只导出副本；原目录HTML/数据库不写入。
            export_user(tmp,out/'index.html')
            assert fingerprint()==original, '导出修改了数据库'
        finally: store.close()
    assert hashes()==before,'真实输出被改变'
    result=dict(mode='development_regression_not_independent_acceptance',real_model_calls=0,
        baseline={k:report.get(k) for k in ('processed_sessions','selected_sessions','committed_sessions','technical_pending_sessions','retryable_tasks','model_calls','tokens')},
        technical_sessions=sum(bool(r.get('retry_tasks')) or r['status'] in ('failed','partial') for r in report['results']),
        tasks_by_category=dict(collections.Counter(t['regression_category'] for t in tasks)), tasks=tasks,
        recovered_committed_sessions=recovered,receipt_recovery_repetitions=2,formal_database_unchanged=True,
        source_sha256=before,source_files_unchanged=True,elapsed_seconds=round(time.perf_counter()-started,3),
        python_peak_bytes=tracemalloc.get_traced_memory()[1],
        limitation='57个任务按原失败类别归档供开发回归；并未重新调用模型，不代表36个Session已经恢复。图片回执恢复使用原已绑定输入指纹，不重新获取图像。')
    (out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('tasks','source_sha256')},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
