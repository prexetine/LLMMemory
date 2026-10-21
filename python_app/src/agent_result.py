"""运行与展示共用的结果计数，不把语义待确认当作技术失败。"""
from collections import Counter


def summarize_results(results: list[dict], selected: int) -> dict:
    counts = Counter()
    retention = Counter()
    versions = Counter()
    invalidations = []
    for result in results:
        versions[result.get('processing_revision','pre-contract-fix')]+=1
        invalidations.extend(result.get('cache_invalidations',[]))
        decisions = {d['candidate_id']:d for d in result.get('maintenance',[])}
        committed = result.get('commit',{}).get('status') == 'committed'
        for p in result.get('reviews',[]):
            cid=p['candidate_id']; d=decisions.get(cid,{})
            source=p.get('source_decision',p.get('decision'))
            control = d.get('storage_control', {})
            # 旧事实流程报告使用字符串，新报告与通用流程使用控制对象。
            control_status = control if isinstance(control, str) else control.get('status')
            if d.get('disposition') == 'discarded': outcome='discarded'
            elif source in ('reject','rejected'): outcome='rejected'
            elif control_status == 'blocked': outcome='storage_blocked'
            elif not committed: outcome='awaiting_commit'
            elif d.get('action') in ('new','reinforce','coexist','supersede','conflict'): outcome=d['action']
            else: outcome='semantic_pending'
            counts[outcome]+=1
            if committed and outcome in ('new','reinforce','coexist','supersede','conflict'):
                retention[p.get('candidate',{}).get('retention','unknown')]+=1
    technical = [r for r in results if r.get('retry_tasks') or r.get('status') in ('failed','partial') or r.get('commit_status') == 'not_committed']
    committed = sum(r.get('commit',{}).get('status') == 'committed' for r in results)
    return dict(traversed_sessions=len(results), committed_sessions=committed,
                technical_pending_sessions=len(technical), unvisited_sessions=max(0,selected-len(results)),
                retryable_tasks=sum(len(r.get('retry_tasks',[])) for r in results), candidate_outcomes=dict(counts),
                admitted_candidate_retention=dict(retention), processing_versions=dict(versions),
                mixed_versions=len(versions)>1, cache_invalidations=invalidations)


def summarize_calls(calls: list[dict], results: list[dict] = ()) -> dict:
    stages = {}
    def entry(name):
        return stages.setdefault(name,dict(calls=0,retries=0,cache_hits=0,cache_invalidated=0,known_tokens=0,usage_missing=0,elapsed_seconds=0.0))
    for c in calls:
        name=c.get('role','unknown')
        s=entry(name)
        s['calls']+=1; s['elapsed_seconds']+=c.get('elapsed_seconds',0)
        s['retries']+=int(c.get('attempt',1)>1)
        total=(c.get('usage') or {}).get('total_tokens')
        if type(total) is int: s['known_tokens']+=total
        else: s['usage_missing']+=1
    # 缓存命中来自当前Session结果快照；调用和重试则来自累计实际请求日志。
    for result in results:
        for name,count in result.get('batch_metrics',{}).items():
            if name.endswith(':cache_invalidated'):
                entry(name.rsplit(':',1)[0])['cache_invalidated']+=count
            if name.endswith((':cached',':item_cached')):
                role=name.rsplit(':',1)[0]
                entry('source_reviewer' if role=='source_proof' else role)['cache_hits']+=count
    return stages
