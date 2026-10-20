"""关系凭证到维护计划的确定性转换；事务中重算验证，不改来源凭证。"""
from __future__ import annotations
import copy
from agent_data import digest
from agent_contracts import seal, verify_seal
from agent_relations import FLOW, cref, mref, tasks_for, dependency, validate_row, eligible
from agent_retrieval import Retriever, RetrievalError, compatible


def compile_decisions(initial, candidates, proofs, signals, session, histories, tasks, rows, missing=None):
    from agent_maintenance import apply_decision, storage_control, suspend
    source={p['candidate_id']:p for p in proofs}
    blocked={}
    taskmap={t['pair_id']:t for t in tasks}
    pairmap={(t['left'],t['right']):rows[t['pair_id']] for t in tasks if t['pair_id'] in rows}
    for task in tasks:
        if 'source' not in task['purpose']: continue
        row=rows.get(task['pair_id'])
        left,right=task['left'][2:],task['right'][2:]
        disagree=any(source[cid]['decision']!='pass' and any(source[cid]['checks'].get(k)!='pass' for k in ('evidence','semantics','subject')) for cid in (left,right))
        if row is None or row['relation'] in ('conflict','uncertain') or row['relation']=='equivalent' and disagree:
            for cid in (left,right): blocked[cid]='同源提取分歧：'+(row['reason'] if row else '关系技术工作未完成')
        elif row['relation'] == 'equivalent':
            indexed = {c['candidate_id']: c for c in candidates}
            if not compatible(indexed[left], indexed[right]):
                for cid in (left, right):
                    blocked[cid] = '同源等价结论与主体/现实场景隔离不一致，需要整理，不能重复新增'
    # 只传递保守隔离，不通过等价链条传递真实性或执行合并。
    while True:
        count=len(blocked)
        for task in tasks:
            row=rows.get(task['pair_id'])
            if 'source' in task['purpose'] and row and row['relation']=='equivalent':
                ids=(task['left'][2:],task['right'][2:])
                if any(cid in blocked for cid in ids):
                    for cid in ids: blocked.setdefault(cid,'同源等价链存在未解决审核分歧')
        if len(blocked)==count: break
    state=copy.deepcopy(initial); decisions=[]; applied={}; roots={}
    for c in candidates:
        cid=c['candidate_id']; p=source[cid]
        d=dict(candidate_id=cid,action='new',targets=[],basis=[],reason='召回及本批关系已处理，无需修改已有记忆',semantic_verified=True,
               source_proof=p['proof_hash'],storage_control=storage_control(c,signals),relation_ids=[])
        if not eligible(c,p,signals):
            d.update(action='pending',reason=p['reason'])
            if d['storage_control']['status']=='blocked': d['reason']=d['storage_control']['reason']
            if c.get('retention')=='discard': d.update(disposition='discarded',reason=c['retention_reason'])
        elif cid in blocked:
            d.update(action='pending',reason=blocked[cid])
        else:
            comparisons={mref(m):m['memory_id'] for m in histories.get(cid,[])}
            for previous,mid in applied.items():
                # 未召回的候选不再强制比较；召回项若已归并，仍需与最终代表直接互证。
                if (cref(c), 'c:'+previous) in pairmap and compatible(c,state['memories'][mid]):
                    comparisons[roots[mid]]=mid
            relations={}; basis=[]; reasons=[]
            for ref,mid in comparisons.items():
                row=pairmap.get((cref(c),ref))
                if row is None:
                    if missing is not None:
                        missing.append((cref(c), ref))
                    d.update(action='pending',reason='缺少与最终代表事实的直接关系，禁止传递式合并'); break
                memory=state['memories'].get(mid)
                if memory is None or memory['state'] not in ('reviewed','conflicted'):
                    d.update(action='pending',reason='相关历史已被本批替代，需要重新判断'); break
                relations[mid]=row['relation']; basis.extend(row['basis']); reasons.append(row['reason']); d['relation_ids'].append(row['pair_id'])
            if d['action']!='pending' and relations:
                kinds=set(relations.values())-{'compatible'}
                if 'uncertain' in kinds or 'right_updates_left' in kinds or len(kinds)>1:
                    d.update(action='pending',reason='相关关系无法安全归并：'+'；'.join(reasons))
                else:
                    relation=next(iter(kinds),'compatible')
                    action={'equivalent':'reinforce','compatible':'coexist','conflict':'conflict','left_updates_right':'supersede'}[relation]
                    targets=[mid for mid,r in relations.items() if r==relation]
                    # 多历史目标等价不能通过共同候选传递证明，保守保留待确认。
                    if action=='reinforce' and len(targets)>1:
                        d.update(action='pending',reason='多个等价历史代表尚未直接互证，不自动传递归并')
                    else:
                        d.update(action=action,targets=[dict(memory_id=mid,revision=state['memories'][mid]['revision']) for mid in sorted(targets)],
                                 basis=list({(e['evidence_id'],e['quote']):e for e in basis}.values()) if action=='supersede' else [],reason='；'.join(reasons))
        # 新流程的维护依据来自经过验证的关系，而非同维度全库列表。
        d['history_hash']=digest([(m['memory_id'],m['revision']) for m in histories.get(cid,[])])
        trial=copy.deepcopy(state)
        try:
            item=apply_decision(trial,c,d,p,session,signals)
        except ValueError as exc:
            d.update(action='pending',targets=[],basis=[],reason='维护门禁：'+str(exc))
        else:
            state=trial
            if item:
                mid=item['memory_id']; applied[cid]=mid
                roots.setdefault(mid, next((mref(m) for m in initial['memories'].values() if m['memory_id']==mid),cref(c)))
        decisions.append(d)
    suspend(state,signals)
    return state,decisions


def representative_tasks(initial, candidates, proofs, signals, session, histories, tasks, rows):
    """仅补实际维护路径缺失的代表边，不对等价链做传递合并。"""
    missing = []
    compile_decisions(initial, candidates, proofs, signals, session, histories, tasks, rows, missing)
    known = {t['pair_id'] for t in tasks}
    result = []
    for left, right in sorted(set(missing)):
        pid = digest([left, right])[:32]
        if pid not in known:
            result.append(dict(pair_id=pid, left=left, right=right, purpose=['representative'], context_policy='sparse-relations-v2'))
    return result


def verify_relation_plan(initial,candidates,proofs,plan,signals,session,db=None):
    manifest=plan['recall_manifest']
    if set(manifest)!={c['candidate_id'] for c in candidates}: raise ValueError('召回清单不完整')
    histories={}
    for cid,refs in manifest.items():
        histories[cid]=[]
        if len({r['memory_id'] for r in refs})!=len(refs): raise ValueError('召回ID重复')
        c=next(c for c in candidates if c['candidate_id']==cid)
        for ref in refs:
            m=initial['memories'].get(ref['memory_id'])
            if not m or m['revision']!=ref['revision'] or m['state'] not in ('reviewed','conflicted') or not compatible(c,m):
                raise ValueError('召回版本/范围无效')
            histories[cid].append(m)
    selection=plan.get('batch_recall_manifest')
    if selection is not None:
        from agent_sparse_relations import verify_manifest
        source={p['candidate_id']:p for p in proofs}
        active=[c for c in candidates if eligible(c,source[c['candidate_id']],signals)]
        verify_manifest(db,selection,active)
    facts,tasks=tasks_for(candidates,proofs,histories,signals,selection)
    if plan.get('representative_completion') == 1:
        # 提交端按同一稳定维护顺序重新推导补边，禁止注入任意跳过召回的任务。
        for _ in range(len(candidates) + 1):
            extra = representative_tasks(initial,candidates,proofs,signals,session,histories,tasks,plan['relation_rows'])
            if not extra:
                break
            tasks.extend(extra)
        else:
            raise ValueError('代表关系补齐未收敛')
    if tasks!=plan['relation_tasks']: raise ValueError('关系任务遗漏或变化')
    rows=plan['relation_rows']
    if set(rows)!={t['pair_id'] for t in tasks}: raise ValueError('关系凭证不完整')
    for task in tasks:
        row=rows[task['pair_id']]; verify_seal(row)
        dep=dependency(task,facts,proofs,session,signals)
        valid=validate_row({k:row[k] for k in ('pair_id','relation','reason','basis_refs')},task,dep,session)
        if row['dependency_hash']!=digest(dep) or row['basis']!=valid['basis']: raise ValueError('关系依据变化')
    state,decisions=compile_decisions(initial,candidates,proofs,signals,session,histories,tasks,rows)
    if decisions!=plan['decisions']: raise ValueError('维护动作不符合关系凭证')
    return state


def maintain(p,candidates,proofs,signals,coverage):
    from agent_relations import RelationEngine
    from agent_maintenance import StaleHistory
    from agent_memory_structure import rebuild_summaries
    from agent_contracts import VERSION
    p.classify(candidates,proofs)
    sparse=p.a.config.get('relation_policy')=='sparse-relations-v2'
    for attempt in range(3):
        initial=p.store.snapshot(p.s['device_id'])
        selection=None
        try:
            active=[c for c in candidates if eligible(c,next(x for x in proofs if x['candidate_id']==c['candidate_id']),signals)]
            if active:
                retriever=getattr(p.a,'retriever',None)
                encoder=getattr(p.a,'embedding_client',None)
                if retriever is None or retriever.db is not p.store.db or (encoder is not None and retriever.encoder is not encoder):
                    retriever=Retriever(p.store.db,p.a.config['retrieval'],p.stats,encoder)
                    p.a.retriever=retriever
                retriever.config=p.a.config['retrieval']
                retriever.stats=p.stats
                histories=retriever.recall(active,initial)
                if sparse:
                    from agent_sparse_relations import batch_manifest
                    selection=batch_manifest(retriever,active,p.a.config['retrieval'].get('batch_top_k',8))
            else:
                histories={}
                if sparse:
                    selection=dict(policy='sparse-relations-v2',top_k=p.a.config['retrieval'].get('batch_top_k',8),model_signature='',dimension=0,candidates=[],pairs=[])
        except RetrievalError as exc:
            p.task('retrieval',[c['candidate_id'] for c in candidates],exc,code=exc.code)
            histories={}
        for c in candidates: histories.setdefault(c['candidate_id'],[])
        if sparse and selection is None:
            selection=dict(policy='sparse-relations-v2',top_k=8,model_signature='',dimension=0,candidates=[],pairs=[])
        facts,tasks=tasks_for(candidates,proofs,histories,signals,selection)
        rows=RelationEngine(p).run(facts,tasks,proofs,signals)
        for _ in range(len(candidates) + 1):
            extra = representative_tasks(initial,candidates,proofs,signals,p.s,histories,tasks,rows)
            if not extra:
                break
            tasks.extend(extra)
            p.stats['representative_relations_added'] += len(extra)
            rows.update(RelationEngine(p).run(facts,extra,proofs,signals))
        else:
            p.task('source_relations',[], '代表关系补齐未收敛', code='representative_not_converged')
        state,decisions=compile_decisions(initial,candidates,proofs,signals,p.s,histories,tasks,rows)
        summary=dict(days=p.a.config.get('summary_window_days',30),min_sessions=p.a.config.get('summary_min_sessions',2))
        rebuild_summaries(state,p.s['end'],**summary)
        plan=seal(dict(policy=VERSION,input_hash=p.s['input_hash'],base_hash=digest(initial),signals_hash=digest(signals),signals=signals,
                       decisions=decisions,summary_config=summary,extraction_issues=p.extraction_issues,result_hash=digest(state),
                       recall_manifest={cid:[dict(memory_id=m['memory_id'],revision=m['revision']) for m in h] for cid,h in histories.items()},
                       relation_tasks=tasks,relation_rows=rows,flow_version=FLOW,representative_completion=1))
        if selection is not None:
            plan=seal(dict(plan,batch_recall_manifest=selection))
        try:
            return p.commit_plan(candidates,proofs,plan,signals,coverage),decisions
        except StaleHistory:
            p.stats['history_revision_retries']+=1
            if attempt==2: raise
