"""统一关系工作单元：同源、同批及历史共用协议和逐条缓存。"""
from __future__ import annotations
import copy
import logging
from pathlib import Path
from agent_data import digest, exact_keys
from agent_contracts import seal, verify_seal
from agent_flow_data import compact_fact, source_pairs, materialize_references
from agent_fact_policy import same_fact
from agent_retrieval import compatible, text_for
from agent_stage_runtime import model_signature
from agent_context import ContextCapacityError, size
from agent_llm import TransientModelError

FLOW = 'generic-flow-v3-relations'
PROMPT = (Path(__file__).resolve().parents[1] / 'prompts/memory_relations_v1.txt').read_text(encoding='utf-8')
RELATIONS = ('equivalent','compatible','conflict','left_updates_right','right_updates_left','uncertain')


def cref(c):
    return 'c:' + c['candidate_id']


def mref(m):
    return 'm:' + m['memory_id'] + ':' + str(m['revision'])


def eligible(c, p, signals):
    from agent_maintenance import storage_control
    return c.get('retention') != 'discard' and p['decision']=='pass' and p['repair']['kind']=='none' and storage_control(c,signals)['status']!='blocked'


def tasks_for(candidates, proofs, histories, signals, selection=None):
    sources = {p['candidate_id']:p for p in proofs}
    facts = {cref(c):c for c in candidates}
    tasks = {}
    selected = {frozenset(pair) for pair in selection['pairs']} if selection is not None else None
    def add(left, right, purpose):
        pid = digest([left,right])[:32]
        row = tasks.setdefault(pid, dict(pair_id=pid,left=left,right=right,purpose=[]))
        if purpose not in row['purpose']: row['purpose'].append(purpose)
        if selection is not None: row['context_policy']='sparse-relations-v2'
    source_ids = {frozenset((a['candidate_id'],b['candidate_id'])) for a,b in source_pairs(candidates)
                  if a.get('retention')!='discard' and b.get('retention')!='discard'}
    for i,c in enumerate(candidates):
        for b in candidates[:i]:
            if frozenset((c['candidate_id'],b['candidate_id'])) in source_ids:
                add(cref(c),cref(b),'source')
            if eligible(c,sources[c['candidate_id']],signals) and eligible(b,sources[b['candidate_id']],signals) and compatible(c,b) and (selected is None or frozenset((c['candidate_id'],b['candidate_id'])) in selected):
                add(cref(c),cref(b),'batch')
        for m in histories.get(c['candidate_id'],[]):
            facts[mref(m)] = m
            add(cref(c),mref(m),'history')
    return facts, list(tasks.values())


def dependency(task, facts, proofs, session, signals):
    if task.get('context_policy') == 'sparse-relations-v2':
        from agent_sparse_relations import relevant_signals
        signals = relevant_signals(signals,[facts[task['left']],facts[task['right']]])
    source = {cref({'candidate_id':p['candidate_id']}):p for p in proofs}
    current_ids = {t['evidence_id'] for t in session['turns']}
    wanted = {e['evidence_id'] for ref in (task['left'],task['right']) for e in facts[ref].get('evidence',[]) if e['evidence_id'] in current_ids}
    wanted.update(e['evidence_id'] for s in signals for e in s['evidence'])
    pack = [t for t in session['turns'] if t['evidence_id'] in wanted]
    if task['purpose'] == ['history'] and not signals:
        # 来源已单独审核；纯历史关系只需要用户原话和完整事实限定。
        # 有更正线索、同源分歧或代表补边时仍保留原始上下文。
        pack = [dict(evidence_id=t['evidence_id'], user_text=t.get('user_text', ''),
                     assistant_text='', **({'source_spans': {'user_text': t['source_spans']['user_text']}}
                     if 'user_text' in t.get('source_spans', {}) else {})) for t in pack]
    return dict(task=task, facts=[dict(ref=ref,fact=compact_fact(facts[ref])) for ref in (task['left'],task['right'])],
                evidence_pack=pack, signals=signals,
                source_reviews=[source[ref] for ref in (task['left'],task['right']) if ref in source] if 'source' in task['purpose'] else [])


def validate_row(raw, task, dep, session):
    exact_keys(raw, {'pair_id','relation','reason','basis_refs'})
    if raw['pair_id']!=task['pair_id'] or raw['relation'] not in RELATIONS or not isinstance(raw['reason'],str) or not raw['reason'].strip() or not isinstance(raw['basis_refs'],list):
        raise ValueError('关系条目ID/枚举/类型错误')
    basis = materialize_references(raw['basis_refs'],dict(session,turns=dep['evidence_pack']))
    if raw['relation'].endswith('updates_right') or raw['relation']=='right_updates_left':
        if not basis: raise ValueError('更新关系缺少当前用户依据')
        allowed={(e['evidence_id'],e['quote']) for s in dep['signals'] if s['kind'] in ('correction','change') for e in s['evidence']}
        if not allowed.intersection((e['evidence_id'],e['quote']) for e in basis):
            raise ValueError('更新关系没有对应更正/变化线索')
    return dict(raw,basis=basis)


def compatible_cache_key(task, facts, dep, config, model):
    """仅复用稳定属性之间的无冲突结论；不复用来源、更正、事件或肯否判断。"""
    if task['purpose'] != ['history'] or dep['signals']:
        return None
    records = [facts[task[k]] for k in ('left', 'right')]
    if any(c.get('fact', {}).get('type') != 'attribute' or c.get('scope') != 'all_time'
           or c.get('assertion_mode') != 'self_report' or c.get('retention') != 'memory'
           or any(c.get(k) for k in ('valid_from', 'valid_to'))
           or c.get('fact', {}).get('time_text') or c.get('fact', {}).get('time_value')
           or c.get('state') == 'conflicted' for c in records):
        return None
    if {e['evidence_id'] for e in records[0].get('evidence', [])} & {e['evidence_id'] for e in records[1].get('evidence', [])}:
        return None
    return digest(['stable-compatible-v1', PROMPT, model_signature(config), model,
                   [(c.get('device_id'), c.get('subject_key'), c.get('structure', {}).get('context', 'real'), text_for(c)) for c in records]])


class RelationEngine:
    def __init__(self,pipeline):
        self.p=pipeline

    def run(self,facts,tasks,proofs,signals):
        p=self.p
        rows, deps, keys, semantic_hashes, pending = {}, {}, {}, {}, []
        compatible_keys = {}
        for task in tasks:
            pid=task['pair_id']; dep=dependency(task,facts,proofs,p.s,signals); deps[pid]=dep
            semantic=dict(purpose=task['purpose'],left=text_for(facts[task['left']]),right=text_for(facts[task['right']]),
                          evidence_pack=dep['evidence_pack'],signals=dep['signals'],source_reviews=dep['source_reviews'])
            semantic_hashes[pid]=digest(semantic)
            key=digest(['relation-v1',PROMPT,model_signature(p.a.config),getattr(p.a.client.client,'model',None),p.s['device_id'],semantic])
            keys[pid]=key
            stable_key = compatible_cache_key(task,facts,dep,p.a.config,getattr(p.a.client.client,'model',None))
            compatible_keys[pid] = stable_key
            cached=p.a.store.load_work(key)
            if cached:
                try:
                    verify_seal(cached)
                    raw={k:cached[k] for k in ('pair_id','relation','reason','basis_refs')}
                    raw['pair_id']=pid
                    valid=validate_row(raw,task,dep,p.s)
                    if cached.get('semantic_hash')!=semantic_hashes[pid] or cached.get('basis')!=valid['basis']: raise ValueError('关系缓存依赖变化')
                except (ValueError,TypeError,KeyError) as exc:
                    p.runtime.invalidate(key,'source_relations',cached,exc)
                else:
                    rows[pid]=seal(dict(valid,dependency_hash=digest(dep),semantic_hash=semantic_hashes[pid]))
                    p.stats['source_relations:item_cached']+=1; continue
            stable = p.a.store.load_work(stable_key) if stable_key else None
            if stable:
                try:
                    verify_seal(stable)
                    if stable.get('cache_key') != stable_key or stable.get('relation') != 'compatible':
                        raise ValueError('稳定相容缓存不匹配')
                    raw = dict(pair_id=pid, relation='compatible', reason=stable['reason'], basis_refs=[])
                    valid = validate_row(raw,task,dep,p.s)
                except (ValueError,TypeError,KeyError) as exc:
                    p.runtime.invalidate(stable_key,'source_relations',stable,exc)
                else:
                    rows[pid] = seal(dict(valid,dependency_hash=digest(dep),semantic_hash=semantic_hashes[pid]))
                    p.stats['relation_compatible_reused'] += 1
                    continue
            if same_fact(facts[task['left']],facts[task['right']]):
                raw=dict(pair_id=pid,relation='equivalent',reason='独立事实结构及范围完全相同',basis_refs=[])
                rows[pid]=seal(dict(validate_row(raw,task,dep,p.s),dependency_hash=digest(dep),semantic_hash=semantic_hashes[pid]))
                p.a.store.save_work(key,p.s,rows[pid]); p.stats['relation_exact_matches']+=1
            else: pending.append(task)
        limit=p.a.config.get('relation_batch_size',16)
        sparse=p.a.config.get('relation_policy')=='sparse-relations-v2'
        def payload(group):
            ids={ref for t in group for ref in (t['left'],t['right'])}
            pack={t['evidence_id']:t for task in group for t in deps[task['pair_id']]['evidence_pack']}
            used_signals=list({digest(s):s for t in group for s in deps[t['pair_id']]['signals']}.values()) if sparse else signals
            source_ids={ref for t in group if 'source' in t['purpose'] for ref in (t['left'],t['right'])} if sparse else ids
            return dict(flow_version=FLOW,tasks=group,facts=[dict(ref=r,fact=compact_fact(facts[r])) for r in sorted(ids)],
                        evidence_pack=list(pack.values()),signals=[dict(s,evidence=[{'evidence_id':e['evidence_id']} for e in s['evidence']]) for s in used_signals],
                        source_reviews=[{k:proof[k] for k in ('candidate_id','decision','checks','reason')} for proof in proofs if cref({'candidate_id':proof['candidate_id']}) in source_ids])
        # 按实际请求大小贪心装箱；不再16→8→4→2二分浪费剩余容量。
        groups=[]; current=[]
        for task in pending:
            trial=current+[task]
            if current and (len(trial)>limit or size(payload(trial))>p.a.config.get('relation_batch_bytes',24000)):
                groups.append(current); current=[]
            current.append(task)
        if current: groups.append(current)
        logging.info('Session=%s 关系任务=%s 本批=%s 历史=%s 同源=%s 缓存或精确命中=%s 待请求批次=%s',p.s['session_id'],len(tasks),sum('batch' in t['purpose'] for t in tasks),sum('history' in t['purpose'] for t in tasks),sum('source' in t['purpose'] for t in tasks),len(rows),len(groups))
        p.stats['relation_tasks_total']+=len(tasks)
        p.stats['relation_batches_planned']+=len(groups)
        if sparse and (len(tasks)>p.a.config.get('relation_warn_tasks',2000) or len(groups)>p.a.config.get('relation_warn_batches',200)):
            logging.warning('Session=%s 关系任务量异常：%s对/%s批，请核对候选粒度与召回范围',p.s['session_id'],len(tasks),len(groups))
        if sparse and p.stats['relation_batches_planned']>p.a.config.get('relation_max_batches',600):
            p.task('source_relations',[],f'关系任务预算阻塞：预计{len(groups)}批，未发送剩余任务',code='relation_budget_exceeded')
            return rows
        def batch(group):
            if not group: return
            ids={ref for t in group for ref in (t['left'],t['right'])}
            pack={t['evidence_id']:t for task in group for t in deps[task['pair_id']]['evidence_pack']}
            data=payload(group)
            if len(group)>1 and size(data)>p.a.config.get('relation_batch_bytes',24000):
                half=len(group)//2; batch(group[:half]); batch(group[half:]); return
            remaining=list(group)
            errors={}
            while remaining:
                available=[t for t in remaining if p.runtime.attempts[keys[t['pair_id']]]<p.a.config.get('stage_max_attempts',3)]
                if not available: break
                # 单工作单元共享预算；StageRuntime每次只发一次，避免嵌套重试。
                for t in available: p.runtime.attempts[keys[t['pair_id']]]+=1
                unit=digest(['relation-batch',[(t['pair_id'],p.runtime.attempts[keys[t['pair_id']]]) for t in available]])
                wanted={r for t in available for r in (t['left'],t['right'])}
                request=payload(available)
                if errors:
                    request['validation_feedback']={t['pair_id']:errors[t['pair_id']] for t in available if t['pair_id'] in errors}
                def envelope(result):
                    exact_keys(result,{'results'})
                    if not isinstance(result['results'],list): raise ValueError('关系结果不是数组')
                try:
                    result=p.runtime.ask('source_relations',PROMPT,request,envelope,attempt_limit=1,unit_key=unit)
                except ContextCapacityError as exc:
                    for t in available: p.runtime.attempts[keys[t['pair_id']]]-=1
                    if len(available)>1:
                        half=len(available)//2; batch(available[:half]); batch(available[half:]); return
                    p.task('source_relations',[facts[available[0]['left']]['candidate_id']],exc); return
                except (ValueError,TypeError,KeyError,TransientModelError):
                    continue
                byid={}; extras=False
                allowed={t['pair_id'] for t in available}
                for raw in result['results']:
                    if not isinstance(raw,dict) or raw.get('pair_id') not in allowed: extras=True; continue
                    byid.setdefault(raw['pair_id'],[]).append(raw)
                if extras:
                    p.record_validation('source_relations',1,'invalid_extra_id','关系批次有额外或非法ID，不接收本批')
                    from agent_contracts import VERSION
                    badkey=p.runtime.fingerprint('source_relations',PROMPT,dict(request,policy=VERSION))
                    p.a.store.save_work(badkey,p.s,dict(status='invalidated',reason='extra relation IDs'))
                    continue
                invalid=False
                for task in available:
                    pid=task['pair_id']
                    try:
                        if len(byid.get(pid,[]))!=1: raise ValueError('关系ID缺失或重复')
                        row=validate_row(byid[pid][0],task,deps[pid],p.s)
                    except (ValueError,TypeError,KeyError) as exc:
                        invalid=True
                        errors[pid]=str(exc)
                        p.record_validation('source_relations',p.runtime.attempts[keys[pid]],'invalid',str(exc)); continue
                    rows[pid]=seal(dict(row,dependency_hash=digest(deps[pid]),semantic_hash=semantic_hashes[pid]))
                    p.a.store.save_work(keys[pid],p.s,rows[pid]); remaining.remove(task)
                    if compatible_keys[pid] and row['relation']=='compatible' and not row['basis_refs']:
                        p.a.store.save_work(compatible_keys[pid],p.s,seal(dict(cache_key=compatible_keys[pid],
                            relation='compatible',reason=row['reason'])))
                if invalid:
                    from agent_contracts import VERSION
                    badkey=p.runtime.fingerprint('source_relations',PROMPT,dict(request,policy=VERSION))
                    p.a.store.save_work(badkey,p.s,dict(status='invalidated',reason='invalid relation rows; valid items cached separately'))
            for task in remaining:
                p.task('source_relations',[facts[task['left']]['candidate_id']], '关系校验重试耗尽 pair='+task['pair_id']+'；'+errors.get(task['pair_id'],'结果封套、网络或重试预算未完成'))
        for group in groups: batch(group)
        return rows
