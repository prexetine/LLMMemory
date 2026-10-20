"""坏提取候选按来源组批修复，逐项缓存/预算，不重跑成功项。"""
from __future__ import annotations
from pathlib import Path
from agent_data import digest, exact_keys
from agent_context import ContextCapacityError, view
from agent_contracts import ModelOutputError
from agent_llm import TransientModelError
from agent_prompts import EXTRACT

PROMPT = (Path(__file__).resolve().parents[1]/'prompts/memory_extraction_row_repair_v1.txt').read_text(encoding='utf-8')


def repair_window(w, raw_rows):
    """证据、步骤、时间定位均保留；不可靠引用不猜测裁剪。"""
    ids=set()
    def collect(value):
        if isinstance(value,dict):
            if isinstance(value.get('evidence_id'),str):ids.add(value['evidence_id'])
            for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    for raw in raw_rows:
        if not isinstance(raw,dict) or not isinstance(raw.get('evidence'),list) or not raw['evidence']:return w
        if any(not isinstance(ref,dict) or not isinstance(ref.get('evidence_id'),str) or not ref['evidence_id'] for ref in raw['evidence']):return w
        collect(raw)
    turns=w['turns']; available={t['evidence_id'] for t in turns}
    if not ids or not ids.issubset(available):return w
    positions={j for i,t in enumerate(turns) if t['evidence_id'] in ids for j in range(max(0,i-1),min(len(turns),i+2))}
    return view(w,[turns[j] for j in sorted(positions)])


def repair_rows(p, w, wid, skill, raw_rows, normalize, batch_unit):
    good=[]; pending=[]
    maximum=p.a.config.get('stage_max_attempts',3)
    for index,raw in enumerate(raw_rows):
        try:
            good.append(normalize(raw)); continue
        except (ValueError,TypeError,KeyError) as exc:
            from agent_generic_flow import failure_details
            error=failure_details(exc)
        rid=digest([wid,skill,index,raw])[:24]
        dep=dict(raw=raw,window=w,skill=skill,repair_protocol='extraction-rows-v1')
        def check(value):
            exact_keys(value,{'candidates'})
            if not isinstance(value['candidates'],list): raise ValueError('修复缓存类型错误')
            for c in value['candidates']: normalize(c)
        key,cached=p.runtime.load_item('source_repair',EXTRACT+'\n'+PROMPT,dep,check)
        if cached is not None:
            good.extend(normalize(c) for c in cached['candidates']);continue
        unit=digest(['repair-row-v1',batch_unit,index,raw])
        p.runtime.attempts[unit]=max(p.runtime.attempts[unit],p.runtime.attempts[batch_unit],1)
        pending.append(dict(id=rid,index=index,raw=raw,dep=dep,key=key,unit=unit,error=error,attempt_errors=[dict(attempt=1,**error)]))

    def source(group):
        return repair_window(w,[x['raw'] for x in group])

    def process(group):
        remaining=list(group)
        while remaining:
            active=[x for x in remaining if p.runtime.attempts[x['unit']]<maximum]
            if not active:return
            evidence=source(active)
            parents=[dict(candidate=dict(x['raw'],candidate_id=x['id']) if isinstance(x['raw'],dict) else dict(candidate_id=x['id'],invalid_value=x['raw']),
                          repair=dict(kind='structure',reason=x['error']['reason'])) for x in active]
            data=dict(parents=parents,session=evidence,signals=[],repair_mode='extraction_rows_v1',repair_skill=skill)
            def envelope(r):
                exact_keys(r,{'results'})
                if not isinstance(r['results'],list): raise ValueError('修复results不是数组')
                allowed={x['id'] for x in active}
                for row in r['results']:
                    if not isinstance(row,dict) or row.get('parent_candidate_id') not in allowed:
                        raise ValueError('修复包含额外parent ID')
            for x in active:p.runtime.attempts[x['unit']]+=1
            try:
                result=p.ask('source_repair',PROMPT,data,envelope,attempt_limit=1,
                             unit_key=digest(['repair-batch',[(x['unit'],p.runtime.attempts[x['unit']]) for x in active]]))
            except ContextCapacityError as exc:
                if len(active)>1:
                    for x in active:p.runtime.attempts[x['unit']]-=1
                    middle=len(active)//2;process(active[:middle]);process(active[middle:]);return
                active[0]['error']=dict(code='context_capacity',reason=str(exc),evidence_ids=[])
                return
            except (ModelOutputError,TransientModelError) as exc:
                for x in active:
                    x['error']=dict(code='model_protocol' if isinstance(exc,ModelOutputError) else 'transient_service',reason=str(exc),evidence_ids=[])
                    x['attempt_errors'].append(dict(attempt=p.runtime.attempts[x['unit']],**x['error']))
                continue
            byid={}
            for row in result['results']:byid.setdefault(row['parent_candidate_id'],[]).append(row)
            for x in active:
                try:
                    if len(byid.get(x['id'],[]))!=1:raise ValueError('修复parent ID缺失或重复')
                    row=byid[x['id']][0];exact_keys(row,{'parent_candidate_id','candidates','reason'})
                    if not isinstance(row['candidates'],list) or not isinstance(row['reason'],str):raise ValueError('修复条目类型错误')
                    repaired=[normalize(c) for c in row['candidates']]
                except (ValueError,TypeError,KeyError) as exc:
                    from agent_generic_flow import failure_details
                    x['error']=failure_details(exc)
                    x['attempt_errors'].append(dict(attempt=p.runtime.attempts[x['unit']],**x['error']));continue
                p.runtime.save_item(x['key'],'source_repair',x['dep'],dict(candidates=row['candidates']))
                good.extend(repaired);x['done']=True;remaining.remove(x)
                p.stats['extraction_rows_repaired']+=1
    limit=p.a.config.get('extraction_repair_batch_size',8)
    for i in range(0,len(pending),limit):process(pending[i:i+limit])
    for x in pending:
        if x.get('done'):continue
        p.extraction_issues.append(dict(issue_id=x['id'],source_window=wid,skill=skill,row_index=x['index'],raw_candidate=x['raw'],
            reason=x['error']['reason'],status='pending_extraction',code=x['error']['code'],evidence_ids=x['error'].get('evidence_ids',[]),attempt_errors=x['attempt_errors']))
        p.stats['extraction_rows_pending']+=1
        p.record_validation('source_repair',p.runtime.attempts[x['unit']],'pending_extraction',x['error']['reason'])
    return good
