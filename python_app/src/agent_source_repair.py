"""来源修订：逐父候选缓存、共享有界预算、容量拆批；子候选重新审核。"""
from __future__ import annotations

import copy
from pathlib import Path

from agent_context import ContextCapacityError
from agent_contracts import VERSION, ModelOutputError
from agent_data import digest, exact_keys
from agent_llm import TransientModelError
from agent_sparse_relations import relevant_signals

PROMPT = (Path(__file__).resolve().parents[1] / 'prompts/memory_source_repair_v2.txt').read_text(encoding='utf-8')


def repair(p, candidates, proofs, signals, limit):
    byid = {c['candidate_id']: c for c in candidates}
    parents = [q for q in proofs if q['decision'] != 'reject' and q['repair']['kind'] != 'none']
    ready, keys, dependencies, errors = {}, {}, {}, {}
    maximum = p.a.config.get('stage_max_attempts', 3)

    def normalize(row, parent, session):
        exact_keys(row, {'parent_candidate_id', 'candidates', 'reason'})
        if row['parent_candidate_id'] != parent['candidate_id'] or not isinstance(row['candidates'], list) or not isinstance(row['reason'], str):
            raise ValueError('修订父ID或字段类型错误')
        children = []
        for raw in row['candidates']:
            skill = p.a.registry.dimensions.get(raw.get('dimension_id'), {}).get('owner_skill_id', parent['source_skill_id'])
            c = p.normalize_candidate(copy.deepcopy(raw), skill, session, 'repair:' + parent['candidate_id'])
            c.update(parent_candidate_id=parent['candidate_id'], candidate_version=parent['candidate_version'] + 1)
            if c['subject_role'] == 'current_user' and parent['subject_role'] != 'current_user':
                c.update(subject_role=parent['subject_role'], subject_key=parent['subject_key'], subject=parent['subject'])
            c['candidate_id'] = digest([VERSION, 'repair', parent['candidate_id'], c])[:20]
            if c['candidate_id'] not in {x['candidate_id'] for x in children}:
                children.append(c)
        return children

    for q in parents:
        cid = q['candidate_id']; parent = byid[cid]
        dep = dict(parent=parent, repair=q['repair'], evidence_pack=p.pack([parent]), signals=relevant_signals(signals,[parent]))
        session = dict(p.s, turns=dep['evidence_pack'])
        def check(row, parent=parent, session=session):
            normalize(row, parent, session)
        key, cached = p.runtime.load_item('source_repair', PROMPT, dep, check)
        keys[cid], dependencies[cid] = key, dep
        if cached is not None:
            ready[cid] = normalize(cached, parent, session)

    def batch(group):
        remaining = [q for q in group if q['candidate_id'] not in ready]
        while remaining:
            available = [q for q in remaining if p.runtime.attempts[keys[q['candidate_id']]] < maximum]
            if not available:
                break
            originals = [byid[q['candidate_id']] for q in available]
            # 只提供本组来源和相邻语境，不附加所有Skill的shared_facts。
            pack = p.pack(originals)
            session = dict(p.s, turns=pack)
            ids = {q['candidate_id'] for q in available}
            data = dict(repair_mode='source_parents_v2', parents=[dict(candidate=byid[q['candidate_id']], repair=q['repair']) for q in available],
                        evidence_pack=pack, signals=relevant_signals(signals, originals),
                        validation_feedback={cid: str(errors[cid]) for cid in ids if cid in errors})
            def envelope(r):
                exact_keys(r, {'results'})
                if not isinstance(r['results'], list) or any(not isinstance(x, dict) for x in r['results']):
                    raise ValueError('修订结果必须为对象数组')
                if any(x.get('parent_candidate_id') not in ids for x in r['results']):
                    raise ValueError('修订返回额外父ID')
            for cid in ids:
                p.runtime.attempts[keys[cid]] += 1
            unit = digest(['source-parent-batch-v2', sorted((cid, p.runtime.attempts[keys[cid]]) for cid in ids)])
            try:
                result = p.ask('source_repair', PROMPT, data, envelope, attempt_limit=1, unit_key=unit)
            except ContextCapacityError as exc:
                for cid in ids:
                    p.runtime.attempts[keys[cid]] -= 1
                    errors[cid] = exc
                if len(available) > 1:
                    n = len(available) // 2
                    p.stats['source_repair:capacity_splits'] += 1
                    batch(available[:n]); batch(available[n:])
                break
            except (ModelOutputError, TransientModelError) as exc:
                for cid in ids:
                    errors[cid] = exc
                continue
            for q in available:
                cid = q['candidate_id']
                try:
                    rows = [x for x in result['results'] if x.get('parent_candidate_id') == cid]
                    if len(rows) != 1:
                        raise ValueError('修订父ID缺失或重复')
                    # 每个父项只允许引用自己的证据包，不允许批内串用上下文。
                    own_session = dict(p.s, turns=dependencies[cid]['evidence_pack'])
                    children = normalize(rows[0], byid[cid], own_session)
                except (ValueError, TypeError, KeyError) as exc:
                    errors[cid] = exc
                    p.runtime.discard_unit_cache(unit, 'invalid repair rows; valid parents cached separately')
                    p.record_validation('source_repair:item', p.runtime.attempts[keys[cid]], 'invalid', cid + ': ' + str(exc))
                    continue
                ready[cid] = children
                p.runtime.save_item(keys[cid], 'source_repair', dependencies[cid], rows[0])
                remaining.remove(q)

    for offset in range(0, len(parents), limit):
        batch(parents[offset:offset + limit])
    children = []
    for q in parents:
        cid = q['candidate_id']
        if cid in ready:
            children.extend(ready[cid])
            q['reason'] = '需要整理：' + q['repair']['reason'] + '；一轮整理已结束，子候选单独审核'
        else:
            error = errors.get(cid, ModelOutputError('来源修订工作单元重试预算耗尽'))
            p.task('source_repair', [cid], error)
            p.stats['repair_incomplete_items'] += 1
            q['reason'] = '需要整理，定向修订未完成：' + str(error)
        q['proof_hash'] = digest({k: v for k, v in q.items() if k != 'proof_hash'})
    child_proofs = []
    for i in range(0, len(children), limit):
        child_proofs.extend(p.review(children[i:i + limit], signals))
    candidates.extend(children); proofs.extend(child_proofs)
    return candidates, proofs
