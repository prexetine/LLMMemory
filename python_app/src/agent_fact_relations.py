"""逐关系缓存和依赖组规划；不执行数据库写入。"""
from __future__ import annotations

from collections import Counter
from agent_data import digest
from agent_context import ContextCapacityError
from agent_llm import TransientModelError
from agent_flow_data import materialize_references
from agent_fact_model import fact_text


def judge(pipeline, pairs, signals, prompt):
    """只重发失败关系，合法关系立即缓存；两次预算跨外层恢复共享。"""
    result, pending = {}, {}
    keys, inputs = {}, {}
    for pid, (left, right) in pairs.items():
        ids = {e["evidence_id"] for item in (left, right) for e in item.get("evidence", [])}
        related = [s for s in signals if ids & {e["evidence_id"] for e in s["evidence"]}]
        ids.update(e["evidence_id"] for s in related for e in s["evidence"])
        pack = [t for t in pipeline.s["turns"] if t["evidence_id"] in ids]
        data = dict(pair_id=pid, left=fact_text(left), right=fact_text(right),
                    left_evidence=left.get("evidence", []), right_evidence=right.get("evidence", []),
                    evidence_pack=pack, signals=related)
        inputs[pid] = data
        def validate(row, pid=pid):
            pipeline._validate_relation_envelope({"results": [row]}, [(pid, pairs[pid])])
            basis = materialize_references(row["basis_refs"], dict(pipeline.s, turns=inputs[pid]["evidence_pack"]))
            allowed = {e["evidence_id"] for s in inputs[pid]["signals"] if s["kind"] in ("correction", "change") for e in s["evidence"]}
            if row["relation"] == "changed_with_direction" and (not basis or any(e["evidence_id"] not in allowed for e in basis)):
                raise ValueError("更新缺少当前更正或变化证据")
        key, cached = pipeline.runtime.load_item("source_relations", prompt, data, validate)
        keys[pid] = (key, validate)
        if cached:
            result[pid] = cached
        else:
            pending[pid] = None
    limit = pipeline.a.config.get("relation_batch_size", 16)
    budget = pipeline.runtime.attempts
    errors = {}

    def process(ids):
        ids = [pid for pid in ids if pid not in result and budget["pair:" + keys[pid][0]] < 2]
        if not ids:
            return
        for pid in ids:
            budget["pair:" + keys[pid][0]] += 1
        data = {"tasks": [inputs[pid] for pid in ids]}
        def envelope(raw):
            if not isinstance(raw, dict) or set(raw) != {"results"} or not isinstance(raw["results"], list):
                raise ValueError("关系封套非法")
            found = [r.get("pair_id") for r in raw["results"] if isinstance(r, dict)]
            if any(pid not in ids for pid in found):
                raise ValueError("关系包含额外任务ID")
        try:
            raw = pipeline.ask("source_relations", prompt, data, envelope, attempt_limit=1)
        except ContextCapacityError as exc:
            if len(ids) > 1:
                # 未发出的容量失败不消耗语义重试次数。
                for pid in ids:
                    budget["pair:" + keys[pid][0]] -= 1
                mid = len(ids) // 2
                process(ids[:mid]); process(ids[mid:])
            else:
                errors[ids[0]] = str(exc)
            return
        except (ValueError, TypeError, KeyError, TransientModelError) as exc:
            errors.update({pid: str(exc) for pid in ids})
            return
        counts = Counter(r.get("pair_id") for r in raw["results"] if isinstance(r, dict))
        for pid in ids:
            try:
                if counts[pid] != 1:
                    raise ValueError("关系缺失或重复ID")
                row = next(r for r in raw["results"] if isinstance(r, dict) and r.get("pair_id") == pid)
                keys[pid][1](row)
                pipeline.runtime.save_item(keys[pid][0], "source_relations", inputs[pid], row)
                result[pid] = row
            except (ValueError, TypeError, KeyError) as exc:
                errors[pid] = str(exc)
        if any(pid not in result for pid in ids):
            pipeline.runtime.discard_unit_cache(
                pipeline.runtime.fingerprint("source_relations", prompt, dict(data, flow_version=pipeline.flow, policy=__import__("agent_contracts").VERSION)),
                "逐项未完成，保留成功条目缓存")

    for _ in range(2):
        todo = [pid for pid in pending if pid not in result]
        for start in range(0, len(todo), limit):
            process(todo[start:start + limit])
    for pid in pending:
        if pid not in result:
            pipeline._task("semantic_relation", [e["candidate_id"] for e in pairs[pid] if "candidate_id" in e],
                           errors.get(pid, "关系工作单元重试预算耗尽"), unit_id=pid)
    return {pid: dict(row, basis=materialize_references(row["basis_refs"],
                dict(pipeline.s, turns=inputs[pid]["evidence_pack"]))) for pid, row in result.items()}


def unsafe_equivalence(pairs, relations):
    """等价连通组必须是已证明的完全图，不自动传递等价性。"""
    groups, adjacency = [], {}
    for pid, row in relations.items():
        if row["relation"] != "equivalent":
            continue
        ends = pairs[pid]
        a, b = [("c:" + e["candidate_id"]) if "candidate_id" in e else ("h:" + e["fact_id"]) for e in ends]
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    unseen = set(adjacency)
    while unseen:
        component, queue = set(), [min(unseen)]
        while queue:
            key = queue.pop()
            if key in component:
                continue
            component.add(key); queue.extend(adjacency[key] - component)
        unseen -= component
        if any(component - {key} - adjacency[key] for key in component):
            groups.extend(key[2:] for key in component if key.startswith("c:"))
    return set(groups)


def dependency_groups(candidates, relationships):
    """共享候选或历史目标的动作属于同一事务组。"""
    parent = {c["candidate_id"]: c["candidate_id"] for c in candidates}
    def root(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = root(parent[key])
        return parent[key]
    for r in relationships:
        ends = [r.get(side + "_candidate_id") or "h:" + r.get(side + "_fact_id", "") for side in ("left", "right")]
        a, b = map(root, ends)
        parent[max(a,b)] = min(a,b)
    groups = {}
    for c in candidates:
        groups.setdefault(root(c["candidate_id"]), []).append(c)
    return [groups[key] for key in sorted(groups)]
