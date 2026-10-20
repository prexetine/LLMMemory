"""分块提取、分页审核和完整session的逐条Review/事务提交。

窗口检查点不是正式记忆；只有完整session覆盖和逐条Review通过后才调用原存储门禁。
"""
from __future__ import annotations
from typing import Any

from agent_context import (CONTEXT_VERSION, ContextCapacityError, bisect_window, windows,
                           pages, setting, compact_feedback)
from agent_data import Registry, digest, exact_keys
from agent_review import ItemReviewer, REVIEW_VERSION


def extract_pages(agent: Any, session: dict, skill: str, rules: list[dict], history: list[dict], feedback: dict | None) -> dict:
    from legacy_memory_agent import EXTRACTOR
    candidates, notes = [], []

    def extract(page: list[dict], depth: int = 0) -> None:
        try:
            response = agent.client.ask("extractor:" + skill, EXTRACTOR, {
                "session": session, "skill_rules": page, "history": history,
                "audit_feedback": compact_feedback(feedback),
                "validation_feedback": getattr(agent, "extraction_validation_feedback", []),
                "unavailable_sources": ["图片视觉", "定位", "通话", "资料", "统计聚合"]})
        except ContextCapacityError:
            if len(page) < 2 or depth >= setting(agent.config, "context_split_retries"):
                raise
            middle = len(page) // 2
            extract(page[:middle], depth + 1)
            extract(page[middle:], depth + 1)
            return
        exact_keys(response, {"candidates", "notes"})
        if not isinstance(response["candidates"], list) or not isinstance(response["notes"], str):
            raise ValueError("SkillResult类型错误")
        allowed = {d["dimension_id"] for d in page}
        if any(not isinstance(c, dict) or c.get("dimension_id") not in allowed for c in response["candidates"]):
            raise ValueError("候选维度不在本次规则页中")
        candidates.extend(response["candidates"])
        notes.append(response["notes"])

    for page in pages(rules, setting(agent.config, "rule_page_tokens")):
        extract(page)
    return {"candidates": candidates, "notes": "\n".join(notes)}


def merge_audits(audits: list[dict], candidates: list[dict], registry: Registry) -> dict:
    approved = {cid for a in audits for cid in a["approved_candidate_ids"]}
    rejected = {r["candidate_id"]: r for a in audits for r in a["rejected"]}
    approved -= set(rejected)
    dims = {c["dimension_id"] for c in candidates if c["candidate_id"] in approved}
    groups = {k: [] for k in ("with_candidate", "no_evidence", "insufficient", "missing_source", "not_assessed")}
    for dim in registry.dimensions:
        statuses = {status for a in audits for status, ids in a["dimension_statuses"].items() if dim in ids}
        status = "with_candidate" if dim in dims else next((k for k in
                 ("not_assessed", "insufficient", "missing_source", "no_evidence") if k in statuses), "not_assessed")
        groups[status].append(dim)
    return {"approved_candidate_ids": sorted(approved), "rejected": list(rejected.values()),
            "followup_skills": sorted({s for a in audits for s in a["followup_skills"]}),
            "dimension_statuses": groups, "notes": "分块/分页复核合并；不代表全量源数据均可推断106维度"}


def audit_pages(agent: Any, session: dict, candidates: list[dict], history: list[dict]) -> dict:
    from legacy_memory_agent import AUDITOR, validate_audit
    audits = []

    def audit(page: list[dict], depth: int = 0) -> None:
        try:
            needed = {c["target_memory_id"] for c in page}
            local_history = [m for m in history if m["memory_id"] in needed]
            data = {"session": session, "catalog": agent.registry.catalog(), "history": local_history,
                    "candidates": page, "candidate_rules": [agent.registry.dimensions[d] for d in
                                  sorted({c["dimension_id"] for c in page})]}
            data["execution_feedback"] = getattr(agent, "audit_execution_feedback", {})
            data["other_page_dimensions"] = sorted({c["dimension_id"] for c in candidates if c not in page})
            for attempt in range(2):
                result = agent.client.ask("auditor", AUDITOR + "\n当前仅审核本页候选；不得引用其他页候选ID。其他页维度已有候选不构成本页漏提。", data)
                try:
                    validate_audit(result, page, agent.registry)
                    break
                except (ValueError, KeyError, TypeError) as exc:
                    if attempt == 1:
                        raise
                    from legacy_memory_agent import trace
                    trace(agent.trace_path, {"session_id": session["session_id"],
                                            "audit_retry": str(exc)})
                    # 仅反馈有界结构错误，不携带整份执行记录或擅改模型判断。
                    ids = result.get("approved_candidate_ids", []) if isinstance(result, dict) else []
                    ids = ids if isinstance(ids, list) else []
                    data["validation_feedback"] = {
                        "error": str(exc)[:400],
                        "previous_approved_dimensions": sorted({c["dimension_id"] for c in page
                                                                if c["candidate_id"] in ids}),
                        "instruction": "重新审核并返回完整合法结果；with_candidate必须恰好对应本次认可候选维度。",
                    }
        except ContextCapacityError:
            if len(page) < 2 or depth >= setting(agent.config, "context_split_retries"):
                raise
            middle = len(page) // 2
            audit(page[:middle], depth + 1)
            audit(page[middle:], depth + 1)
            return
        validate_audit(result, page, agent.registry)
        audits.append(result)

    for page in pages(candidates, setting(agent.config, "rule_page_tokens")):
        audit(page)
    merged = merge_audits(audits, candidates, agent.registry)
    validate_audit(merged, candidates, agent.registry)
    return merged


def run_bounded_session(agent: Any, session: dict) -> dict:
    from legacy_memory_agent import EXTRACTOR, validate_candidate, trace
    from legacy_parallel import stop_requested
    key = digest([CONTEXT_VERSION, REVIEW_VERSION, agent.registry.version, agent.config,
                  session["device_id"], session["session_id"], session["input_hash"]])
    final_key = key + ":final"
    staged_final = agent.store.load_work(final_key)
    if staged_final:
        # Review完成后崩溃也使用同一批候选和凭证，原事务门禁仍须重新校验。
        result = staged_final["result"]
        result["receipt"] = agent.store.commit(session, result["approved"], result["coverage"],
            agent.registry.version, "completed", staged_final["history"], result["reviews"])
        return result
    results = []

    def process(window: dict, depth: int = 0) -> None:
        if stop_requested():
            raise KeyboardInterrupt()
        work_key = digest([key, window])
        cached = agent.store.load_work(work_key)
        if cached and cached["status"] == "completed":
            trace(agent.trace_path, {"session_id": session["session_id"], "window_reused": work_key})
            results.append(cached)
            return
        try:
            result = agent._run_window(window)
        except ContextCapacityError as exc:
            trace(agent.trace_path, {"session_id": session["session_id"], "window_replan": work_key,
                                    "depth": depth, "reason": str(exc)})
            if depth >= setting(agent.config, "context_split_retries"):
                raise ContextCapacityError(f"分块重试已达上限，窗口{work_key}未完成：{exc}") from exc
            left, right = bisect_window(window)
            process(left, depth + 1)
            process(right, depth + 1)
            return
        result["window_hash"] = work_key
        result["source_spans"] = [{"evidence_id": t["evidence_id"], "spans": t["source_spans"]} for t in window["turns"]]
        agent.store.save_work(work_key, session, result)
        results.append(result)
        trace(agent.trace_path, {"session_id": session["session_id"], "window_finished": work_key,
                                "status": result["status"], "source_spans": result["source_spans"]})

    for window in windows(session, setting(agent.config, "session_chunk_tokens")):
        process(window)
    complete = bool(results) and all(r["status"] == "completed" for r in results)
    candidates = list({c["candidate_id"]: c for r in results for c in r["candidates"]}.values())
    eligible = list({c["candidate_id"]: c for r in results for c in r["approved"]}.values()) if complete else []
    rejected_ids = {item["candidate_id"] for r in results for item in r["coverage"].get("rejected", [])}
    eligible = [c for c in eligible if c["candidate_id"] not in rejected_ids]
    history = list({m["memory_id"]: m for r in results for m in r["history"]}.values())
    reviews, accepted = [], []
    if complete:
        # 不以窗口审核替代session审核：每个候选对照完整session的全部有界窗口。
        reviewer = ItemReviewer(agent.client, agent.registry, validate_candidate, EXTRACTOR)
        accepted, reviews = reviewer.review(eligible, session, history,
            lambda record: trace(agent.trace_path, {"session_id": session["session_id"], "item_review": record}))
    coverage = merge_audits([r["coverage"] for r in results], candidates, agent.registry) if complete else {"reason": "存在未完成窗口"}
    result = {"session_id": session["session_id"], "turns": len(session["turns"]),
              "status": "completed" if complete else "partial",
              "actions": [a for r in results for a in r["actions"]],
              "issues": [i for r in results for i in r["issues"]],
              "candidates": candidates, "approved": accepted, "coverage": coverage,
              "receipt": {"status": "checkpoint_only_not_committed"},
              "reviews": reviews, "review_version": REVIEW_VERSION,
              "review_summary": {n: sum(r["decision"] == n for r in reviews) for n in ("approve", "revise", "defer", "reject")},
              "data_completeness": "dialogue_only; other_sources_not_processed",
              "context_version": CONTEXT_VERSION,
              "windows": [{k: r[k] for k in ("window_hash", "source_spans", "status")} for r in results]}
    if complete:
        agent.store.save_work(final_key, session, {"result": result, "history": history})
        result["receipt"] = agent.store.commit(session, accepted, coverage, agent.registry.version,
                                              "completed", history, reviews)
    return result
