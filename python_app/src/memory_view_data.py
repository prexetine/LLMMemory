"""只读导出真实记忆库到单文件HTML；不调用模型，不改动候选。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote
from agent_data import Registry, SKILL_NAMES
from agent_result import summarize_results

BASE = Path(__file__).resolve().parents[2]
REVIEW = {
    ("chat-619253", "G01"): "助手被起名为鸡蛋，不等于用户昵称；建议核对 E07。",
    ("chat-619253", "E21"): "请核对记忆请求的方向：用户想记住助手，不等于要求助手记住用户；起名也不自动构成记忆请求。",
    ("chat-619253", "E05"): "声音相似不自动意味着用户对助手的角色定位。",
    ("chat-616301", "G17"): "不讲英语不自动等于不喜欢英语；需与 G42 的语言偏好核对范围。",
    ("chat-616301", "G21"): "车辆归属需要区分父亲和孩子，不能把父亲拥有的车当作孩子的物品。",
    ("chat-616301", "G09"): "此条组合了多项事实，建议拆成可独立更新的原子记忆。",
}


def build_data(folder: Path, rules: Path, *, internal_view: bool = True) -> dict:
    """关联维度、会话、候选与原文证据；抽查注释仅作用于展示。"""
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    device = report["device_id"]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    reviewed = bool(manifest.get("review_version"))
    is_v4 = manifest.get("review_version") == "source-maintenance-v4"
    is_v3 = is_v4 or manifest.get("review_version") == "source-maintenance-v3"
    results = {r["session_id"]: r for r in report["results"]}
    history = {}
    trace_path = folder / "trace.jsonl"
    if trace_path.exists():
        with trace_path.open(encoding="utf-8") as trace_stream:
            for line in trace_stream:
                event = json.loads(line)
                if "history_bootstrap" in event:
                    history[event["session_id"]] = event["history_bootstrap"]["count"] > 0
    connection = sqlite3.connect((folder / "memory.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        sessions = []
        for row in connection.execute("SELECT * FROM sessions WHERE device_id=? ORDER BY start", (device,)):
            result = results[row["session_id"]]
            sessions.append(dict(id=row["session_id"], start=row["start"], end=row["end"],
                                 turns=result["turns"], stored=len(result["approved"]),
                                 history_read=history.get(row["session_id"], False)))
        dimensions = []
        for dim in Registry(rules).dimensions.values():
            coverage = {}
            for sid, result in results.items():
                for status, ids in result.get("coverage", {}).get("dimension_statuses", {}).items():
                    if dim["dimension_id"] in ids:
                        coverage[sid] = status
            dimensions.append(dict(id=dim["dimension_id"], name=dim["name"],
                                   rules=dim["rules"], coverage=coverage))
        memories = []
        for row in connection.execute("SELECT * FROM memories WHERE device_id=? ORDER BY dimension_id, memory_id", (device,)):
            if reviewed and not is_v3 and (row["state"] != "reviewed" or not connection.execute(
                    "SELECT 1 FROM memory_reviews mr JOIN reviews r USING(review_id) "
                    "WHERE mr.memory_id=? AND mr.revision=? AND r.decision='approve'",
                    (row["memory_id"], row["revision"])).fetchone()):
                continue
            raw = json.loads(row["data"])
            evidence = []
            for erow in connection.execute(
                "SELECT e.evidence_id,e.session_id,e.data,me.quote FROM memory_evidence me "
                "JOIN evidence e ON e.evidence_id=me.evidence_id "
                "WHERE me.memory_id=? AND e.device_id=? ORDER BY e.session_id,e.evidence_id",
                (row["memory_id"], device),
            ):
                source = json.loads(erow["data"])
                evidence.append(dict(id=erow["evidence_id"], quote=erow["quote"],
                                     session=erow["session_id"], time=source["occurred_at"],
                                     source=source.get("source_file", ""), sheet=source.get("sheet", ""),
                                     row=source.get("source_row", ""), source_kind=source.get("source_kind", "dialogue"),
                                     image_sha256=source.get("image_sha256")))
            if not evidence:
                raise ValueError("记忆缺少原文证据：" + row["memory_id"])
            notes = sorted({REVIEW[(e["session"], row["dimension_id"])] for e in evidence
                            if not reviewed and device == "12354838B6679292"
                            and (e["session"], row["dimension_id"]) in REVIEW})
            memories.append(dict(id=row["memory_id"], dimension=row["dimension_id"],
                                 claim=row["claim"], subject=row["subject"], scope=row["scope"],
                                 kind=row["kind"], state=row["state"],
                                 revision=row["revision"], support_count=raw.get("support_count"),
                                 source_proof=raw.get("source_proof"), candidate_id=raw.get("candidate_id"),
                                 assertion_mode=raw.get("assertion_mode"), subject_role=raw.get("subject_role"),
                                 retention=raw.get("retention", "memory"), retention_reason=raw.get("retention_reason", "旧结果未进行价值分流"), merged_into=raw.get("merged_into"),
                                 event=raw.get("event"), fact=raw.get("fact"), dimension_ids=raw.get("dimension_ids", [row["dimension_id"]]),
                                 structure=raw.get("structure"), support_annotations=raw.get("support_annotations", []),
                                 supersedes=raw.get("supersedes", []),
                                 skill_name=SKILL_NAMES.get(raw.get("source_skill_id"), "未记录"),
                                 keywords=raw["keywords"], review_notes=notes, evidence=evidence))
        reviews = [json.loads(r["data"]) for r in connection.execute(
            "SELECT data FROM reviews ORDER BY rowid")] if reviewed else []
        reviews = [r for r in reviews if r["device_id"] == device]
        for result in results.values():
            if result.get("flow_version") in ("generic-flow-v1", "generic-flow-v2", "generic-flow-v3-relations") and result.get("commit_status") == "not_committed":
                reviews.extend(dict(r, reviewed_at="尚未提交") for r in result.get("reviews", []))
        for review in reviews:
            if review.get("candidate", {}).get("retention") == "discard":
                review["display_decision"] = "skipped"
        if is_v4:
            from agent_fact_policy import latest_candidates
            latest_ids = {(sid, c["candidate_id"]) for sid in {r["session_id"] for r in reviews}
                          for c in latest_candidates([r["candidate"] for r in reviews if r["session_id"] == sid])}
            reviews = [r for r in reviews if (r["session_id"], r["candidate_id"]) in latest_ids]
        groups, versions, summaries = [], [], []
        if is_v3:
            if is_v4:
                from agent_maintenance import applies
            else:
                from agent_store_v3 import applies
            state_row = connection.execute("SELECT data FROM v3_state WHERE device_id=?", (device,)).fetchone()
            state = json.loads(state_row[0]) if state_row else {}
            groups = list(state.get("groups", {}).values())
            versions = state.get("versions", [])
            summaries = [s for s in state.get("summaries", {}).values() if s["status"] == "active"]
            withdrawals = [s for result in results.values() for s in result.get("signals", []) if s["kind"] == "withdrawal"]
            restricted_reviews = []
            for result in results.values():
                decisions = {d["candidate_id"]: d for d in result.get("maintenance", [])}
                for review in reviews:
                    if review["session_id"] == result["session_id"] and review["candidate_id"] in decisions:
                        d = decisions[review["candidate_id"]]
                        review["reason"] += "；维护：" + d["action"] + " / " + d["reason"]
                        review["maintenance_action"] = "discarded" if review.get("candidate", {}).get("retention") == "discard" else d["action"]
                        review["storage_control"] = d.get("storage_control", {"status": "legacy", "reason": "旧版未独立判断"})
            # 禁止保存项不嵌入可转发HTML正文；数据库审计留在本地。
            for m in memories:
                if m["state"] == "pending_control" and not internal_view:
                    m.update(claim="受限内容 · 已停止有效使用", subject="受限", scope="受限", keywords=[], evidence=[], event=None, fact=None, structure=None, support_annotations=[])
            for r in reviews:
                if not internal_view and ((not is_v4 and r.get("checks", {}).get("privacy") != "pass") or any(applies(s, r["candidate"]) for s in withdrawals)):
                    restricted_reviews.append([r["session_id"], r["candidate_id"]])
                    r["candidate"] = dict(r["candidate"], claim="受限候选", subject="受限", scope="受限", keywords=[], evidence=[], event=None, fact=None, structure=None)
                    r["reason"] = "隐私检查未通过或不确定；正文未嵌入HTML。"
            restricted_ids = {m["id"] for m in memories if m["state"] == "pending_control"}
            for g in groups:
                if not internal_view and restricted_ids.intersection(m["memory_id"] for m in g["members"]):
                    g.update(reason="受限冲突组，详情不嵌入HTML", evidence_ids=[], resolution=None)
            versions = [{k: m.get(k) for k in ("memory_id", "revision", "state", "dimension_id")} for m in versions]
        return dict(device_id=device, internal_view=internal_view, dimensions=dimensions, memories=memories, sessions=sessions,
                    execution_summary=summarize_results(report["results"],report["selected_sessions"]),
                    actual_stage_metrics=report.get("actual_stage_metrics",{}),
                    policy=manifest.get("review_version"), conflict_groups=groups, versions=versions, summaries=summaries,
                    retry_tasks=[dict(t, session_id=r["session_id"]) for r in results.values() for t in r.get("retry_tasks", [])],
                    flow_version=report.get("flow_version", "legacy"),
                    restricted_reviews=restricted_reviews if is_v3 else [],
                    extraction_issues=[dict(i, raw_candidate=i.get("raw_candidate") if internal_view else None) for r in results.values() for i in r.get("extraction_issues", [])],
                    reviewed=reviewed, reviews=reviews, run_status=report["status"], source_scope=report.get("source_scope", "dialogue_only"),
                    run_error=report.get("error"), selected_sessions=report.get("selected_sessions", len(sessions)))
    finally:
        connection.close()



from collections import Counter

def candidate_state(candidate: dict, review: dict | None, rejected: dict) -> tuple[str, str, str]:
    """以最后逐条Review为准；窗口认可不等于入库或逐条Review通过。"""
    if review:
        decision = review["decision"]
        if decision == "reject":
            return "rejected", "逐条审核拒绝", review["reason"]
        label = {"defer": "待确认", "revise": "待修订", "approve": "审核通过 · 尚未确认提交"}[decision]
        return "pending", label, review["reason"]
    if candidate["candidate_id"] in rejected:
        return "rejected", "覆盖审核拒绝", rejected[candidate["candidate_id"]]
    return "pending", "已提取 · 未完成逐条审核", "有候选内容，但没有最终逐条Review及正式提交凭证。"


def build_report_user(folder: Path, formal: dict) -> dict:
    """按用户、会话、候选ID隔离；只读取当前报告所引用的窗口检查点。

    cards的status为confirmed/pending/rejected；confirmed按memory_id去重，
    其他状态按session_id+candidate_id去重。counts不是去重后的现实事实数。
    restricted表示隐私检查未通过，正文和原话不嵌入汇报HTML。
    """
    if not (folder / "report.json").exists():
        return dict(device_id=formal["device_id"], run_status=formal["run_status"],
                    selected_sessions=0, committed_sessions=0, sessions=[], cards=[],
                    dimensions=formal["dimensions"], counts={}, source_scope="unknown")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    device = formal["device_id"]
    internal_view = formal.get("internal_view", True)
    policy = formal.get("policy", report.get("review_version", "legacy"))
    v4 = policy == "source-maintenance-v4"
    if report["device_id"] != device:
        raise ValueError("汇报视图用户与报告不一致")
    cards = [dict(m, status="confirmed", label="已审核入库", session_id="",
                  reason="已通过机器Review，并有正式数据库提交与审核凭证；不代表人工确认。",
                  checks={}, restricted=False) for m in formal["memories"]]
    state_labels = {"conflicted": ("conflicted", "未解冲突", "来源有依据，但存在互斥记忆；不设置默认正确方。"),
                    "merged": ("merged", "重复记录已归并", "证据和维度已归并到主记录，本条仅保留审计关联。"),
                    "superseded": ("superseded", "已被替代", "存在明确更正或状态变化；旧版本保留，不是当前有效结论。"),
                    "pending_control": ("pending", "已停止有效使用", "存在禁止保存或撤回线索，内容已隐藏；本轮未执行删除。")}
    for card in cards:
        source = next((r for r in formal.get("reviews", []) if r.get("proof_hash") == card.get("source_proof")), None)
        card.update(source_decision=(source.get("source_decision") or {"approve": "pass", "defer": "pending", "reject": "reject"}.get(source.get("decision"))) if source else "pass",
                    storage_control={"status": "blocked" if card.get("state") == "pending_control" else "allowed", "reason": "当前正式库状态"},
                    maintenance_action=card.get("state", "reviewed"), block_reason="已停止有效使用" if card.get("state") == "pending_control" else "", policy=policy)
        if source:
            card["checks"] = source.get("checks", {})
        if card.get("state") == "reviewed" and card.get("retention") in ("event", "session_record"):
            card["status"] = card["retention"]
            card["label"] = {"event": "事件记忆", "session_record": "当次需求 · 会话记录"}[card["retention"]]
            card["reason"] = card.get("retention_reason", "") + "；来源审核通过，不代表长期偏好。"
        if card.get("state") in state_labels:
            card["status"], card["label"], card["reason"] = state_labels[card["state"]]
            if internal_view and card["state"] == "pending_control":
                card["reason"] = "存在禁止保存或撤回线索，已停止有效使用；内部查看保留原内容，本轮未执行删除。"
        card["conflict_groups"] = [g["conflict_group_id"] for g in formal.get("conflict_groups", [])
                                   if g["status"] == "open" and any(m["memory_id"] == card["id"] for m in g["members"])]
    db = sqlite3.connect((folder / "memory.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        committed = {(sid, cid): mid for sid, cid, mid in db.execute(
            "SELECT s.session_id,c.candidate_id,c.memory_id FROM changes c "
            "JOIN sessions s USING(batch_id) WHERE s.device_id=?", (device,))}
        evidence = {}
        for eid, sid, raw in db.execute("SELECT evidence_id,session_id,data FROM evidence WHERE device_id=?", (device,)):
            source = json.loads(raw)
            evidence[eid] = dict(id=eid, session=sid, time=source.get("occurred_at", ""),
                                 source=source.get("source_file", ""), sheet=source.get("sheet", ""),
                                 row=source.get("source_row", ""), source_kind=source.get("source_kind", "dialogue"),
                                 image_sha256=source.get("image_sha256"))
        sessions = []
        for result in report["results"]:
            sid = result["session_id"]
            candidates = {c["candidate_id"]: c for c in result.get("candidates", [])}
            latest = {}
            for review in result.get("reviews", []):
                if review["device_id"] != device or review["session_id"] != sid:
                    raise ValueError("审核记录越过用户或会话边界")
                cid = review["candidate_id"]
                if review["candidate"]["candidate_id"] != cid:
                    raise ValueError("审核记录与候选ID不一致")
                candidates[cid] = review["candidate"]
                latest[cid] = review
            rejected = {r["candidate_id"]: r["reason"] for r in result.get("coverage", {}).get("rejected", [])}
            # partial会话无合并账本，只从该次报告实际引用的窗口取拒绝原因。
            if result["status"] != "completed":
                for window in result.get("windows", []):
                    row = db.execute("SELECT data FROM context_work WHERE work_key=? AND device_id=? AND session_id=?",
                                     (window["window_hash"], device, sid)).fetchone()
                    if row:
                        work = json.loads(row[0])
                        rejected.update({r["candidate_id"]: r["reason"] for r in work.get("coverage", {}).get("rejected", [])})
            for cid, candidate in candidates.items():
                if v4 and cid in {c.get("parent_candidate_id") for c in candidates.values()}:
                    continue
                # 已提交的候选可能强化同一条记忆或已被后续替代，不重复当成待确认。
                if (sid, cid) in committed:
                    continue
                review = latest.get(cid)
                status, label, reason = candidate_state(candidate, review, rejected)
                maintenance = next((d for d in result.get("maintenance", []) if d["candidate_id"] == cid), None)
                if maintenance and maintenance["action"] == "pending" and status != "rejected":
                    label = "历史维护待确认" if review and review["decision"] == "approve" else label
                    if maintenance["reason"] != reason:
                        reason += "；维护：" + maintenance["reason"]
                if candidate.get("retention") == "discard":
                    status, label, reason = "discarded", "不生成记忆", candidate.get("retention_reason", "没有新增信息")
                checks = review.get("checks", {}) if review else {}
                restricted = not internal_view and (bool(not v4 and checks and checks.get("privacy") != "pass") or [sid, cid] in formal.get("restricted_reviews", []))
                evs = [dict(evidence.get(e["evidence_id"], {"id": e["evidence_id"], "session": sid}), quote=e["quote"])
                       for e in candidate["evidence"]] if not restricted else []
                cards.append(dict(id=f"{sid}:{cid}", candidate_id=cid, session_id=sid,
                                  dimension=candidate["dimension_id"], status=status, label=label,
                                  claim="受限内容 · 请在受控审核环境中核验" if restricted else candidate["claim"],
                                  subject="受限" if restricted else candidate["subject"],
                                  scope="受限" if restricted else candidate["scope"], kind=candidate["kind"],
                                  keywords=[] if restricted else candidate["keywords"], evidence=evs,
                                  reason="隐私检查未通过或不确定，汇报文件不包含正文及原话。" if restricted else reason,
                                  checks=checks, restricted=restricted,
                                  source_decision=(review.get("source_decision") or {"approve": "pass", "defer": "pending", "reject": "reject"}.get(review.get("decision"))) if review else "pending",
                                  storage_control=(maintenance or {}).get("storage_control", {"status": "legacy" if not v4 else "allowed", "reason": "旧版未独立判断" if not v4 else "无明确禁止保存"}),
                                  maintenance_action=(maintenance or {}).get("action", "pending"),
                                  block_reason="受限内容" if restricted else (maintenance or {}).get("reason", reason),
                                  parent_candidate_id=candidate.get("parent_candidate_id"),
                                  assertion_mode=candidate.get("assertion_mode"), subject_role=candidate.get("subject_role"), policy=policy))
                cards[-1].update(event=None if restricted else candidate.get("event"), fact=None if restricted else candidate.get("fact"), structure=None if restricted else candidate.get("structure"), dimension_ids=candidate.get("dimension_ids", [candidate["dimension_id"]]), retention=candidate.get("retention", "memory"))
                if status == "discarded":
                    cards[-1].update(source_decision="skipped", maintenance_action="discarded", block_reason="")
                elif result.get("commit_status") == "not_committed" and maintenance and maintenance["action"] != "pending":
                    cards[-1].update(maintenance_action="awaiting_commit",label="来源已审核 · 等待Session提交",
                                     reason="当前Session还有技术任务待恢复，维护仅为预演，尚未入库。",block_reason="等待完整Session原子提交")
            for issue in result.get("extraction_issues", []):
                raw = issue.get("raw_candidate")
                raw = raw if isinstance(raw, dict) else {}
                cards.append(dict(id=issue["issue_id"], candidate_id=issue["issue_id"], session_id=sid,
                                  dimension="UNCLASSIFIED", status="pending", label="提取待整理",
                                  claim=str(raw.get("claim") or "未形成有效候选") if internal_view else "受限内容",
                                  subject=str(raw.get("subject") or "未整理") if internal_view else "受限",
                                  scope="", kind="fact", keywords=[], evidence=[], checks={},
                                  reason=issue["reason"], block_reason=issue["reason"], restricted=not internal_view,
                                  source_decision="pending", storage_control={"status": "not_assessed", "reason": "尚未进入来源审核"},
                                  maintenance_action="pending", policy=policy))
            sessions.append(dict(id=sid, status="partial" if result.get("needs_attention") else result["status"], turns=result["turns"],
                                 candidates=len(candidates), stored=len(result.get("approved", [])),
                                 reason=("待恢复阶段："+"、".join(sorted({t["stage"] for t in result["retry_tasks"]}))) if result.get("retry_tasks") else result.get("error", "存在未完成窗口，候选尚未正式提交。") if result["status"] != "completed" else "流程完成，不等于所有候选通过。"))
    finally:
        db.close()
    counts = dict(Counter(c["status"] for c in cards))
    return dict(device_id=device, internal_view=internal_view, policy=policy, run_status=report["status"],
                execution_summary=formal.get("execution_summary",{}),actual_stage_metrics=formal.get("actual_stage_metrics",{}),
                selected_sessions=report["selected_sessions"], committed_sessions=len(formal["sessions"]),
                sessions=sessions, cards=cards, dimensions=formal["dimensions"], counts=counts, summaries=formal.get("summaries", []),
                retry_tasks=formal.get("retry_tasks", []), flow_version=report.get("flow_version", "legacy"),
                source_scope=report.get("source_scope", "unknown"))
