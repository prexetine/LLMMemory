"""新事实库到共享汇报界面的只读投影。"""
import json
from collections import Counter
from pathlib import Path
from agent_data import Registry
from agent_fact_store import FactStore
from agent_dimension_mapping import catalog


def build_fact_report(db, manifest=None):
    root = Path(__file__).resolve().parents[2]
    definitions = catalog(Registry(root / "J20核心用户记忆维度与提取存储规划.md"))
    has_results = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='fact_dimension_results'").fetchone()
    dimensions = [dict(id=v["dimension_id"], name=v["name"], rules=(
        "需统计分析记录；不能由单条事实推断。 " if v["aggregation_required"] else
        "需明确的跨来源关联；当前事实接口不自动推断关联。 " if v["cross_source_required"] else
        "需跨会话、跨日期支持并满足规则。 " if v["requires_repetition"] else
        "由已审核事实按完整语义规则映射。 ") + str(v["rules"]))
        for v in definitions]
    dimensions.append(dict(id="UNCLASSIFIED",name="未分类事实",rules="没有可靠维度映射，仍保留事实及审核记录，不强行归类。"))
    labels = {"accepted":"已审核入库","conflicted":"未解冲突","superseded":"被替代版本",
              "merged":"已归并","provisional":"待确认或技术待补","rejected":"审核不通过","suspended":"停止有效使用"}
    cards = []
    for row in db.execute("SELECT * FROM fact_assertions ORDER BY fact_id"):
        item=json.loads(row["data"])
        dims=[r[0] for r in db.execute("SELECT DISTINCT dimension_id FROM fact_dimension_maps WHERE fact_id=?",(row["fact_id"],))]
        status={"accepted":"confirmed","provisional":"pending","rejected":"rejected","suspended":"pending"}.get(row["state"],row["state"])
        if status=="confirmed":
            status={"event":"event","request":"session_record","observation":"event"}.get(item.get("fact",{}).get("type"),status)
        evidence=[dict(id=r["evidence_id"],quote=r["quote"]) for r in db.execute("SELECT evidence_id,quote FROM fact_evidence_links WHERE fact_id=?",(row["fact_id"],))]
        maintenance=item.get("maintenance",{})
        mapping = db.execute("SELECT status,data FROM fact_dimension_results WHERE fact_id=?", (row["fact_id"],)).fetchone() if has_results else None
        if mapping:
            detail = json.loads(mapping["data"])
            mapping_note = ("维度映射待重试" if mapping["status"] == "technical_pending" else
                            "维度映射：" + detail["reason"])
            mapping_note += "".join("；" + x["dimension_id"] + "：" + x["reason"] for x in detail["matches"])
            mapping_note += "".join("；条件待补 " + x["dimension_id"] + "：" + x["reason"] for x in detail["deferred"])
        else:
            mapping_note = "旧版/尚未执行106维语义映射；不能把未分类当作没有记忆"
        cards.append(dict(id=row["fact_id"],dimension=(dims or ["UNCLASSIFIED"])[0],dimension_ids=dims or ["UNCLASSIFIED"],
            status=status,label=labels.get(row["state"],row["state"]),claim=item.get("claim",""),subject=item.get("subject",""),
            mapping_deferred=detail["deferred"] if mapping else [],
            scope=item.get("scope",""),keywords=item.get("keywords",[]),evidence=evidence,checks={},
            reason=maintenance.get("reason","原话支持，不代表客观事实已外部核实") + "。 " + mapping_note,source_decision=row["source_decision"],
            maintenance_action=maintenance.get("action","pending"),storage_control={"status":"blocked" if row["state"]=="suspended" else "allowed"},
            policy="session-reliable-v2",session_id=item.get("origin_session_id",item.get("session_id",""))))
    reader=object.__new__(FactStore); reader.db=db
    sessions, tasks=[],[]
    for row in db.execute("SELECT * FROM fact_sessions ORDER BY session_id"):
        evidence_count=db.execute("SELECT COUNT(*) FROM fact_evidence WHERE session_id=?",(row["session_id"],)).fetchone()[0]
        view=reader.session_result(dict(row,turns=[None]*evidence_count))
        tasks.extend(view["retry_tasks"])
        sessions.append(dict(id=row["session_id"],status="partial" if view["retry_tasks"] else "completed",
            turns=evidence_count,candidates=len(view["candidates"]),stored=len(view["approved"]),
            reason={"partial":"部分完成，有技术待补","evidence_only":"证据已保存，提取未完成",
                    "semantic_pending":"完整处理，有语义待确认","complete_empty":"完整处理，无有效记忆",
                    "complete_with_facts":"完整处理，有有效记忆"}[view["conclusion"]]))
        for audit in db.execute("SELECT * FROM fact_candidate_audits WHERE session_id=? AND fact_id IS NULL",(row["session_id"],)):
            audit_data=json.loads(audit["data"])
            item=audit_data["candidate"]
            discarded=audit_data.get("decision",{}).get("discard",False)
            cards.append(dict(id=audit["candidate_id"],dimension="UNCLASSIFIED",dimension_ids=["UNCLASSIFIED"],
                status="discarded" if discarded else "pending",label="原始候选已去重/修订" if discarded else "候选已保存，尚未完成维护",claim=item.get("claim",""),subject=item.get("subject",""),
                scope=item.get("scope",""),keywords=item.get("keywords",[]),
                evidence=[dict(id=e["evidence_id"],quote=e["quote"]) for e in item.get("evidence",[])],
                checks={},reason=audit_data.get("decision",{}).get("reason","技术工作待补，尚未成为正式记忆"),session_id=row["session_id"],policy="session-reliable-v2"))
    if manifest:
        seen={s["id"] for s in sessions}
        for selected in manifest.get("selected_sessions",[]):
            if selected["session_id"] not in seen:
                sessions.append(dict(id=selected["session_id"],status="not_started",turns=selected["turns"],
                    candidates=0,stored=0,reason="尚未读取处理，不代表空提取"))
    device=db.execute("SELECT device_id FROM fact_sessions LIMIT 1").fetchone()
    return dict(device_id=device[0] if device else (manifest or {}).get("device_id","未开始"),
        run_status="partial" if tasks or any(s["status"]!="completed" for s in sessions) else "completed",
        selected_sessions=len(sessions),committed_sessions=sum(s["status"]=="completed" for s in sessions),
        sessions=sessions,cards=cards,dimensions=dimensions,counts=dict(Counter(c["status"] for c in cards)),
        source_scope="dialogue_and_photo",summaries=[],retry_tasks=tasks,flow_version="fact-driven-v1")
