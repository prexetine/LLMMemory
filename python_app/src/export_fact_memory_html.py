"""事实驱动库的独立 HTML 投影；不读取或修改旧 memories 表。"""
from __future__ import annotations

import html
import json
from pathlib import Path


def export_fact_html(db, destination: Path) -> None:
    rows = db.execute("SELECT fact_id,state,source_decision,data FROM fact_assertions ORDER BY last_seen DESC,fact_id").fetchall()
    has_mapping_results = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='fact_dimension_results'").fetchone()
    cards = []
    for row in rows:
        item = json.loads(row["data"])
        evidence = db.execute("SELECT evidence_id,quote FROM fact_evidence_links WHERE fact_id=?", (row["fact_id"],)).fetchall()
        mapping = [entry[0] for entry in db.execute("SELECT dimension_id FROM fact_dimension_maps WHERE fact_id=? ORDER BY dimension_id", (row["fact_id"],))]
        mapping_result = db.execute("SELECT status,data FROM fact_dimension_results WHERE fact_id=?", (row["fact_id"],)).fetchone() if has_mapping_results else None
        mapping_note = "旧版/尚未执行106维语义映射"
        if mapping_result:
            detail = json.loads(mapping_result["data"])
            mapping_note = ("维度投影技术待补；" if mapping_result["status"] == "technical_pending" else "") + detail["reason"]
            mapping_note += "".join("；" + x["dimension_id"] + "：" + x["reason"] for x in detail["matches"])
            mapping_note += "".join("；条件待补 " + x["dimension_id"] + "：" + x["reason"] for x in detail["deferred"])
        projection = db.execute("SELECT state,data FROM fact_projections WHERE fact_id=? AND projection_type='profile'", (row["fact_id"],)).fetchone()
        cards.append("<article class='card'><div class='meta'>%s · 来源审核：%s · 画像：%s</div><h3>%s</h3><p>主体：%s　范围：%s</p><p>分类：%s</p><p>106映射：%s</p><p>映射说明：%s</p><details><summary>证据与处理依据</summary>%s</details></article>" % (
            html.escape(row["state"]), html.escape(row["source_decision"]), html.escape(projection["state"] if projection else "未投影"),
            html.escape(item.get("claim", "")), html.escape(item.get("subject", "")), html.escape(item.get("scope", "")),
            html.escape(" / ".join(item.get("facets", {}).values())), html.escape("、".join(sorted(set(mapping))) or "未匹配"), html.escape(mapping_note),
            "".join("<p><code>%s</code> %s</p>" % (html.escape(e["evidence_id"]), html.escape(e["quote"])) for e in evidence)))
    observations = db.execute("SELECT data FROM analysis_observations ORDER BY period_end DESC").fetchall()
    observation_html = "".join("<li>%s（%s，覆盖%s天）</li>" % (html.escape(json.loads(row["data"])["claim"]), html.escape(json.loads(row["data"])["state"]), html.escape(str(json.loads(row["data"])["distinct_dates"]))) for row in observations)
    gaps = db.execute("SELECT COUNT(*) FROM fact_taxonomy_gaps WHERE state='open'").fetchone()[0]
    document = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>事实驱动用户记忆</title><style>
body{margin:0;background:#f6f8fb;color:#172033;font:15px system-ui,'Microsoft YaHei',sans-serif}.wrap{max-width:1000px;margin:auto;padding:36px 22px}.hero{padding:26px;border-radius:20px;background:linear-gradient(135deg,#173b6c,#287a80);color:#fff}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-top:20px}.card{background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:18px;box-shadow:0 4px 18px #1720330c}.meta{color:#087e8b;font-size:13px}h3{margin:12px 0 8px}p{line-height:1.65;margin:6px 0}summary{cursor:pointer;color:#345}</style><main class='wrap'><section class='hero'><h1>事实驱动用户记忆</h1><p>事实、审核、关系和产品维度映射彼此独立；本次有%s条待治理维度映射。</p></section><section><h2>分析观察</h2><ul>%s</ul></section><section class='grid'>%s</section></main></html>""" % (gaps, observation_html or "<li>暂无跨日期分析观察。</li>", "".join(cards or ["<p>暂无事实记录。</p>"]))
    session_rows = []
    for session in db.execute("SELECT * FROM fact_sessions ORDER BY session_id"):
        tasks = db.execute("SELECT stage,reason FROM fact_tasks WHERE session_id=? AND state='technical_retry'",
                           (session["session_id"],)).fetchall()
        audit = db.execute("SELECT source_decision,maintenance_action FROM fact_candidate_audits WHERE session_id=?",
                           (session["session_id"],)).fetchall()
        evidence_count = db.execute("SELECT COUNT(*) FROM fact_evidence WHERE session_id=?", (session["session_id"],)).fetchone()[0]
        conclusion = ("部分完成，有技术待补" if audit else "证据已保存，提取未完成") if tasks or session["status"] != "closed" else (
            "完整处理，有语义待确认" if any(a["maintenance_action"] == "provisional" and a["source_decision"] != "reject" for a in audit)
            else "完整处理，有有效记忆" if any(a["maintenance_action"] != "provisional" for a in audit) else "完整处理，无有效记忆")
        from agent_fact_store import FactStore
        # 不实例化存储对象，导出只能读取，不执行建表或提交。
        reader = object.__new__(FactStore)
        reader.db = db
        view = reader.session_result(dict(session, turns=[None] * evidence_count))
        if session["status"] in ("closed", "closed_with_tasks"):
            conclusion = {"partial":"部分完成，有技术待补","evidence_only":"证据已保存，提取未完成",
                          "semantic_pending":"完整处理，有语义待确认","complete_empty":"完整处理，无有效记忆",
                          "complete_with_facts":"完整处理，有有效记忆"}[view["conclusion"]]
        summary = view["session_summary"]
        session_rows.append("<li>%s：%s；证据%s，候选%s，有效事实引用%s，拒绝%s，语义待确认%s，技术待补%s<ul>%s</ul></li>" % (
            html.escape(session["session_id"]), conclusion, evidence_count, len(audit),
            summary["effective_count"], summary["rejected_count"], summary["semantic_pending_count"], len(tasks),
            "".join("<li>%s：%s</li>" % (html.escape(t["stage"]), html.escape(t["reason"])) for t in tasks)))
    document = document.replace("</main>", "<section><h2>Session处理结论</h2><ul>" + "".join(session_rows) + "</ul></section></main>")
    destination.write_text(document, encoding="utf-8", newline="\n")
    from fact_report_view import build_fact_report
    from export_memory_html import export_report
    manifest_path=destination.parent / "manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    export_report([build_fact_report(db,manifest)], destination)


def export_fact_index(batch_dir: Path, rows: list[dict]) -> None:
    """批次首页仅链接各用户独立事实视图，不跨用户聚合原始记忆。"""
    links = []
    for row in rows:
        device = row["device_id"]
        report = batch_dir / device / "report.json"
        status = "运行中"
        if report.is_file():
            status = json.loads(report.read_text(encoding="utf-8")).get("status", status)
        links.append("<li><a href='%s/index.html'>%s</a> <small>%s</small></li>" % (html.escape(device), html.escape(device), html.escape(status)))
    page = "<!doctype html><meta charset='utf-8'><title>事实驱动记忆批次</title><style>body{font:16px system-ui;margin:40px;background:#f6f8fb}main{max-width:760px;background:#fff;border-radius:16px;padding:28px}li{margin:14px 0}a{color:#087e8b}</style><main><h1>事实驱动记忆批次</h1><p><a href='index_report.html'>打开多用户汇报版 · 维度与具体记忆</a></p><p>每位用户的数据与事实库独立保存。</p><ul>%s</ul></main>" % "".join(links)
    (batch_dir / "index.html").write_text(page, encoding="utf-8", newline="\n")
    import sqlite3
    from contextlib import closing
    from fact_report_view import build_fact_report
    from export_memory_html import export_report
    users = []
    for row in rows:
        database = batch_dir / row["device_id"] / "memory.sqlite3"
        if database.is_file():
            with closing(sqlite3.connect(database.as_uri()+"?mode=ro",uri=True)) as db:
                db.row_factory=sqlite3.Row
                manifest_path=database.parent / "manifest.json"
                manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
                users.append(build_fact_report(db,manifest))
    if users:
        export_report(users, batch_dir / "index.html")
