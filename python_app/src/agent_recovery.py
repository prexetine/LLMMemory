"""从已提交事务恢复报告；报告不是恢复的真源，不执行提取或重新提交。"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

from agent_contracts import VERSION
from agent_data import digest


def committed_result(db, session: dict) -> dict | None:
    row = db.execute('SELECT input_hash,status,batch_id,coverage FROM sessions WHERE device_id=? AND session_id=?',
                     (session['device_id'], session['session_id'])).fetchone()
    if row is None:
        return None
    if row[0] != session['input_hash'] or row[1] != 'completed':
        raise ValueError('数据库Session与输入或提交状态不一致，禁止自动重跑')
    bid = digest([VERSION, session['device_id'], session['session_id'], session['input_hash']])
    if row[2] != bid:
        raise ValueError('Session事务标识不匹配')
    receipt_row = db.execute('SELECT data FROM v3_receipts WHERE batch_id=?', (bid,)).fetchone()
    plan_row = db.execute('SELECT data FROM v3_plans WHERE batch_id=?', (bid,)).fetchone()
    if not receipt_row or not plan_row:
        raise ValueError('已完成Session缺少原子回执或维护计划，禁止重新入库')
    receipt, plan = json.loads(receipt_row[0]), json.loads(plan_row[0])
    if receipt.get('status') != 'committed' or plan.get('input_hash') != session['input_hash']:
        raise ValueError('提交回执/维护计划状态不一致')
    reviews = [json.loads(r[0]) for r in db.execute('SELECT data FROM reviews WHERE batch_id=? ORDER BY rowid', (bid,))]
    candidates = [r['candidate'] for r in reviews]
    admitted = {m['candidate_id']: m for m in receipt['mapping']}
    return dict(session_id=session['session_id'], turns=len(session['turns']), status='completed', policy=VERSION,
                candidates=candidates, approved=[dict(c, operation=admitted[c['candidate_id']]['operation'])
                    for c in candidates if c['candidate_id'] in admitted], reviews=reviews,
                maintenance=plan['decisions'], signals=plan.get('signals', []), coverage=json.loads(row[3]),
                commit=receipt, extraction_issues=plan.get('extraction_issues', []),
                needs_attention=bool(plan.get('extraction_issues')), retry_tasks=[],
                processing_revision=plan.get('processing_revision', 'pre-contract-fix'),
                flow_version=plan.get('flow_version'), source_kind=session.get('source_kind', 'dialogue'),
                execution_status='completed', commit_status='committed',
                batch_metrics={'recovered_from_atomic_receipt': 1},
                data_completeness=session.get('source_kind', 'dialogue') + '_only')


def resume_report(out: Path, sessions: list[dict]) -> dict:
    """只读恢复缺失、损坏或滞后的报告；未提交Session绝不冒充完成。"""
    path = out / 'report.json'
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(report, dict) or not isinstance(report.get('results', []), list):
            raise ValueError('报告结构非法')
    except (FileNotFoundError, ValueError) as exc:
        logging.warning('最终报告不可用，将从数据库事务恢复：%s', type(exc).__name__)
        report = {'recovery_note': '报告缺失或损坏；历史墙钟耗时未知，未计为零耗时执行'}
    completed = {r['session_id']: r for r in report.get('results', []) if r.get('status') == 'completed'}
    database = out / 'memory.sqlite3'
    if not database.exists():
        raise ValueError('续跑缺少数据库；不能仅凭报告确定已完成状态')
    recovered = []
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='session_protocol'").fetchone():
            return resume_facts(db, sessions, report)
        for session in sessions:
            saved = committed_result(db, session)
            if saved is None:
                if session['session_id'] in completed:
                    raise ValueError('已完成报告与数据库状态不一致')
                continue
            old = completed.get(session['session_id'])
            if old is not None and old.get('commit') != saved['commit']:
                raise ValueError('已完成报告与事务回执不一致')
            recovered.append(old or saved)
    report['results'] = recovered
    logging.info('续跑从事务核实已提交Session=%s；这些会话不会重新提取或增加支持次数', len(recovered))
    return report


def resume_facts(db, sessions, report):
    """新事实库只读恢复：数据库是状态真源，报告只贡献历史用量。"""
    from agent_fact_store import FactStore
    db.row_factory = sqlite3.Row
    if db.execute("SELECT version FROM session_protocol").fetchone()[0] != "session-reliable-v2":
        raise ValueError("旧事实协议仅支持只读导出，禁止跨协议续跑")
    reader = object.__new__(FactStore)
    reader.db = db
    old = {r["session_id"]: r for r in report.get("results", [])}
    results = []
    for session in sessions:
        rows = db.execute("SELECT input_hash,status FROM fact_sessions WHERE device_id=? AND session_id=?",
                          (session["device_id"],session["session_id"])).fetchall()
        if not rows:
            continue
        if len(rows) != 1 or rows[0]["input_hash"] != session["input_hash"]:
            raise ValueError("事实库Session输入指纹不一致")
        view = reader.session_result(session)
        for key in ("usage", "elapsed_seconds", "model_calls", "tokens"):
            if key in old.get(session["session_id"], {}):
                view[key] = old[session["session_id"]][key]
        results.append(view)
    report["results"] = results
    report["recovery_note"] = "状态由事实库恢复；缺失的历史耗时与用量保持未知"
    return report
