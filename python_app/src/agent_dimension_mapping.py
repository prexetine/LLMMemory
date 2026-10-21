"""106维下游投影：不改变事实、凭证、支持计数或画像状态。"""
from __future__ import annotations

import json
from pathlib import Path

from agent_context import ContextCapacityError
from agent_data import digest, exact_keys
from agent_stage_runtime import model_signature

VERSION = "fact-dimensions-106-v1"
PROMPT = (Path(__file__).resolve().parents[1] / "prompts/fact_dimension_mapping.txt").read_text(encoding="utf-8")
# 这是计算来源约束，不是根据编号区间猜测语义。
AGGREGATE = {"P01", "P02", "P05", "P06", "P08", "P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16", "E02"}
REPEATED = {"G20", "G39", "G40", "E03", "E08", "E09", "E30", "E31", "E43"}
CROSS_SOURCE = {"E37", "E38", "E39", "E40", "E41"}


def catalog(registry):
    expected = {f"{prefix}{i:02}" for prefix, n in (("G", 46), ("P", 16), ("E", 44)) for i in range(1, n + 1)}
    if set(registry.dimensions) != expected:
        raise ValueError("产品维度目录必须包含完整且唯一的106项")
    return [dict(dimension_id=k, name=v["name"], rules=v["rules"],
                 aggregation_required=k in AGGREGATE, requires_repetition=k in REPEATED,
                 cross_source_required=k in CROSS_SOURCE)
            for k, v in registry.dimensions.items()]


def validate(value, facts, rules):
    exact_keys(value, {"results"})
    if not isinstance(value["results"], list):
        raise ValueError("results必须为列表")
    by_id = {f["fact_id"]: f for f in facts}
    rule_ids = {r["dimension_id"] for r in rules}
    seen = set()
    for row in value["results"]:
        exact_keys(row, {"fact_id", "matches", "deferred", "reason"})
        fid = row["fact_id"]
        if not isinstance(fid, str) or fid not in by_id or fid in seen:
            raise ValueError("缺失、额外或重复fact_id")
        seen.add(fid)
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("映射说明不能为空")
        used = set()
        fact = by_id[fid]
        evidence = {e["evidence_id"] for e in fact["evidence"]}
        for field in ("matches", "deferred"):
            if not isinstance(row[field], list):
                raise ValueError("维度结果必须为列表")
            for link in row[field]:
                exact_keys(link, {"dimension_id", "reason", "evidence_ids"} if field == "matches" else {"dimension_id", "reason"})
                dim = link["dimension_id"]
                if not isinstance(dim, str) or dim not in rule_ids or dim in used:
                    raise ValueError("未知或重复维度")
                used.add(dim)
                if not isinstance(link["reason"], str) or not link["reason"].strip():
                    raise ValueError("维度理由不能为空")
                if field == "deferred":
                    continue
                refs = link["evidence_ids"]
                if (not isinstance(refs, list) or not refs or any(not isinstance(x, str) for x in refs)
                        or len(refs) != len(set(refs)) or not set(refs).issubset(evidence)):
                    raise ValueError("维度引用不属于本事实")
                if dim in AGGREGATE:
                    raise ValueError("统计维度必须由分析记录计算，不能从单条事实映射")
                if dim in REPEATED and (fact["support_sessions"] < 2 or fact["support_dates"] < 2):
                    raise ValueError("长期/重复维度缺少跨会话跨日期支持")
                if dim in CROSS_SOURCE:
                    raise ValueError("当前事实接口没有可验证跨来源关联，必须标记条件待补")
    if seen != set(by_id):
        raise ValueError("缺少输入事实的映射结果")


def ensure_table(db):
    db.execute("""CREATE TABLE IF NOT EXISTS fact_dimension_results(
        fact_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, mapping_version TEXT NOT NULL,
        status TEXT NOT NULL, data TEXT NOT NULL)""")
    db.commit()


def save_result(db, result, fingerprint):
    fid = result["fact_id"]
    with db:
        db.execute("DELETE FROM fact_dimension_maps WHERE fact_id=?", (fid,))
        for link in result["matches"]:
            db.execute("INSERT INTO fact_dimension_maps VALUES(?,?,?,?)",
                       (fid, link["dimension_id"], VERSION, link["reason"]))
        db.execute("INSERT OR REPLACE INTO fact_dimension_results VALUES(?,?,?,?,?)",
                   (fid, fingerprint, VERSION, "mapped" if result["matches"] else "unmapped",
                    json.dumps(result, ensure_ascii=False)))
        if result["matches"]:
            db.execute("DELETE FROM fact_taxonomy_gaps WHERE fact_id=?", (fid,))
        else:
            db.execute("INSERT OR REPLACE INTO fact_taxonomy_gaps VALUES(?,?,?,?)",
                       (fid, "open", result["reason"], json.dumps(result, ensure_ascii=False)))


def project_session(pipeline):
    """只处理本Session涉及的规范事实；已提交事实不会因为分类失败而回滚。"""
    db, session = pipeline.fact_store.db, pipeline.s
    ensure_table(db)
    rules = catalog(pipeline.a.registry)
    ids = {pipeline.fact_store.canonical(r[0]) for r in db.execute(
        "SELECT fact_id FROM fact_candidate_audits WHERE device_id=? AND session_id=? AND fact_id IS NOT NULL",
        (session["device_id"], session["session_id"]))}
    pending, hashes = [], {}
    for fid in sorted(ids):
        row = db.execute("SELECT * FROM fact_assertions WHERE fact_id=?", (fid,)).fetchone()
        if not row or row["source_decision"] != "pass" or row["state"] not in ("accepted", "conflicted"):
            continue
        item = json.loads(row["data"])
        evidence = [dict(evidence_id=e[0], quote=e[1]) for e in db.execute(
            "SELECT evidence_id,quote FROM fact_evidence_links WHERE fact_id=? ORDER BY evidence_id", (fid,))]
        supports = list(db.execute("""SELECT s.session_id,e.data FROM fact_support s
            JOIN fact_evidence e ON s.evidence_id=e.evidence_id WHERE s.fact_id=?""", (fid,)))
        dates = {json.loads(e[1]).get("occurred_at", "")[:10] for e in supports} - {""}
        fact = dict(fact_id=fid, claim=item.get("claim", ""), subject=item.get("subject", ""),
                    subject_role=item.get("subject_role", ""), fact=item.get("fact", {}),
                    scope=item.get("scope", ""), evidence=evidence,
                    support_sessions=len({e[0] for e in supports}), support_dates=len(dates))
        fingerprint = digest([VERSION, PROMPT, rules, fact, model_signature(pipeline.a.config)])
        hashes[fid] = fingerprint
        cached = db.execute("SELECT * FROM fact_dimension_results WHERE fact_id=?", (fid,)).fetchone()
        if cached and cached["status"] in ("mapped", "unmapped") and cached["fingerprint"] == fingerprint:
            try:
                validate({"results": [json.loads(cached["data"])]}, [fact], rules)
                save_result(db, json.loads(cached["data"]), fingerprint)
                pipeline.stats["dimension_mapping_cache_hits"] += 1
                continue
            except (ValueError, TypeError, KeyError):
                pipeline.stats["dimension_mapping_cache_invalid"] += 1
        pending.append(fact)
        with db:
            db.execute("DELETE FROM fact_dimension_maps WHERE fact_id=?", (fid,))
            db.execute("INSERT OR REPLACE INTO fact_dimension_results VALUES(?,?,?,?,?)",
                       (fid, fingerprint, VERSION, "technical_pending",
                        json.dumps({"fact_id": fid, "matches": [], "deferred": [],
                                    "reason": "维度投影尚未完成，不影响已审核事实"}, ensure_ascii=False)))

    def process(batch):
        try:
            response = pipeline.ask("dimension_mapper", PROMPT, {"catalog": rules, "facts": batch},
                                    lambda v: validate(v, batch, rules))
            for result in response["results"]:
                save_result(db, result, hashes[result["fact_id"]])
        except ContextCapacityError as exc:
            if len(batch) > 1:
                mid = len(batch) // 2
                process(batch[:mid])
                process(batch[mid:])
            else:
                pipeline._task("dimension_mapping", [], str(exc), "context_capacity", unit_id=batch[0]["fact_id"])
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            pipeline._task("dimension_mapping", [], str(exc), "mapping_protocol",
                           unit_id=digest([f["fact_id"] for f in batch]))

    size = pipeline.a.config.get("dimension_mapping_batch_size", 8)
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("dimension_mapping_batch_size必须为正整数")
    for start in range(0, len(pending), size):
        process(pending[start:start + size])
    # 维度投影不触碰事实和支持计数；技术失败仅影响处理完整性。
    with db:
        db.execute("DELETE FROM fact_tasks WHERE device_id=? AND session_id=? AND stage='dimension_mapping'",
                   (session["device_id"], session["session_id"]))
        for task in pipeline.fact_tasks:
            if task["stage"] == "dimension_mapping":
                db.execute("INSERT OR REPLACE INTO fact_tasks VALUES(?,?,?,?,?,?,?)",
                           (task["task_id"], session["device_id"], session["session_id"],
                            task["stage"], task["state"], task["reason"], json.dumps(task, ensure_ascii=False)))
        db.execute("UPDATE fact_sessions SET status=? WHERE device_id=? AND session_id=?",
                   ("closed_with_tasks" if pipeline.fact_tasks else "closed",
                    session["device_id"], session["session_id"]))
