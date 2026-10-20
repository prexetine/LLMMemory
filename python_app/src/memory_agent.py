"""真实中心Agent Demo：动态业务Skill、历史、复核与持久化。"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
import tracemalloc
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from agent_data import Registry, SKILL_NAMES, digest, exact_keys, load_sessions, load_user_roster
from agent_llm import JsonClient
from agent_usage import session_usage
from agent_store import MemoryStore
from agent_parallel import batch_lock, iter_users, stop_requested
from agent_context import BudgetClient, bounded_history, compact_feedback, setting, input_budget
from agent_photos import load_photo_sessions, bind_image, options as photo_options
from agent_v3_contracts import ModelOutputError
from agent_terminal_log import logged_run
from agent_flow_data import validate_flow, semantic_config, PROCESSING_REVISION
from agent_result import summarize_results, summarize_calls

BASE = Path(__file__).resolve().parents[2]


def execute_session(agent, session: dict, out: Path, config: dict, client) -> dict:
    """Session级异常隔离：恢复一次，仍失败落盘上报；不可安全保存则停止。"""
    agent.session_attempt_budget = Counter()
    errors = []
    for attempt in (1, 2):
        stage = "photo_binding" if session.get("source_kind") == "photo" else "extraction"
        try:
            if stage == "photo_binding":
                bind_image(session, out, config, client)
            stage = "extraction"
            result = agent.run_session(session)
            result.update(source_kind=session.get("source_kind", "dialogue"), session_attempts=attempt)
            if errors:
                result["recovered_errors"] = errors
            return result
        except (MemoryError, sqlite3.DatabaseError):
            raise
        except Exception as exc:
            if isinstance(exc, OSError) and stage != "photo_binding":
                raise  # 存储/日志无法写入，不能宣称可恢复地继续。
            agent.store.db.rollback()
            error = dict(attempt=attempt, stage=stage, code=type(exc).__name__, reason=str(exc))
            errors.append(error)
            if attempt == 1:
                logging.warning("Session=%s stage=%s首次异常：%s；利用检查点重试一次", session["session_id"], stage, exc, exc_info=True)
                continue
            logging.error("Session=%s重试后仍失败：%s；上报并继续后续Session", session["session_id"], exc, exc_info=True)
            result = dict(session_id=session["session_id"], source_kind=session.get("source_kind", "dialogue"),
                turns=len(session["turns"]), status="failed", approved=[], reviews=[], candidates=[], coverage={},
                retryable=True, error_category="session_exception", error=f"{type(exc).__name__}: {exc}",
                session_attempts=attempt, errors=errors, flow_version=config["flow_version"], processing_revision='bge-relations-2026-10-15-1' if config['flow_version']=='generic-flow-v3-relations' else PROCESSING_REVISION,
                execution_status="awaiting_retry", commit_status="not_committed",
                retry_tasks=[dict(stage=stage, session_id=session["session_id"], candidate_ids=[], evidence_ids=[],
                    code=type(exc).__name__, retryable=True, status="retryable", reason=str(exc))])
            files = ["session_failures.jsonl"]
            if session.get("source_kind") == "photo":
                files.append("photo_failures.jsonl")
            for name in files:
                with (out / name).open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result
        finally:
            client.image_assets = {}


class CentralAgent:
    """仅绑定基础服务；业务执行只有GenericPipeline一条路径。"""
    def __init__(self, client: Any, registry: Registry, store: MemoryStore, config: dict, trace_path: Path):
        validate_flow(config)
        self.client, self.registry, self.store = BudgetClient(client, config), registry, store
        self.config, self.trace_path = config, trace_path

    def run_session(self, session: dict) -> dict:
        from agent_generic_flow import GenericPipeline
        return GenericPipeline(self, session).run()


def policy_version(config: dict) -> str:
    validate_flow(config)
    from agent_contracts import VERSION
    return VERSION


def select_devices(config: dict) -> list[str]:
    """从核心初选名单解析全部/指定用户；不读取密钥，不调用模型。"""
    if "device_id" in config:
        raise ValueError("user_selection不能与旧device_id同时设置")
    selection = config["user_selection"]
    exact_keys(selection, {"mode", "device_ids"})
    mode, ids = selection["mode"], selection["device_ids"]
    if mode not in ("all", "selected") or not isinstance(ids, list):
        raise ValueError("选择模式必须为all或selected，device_ids必须为数组")
    if any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
        raise ValueError("设备ID必须是非空安全字符串")
    if mode == "all" and ids or mode == "selected" and not ids:
        raise ValueError("all模式ID列表须为空；selected模式至少选择一个用户")
    if len(set(ids)) != len(ids):
        raise ValueError("设备ID重复，请删除重复项")
    root = BASE / config["data_root"]
    available = [row["device_id"] for row in load_user_roster(root)]
    if (not available or len(set(available)) != len(available)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in available)):
        raise ValueError("核心用户名单为空、重复或含非法设备ID")
    chosen = available if mode == "all" else ids
    if set(chosen) - set(available):
        raise ValueError("用户不在核心初选名单：" + ",".join(sorted(set(chosen) - set(available))))
    for device in chosen:
        if not (root / "candidate_data" / device / "dialogue.jsonl.gz").is_file():
            raise ValueError("缺少用户对话包：" + device)
    return chosen


def load_inputs(config: dict, device: str) -> tuple[list[dict], dict]:
    sessions, manifest = load_sessions(BASE / config["data_root"], device, config["gap_minutes"])
    if photo_options(config):
        pictures, photo_manifest = load_photo_sessions(BASE / config["data_root"], device, config)
        manifest.update(photo_manifest, dialogue_sessions=len(sessions))
        sessions = sorted(sessions + pictures, key=lambda s: (s["start"], s["session_id"]))
    return sessions, manifest


def run(config_path: Path, mode: str, output_override: str | None = None, resume: bool = False) -> int:
    """批次加独占锁；旧单用户入口保持兼容。"""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_flow(config)
    if mode == 'live' and config['flow_version'] == 'generic-flow-v3-relations':
        from bge_lifecycle import managed_encoder
        with managed_encoder(config['retrieval']):
            return run_dispatch(config_path, config, mode, output_override, resume)
    return run_dispatch(config_path, config, mode, output_override, resume)


def run_dispatch(config_path: Path, config: dict, mode: str, output_override: str | None, resume: bool) -> int:
    """服务就绪后再调度；服务租约覆盖用户进程和HTML导出。"""
    if "user_selection" not in config:
        code = run_one(config, mode, output_override, resume)
        if mode == "live" and config.get("export_html"):
            from export_memory_html import export_user
            out = Path(output_override) if output_override else BASE / config["output_dir"]
            try:
                export_user(out,out/"index.html",rules=BASE/config["rules_file"])
            except Exception as exc:
                report_path=out/"report.json"
                report=json.loads(report_path.read_text(encoding="utf-8"))
                report.update(status="partial",export_status="failed",export_error=str(exc))
                report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8",newline="\n")
                logging.error("HTML导出失败：%s",exc)
                return 2
        return code
    out = Path(output_override) if output_override else BASE / config["output_dir"]
    with batch_lock(out):
        return run_batch(config_path, mode, output_override, resume)


@logged_run
def run_batch(config_path: Path, mode: str, output_override: str | None = None, resume: bool = False) -> int:
    """兼容旧单用户配置；新配置使用逐用户隔离的批次目录。"""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    flow = validate_flow(config)
    if "user_selection" not in config:
        return run_one(config, mode, output_override, resume)
    devices = select_devices(config)
    workers = config.get("parallel_users", 1)
    if type(workers) is not int or not 1 <= workers <= 3:
        raise ValueError("parallel_users必须为1至3的整数")
    if not isinstance(config.get("export_html", False), bool):
        raise ValueError("export_html必须为布尔值")
    if mode not in ("prepare", "live") or resume and mode != "live":
        raise ValueError("仅live支持续跑")
    out = Path(output_override) if output_override else BASE / config["output_dir"]
    settings = {k: v for k, v in config.items() if k not in ("output_dir", "parallel_users")}
    settings = semantic_config(settings)
    batch = {"device_ids": devices, "config_hash": digest(settings),
             "review_version": policy_version(config), "initial_mode": mode, "flow_version": flow}
    manifest_path = out / "batch_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise ValueError("缺少batch_manifest.json，无法安全续跑；请使用新目录首次运行（不加--resume）")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_flow(config, old)
        if (old.get("initial_mode") != "live"
                or any(old.get(k) != batch[k] for k in ("device_ids", "config_hash", "review_version"))):
            raise ValueError("续跑的用户范围、配置或版本变化；请使用新目录")
    elif out.exists() and any(p.name not in (".batch.lock", "logs") for p in out.iterdir()):
        raise ValueError("批次输出目录非空，请换新目录或显式--resume")
    out.mkdir(parents=True, exist_ok=True)
    if not resume:
        manifest_path.write_text(json.dumps(batch, ensure_ascii=False, indent=2),
                                 encoding="utf-8", newline="\n")
    rows = []
    logging.info("开始前用户清单（以下统计完成后才启动用户任务）：")
    planned_total = 0
    for position, device in enumerate(devices, 1):
        preview, preview_manifest = load_inputs(config, device)
        selected_count = len(configured_sessions(preview, config, device))
        planned_total += selected_count
        logging.info("[%s/%s] 用户=%s 有效Session总数=%s 本次处理Session=%s",
                     position, len(devices), device, len(preview), selected_count)
        if photo_options(config):
            logging.info("用户=%s 对话Session=%s 图片Session=%s 图片排除=%s", device,
                         preview_manifest["dialogue_sessions"], preview_manifest["photo_sessions"], preview_manifest["photo_excluded"])
        del preview
    logging.info("清单总计：用户=%s，本次处理Session=%s", len(devices), planned_total)
    logging.info("选中%s个用户；并行%s人；会话=%s；累计调用预算=%s",
                 len(devices), workers, config["max_sessions"],
                 "无上限（已授权）" if config["max_calls"] is None else config["max_calls"])
    def save_summary(final: bool = False) -> dict:
        ordered = sorted(rows, key=lambda r: devices.index(r["device_id"]))
        complete = len(rows) == len(devices) and all(r["exit_code"] == 0 for r in rows)
        summary = {"selected_users": len(devices), "processed_users": len(rows),
                   "parallel_users": workers,
                   "status": ("completed" if complete else "partial") if final else "running",
                   "users": ordered}
        summary["export_status"] = ("completed" if ordered and all(r.get("html") for r in ordered) else "failed" if final else "pending") if mode == "live" and config.get("export_html") else "not_requested"
        (out / "batch_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                               encoding="utf-8", newline="\n")
        return summary
    for row in iter_users(config, mode, out, devices, resume, workers):
        rows.append(row)
        if mode == "live" and config.get("export_html"):
            from export_memory_html import export_index
            try:
                export_index(out, rows, rules=BASE / config["rules_file"])
                for finished in rows:
                    finished["html"] = str(out / "index.html")
                logging.info("多用户汇总HTML已更新：%s", out / "index.html")
            except Exception as exc:
                row.update(exit_code=2, html=None, error=f"HTML导出失败：{exc}")
                logging.error("汇总HTML导出失败：%s", exc)
        save_summary()
    summary = save_summary(final=True)
    return 0 if summary["status"] == "completed" else 2


def select_sessions(sessions: list[dict], limit: int | str) -> list[dict]:
    """all选择全部会话；正整数兼容原先小样本模式。"""
    if limit == "all":
        return sessions
    if type(limit) is not int or limit <= 0:
        raise ValueError("max_sessions必须是正整数或字符串all")
    return sessions[:limit]


def configured_sessions(sessions: list[dict], config: dict, device: str) -> list[dict]:
    """可选按设备精确选择Session；缺失或拼写错误时拒绝退回全量。"""
    selection = config.get("session_ids_by_device")
    if selection is not None:
        if not isinstance(selection, dict) or not selection or any(
            not isinstance(key, str) or not key or not isinstance(ids, list) or not ids
            or any(not isinstance(sid, str) or not sid for sid in ids)
            or len(set(ids)) != len(ids) for key, ids in selection.items()
        ):
            raise ValueError("session_ids_by_device必须为设备ID到非空、无重复Session ID列表的映射")
        if device not in selection:
            raise ValueError(f"session_ids_by_device缺少用户{device}，禁止全量回退")
        wanted = set(selection[device])
        missing = wanted - {s["session_id"] for s in sessions}
        if missing:
            raise ValueError(f"用户{device}指定Session不存在：{sorted(missing)}")
        sessions = [s for s in sessions if s["session_id"] in wanted]
    return select_sessions(sessions, config["max_sessions"])


@logged_run
def run_one(config: dict, mode: str, output_override: str | None = None, resume: bool = False) -> int:
    """单用户执行引擎；每用户独立模型预算、数据库及日志。"""
    validate_flow(config)
    for name in ("gap_minutes", "timeout_seconds") + (("max_output_tokens",) if "max_output_tokens" in config else ()):
        if not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"配置{name}必须为正整数")
    for name in ("max_calls", "max_total_input_chars"):
        value = config[name]
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"配置{name}必须为正整数或null（无累计上限）")
    registry = Registry(BASE / config["rules_file"])
    input_budget(config)
    for name in ("session_chunk_tokens", "history_context_tokens", "rule_page_tokens", "context_split_retries"):
        setting(config, name)
    select_sessions([], config["max_sessions"])
    sessions, manifest = load_inputs(config, config["device_id"])
    total_sessions = len(sessions)
    sessions = configured_sessions(sessions, config, config["device_id"])
    out = Path(output_override) if output_override else BASE / config["output_dir"]
    previous_report: dict = {}
    context_settings = {k: v for k, v in config.items() if k not in ("output_dir", "parallel_users")}
    context_config_hash = digest(semantic_config(context_settings))
    if resume:
        if mode != "live":
            raise ValueError("仅live支持断点续跑")
        previous_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        validate_flow(config, previous_manifest)
        if previous_manifest.get("review_version") != policy_version(config):
            raise ValueError("旧运行未采用当前逐条Review，禁止直接续跑升级；请使用新输出目录")
        if previous_manifest.get("context_config_hash") != context_config_hash:
            raise ValueError("续跑上下文/模型/处理配置变化，请使用新输出目录")
        expected = [(s["session_id"], s["input_hash"]) for s in sessions]
        actual = [(s["session_id"], s["input_hash"]) for s in previous_manifest["selected_sessions"]]
        if previous_manifest["device_id"] != config["device_id"] or expected != actual:
            raise ValueError("续跑的设备、输入或维度版本与原运行不一致")
        # v2未完成窗口仅在context_work暂存，未写正式sessions，允许安全续跑。
        from agent_recovery import resume_report
        previous_report = resume_report(out, sessions)
    elif out.exists() and any(p.name != "logs" for p in out.iterdir()):
        raise ValueError("输出目录非空；为防止覆盖请指定新的--output目录")
    completed_ids = {r["session_id"] for r in previous_report.get("results", [])
                     if r["status"] == "completed"}
    completed_count = sum(s["session_id"] in completed_ids for s in sessions)
    logging.info("用户=%s 会话统计：有效Session总数=%s，本次选中=%s，已完成=%s，待处理=%s",
                 config["device_id"], total_sessions, len(sessions), completed_count,
                 len(sessions) - completed_count)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({
        **manifest, "selected_sessions": [{"session_id": s["session_id"], "turns": len(s["turns"]),
        "start": s["start"], "end": s["end"], "input_hash": s["input_hash"]} for s in sessions],
        "registry_version": registry.version, "skill_catalog": registry.catalog(), "mode": mode,
        "review_version": policy_version(config),
        "flow_version": config.get("flow_version", "legacy"),
        "context_config_hash": context_config_hash,
    }, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    if mode == "prepare":
        print(json.dumps({"status": "prepared_not_extracted", "output": str(out),
                          "sessions": len(sessions), "dimensions": len(registry.dimensions)}, ensure_ascii=False))
        return 0
    store = MemoryStore(out / "memory.sqlite3")
    from agent_maintenance import V4Store
    V4Store(store)
    started = time.perf_counter()
    tracemalloc.start()
    results, failed = list(previous_report.get("results", [])), None
    client = None
    try:
        client = JsonClient(config, BASE / config["credentials_file"], out / "model_calls.jsonl")
        agent = CentralAgent(client, registry, store, config, out / "trace.jsonl")
        for session_index, s in enumerate(sessions, 1):
            if stop_requested():
                raise KeyboardInterrupt()
            if any(r["session_id"] == s["session_id"] for r in results):
                row = store.db.execute("SELECT input_hash,status FROM sessions WHERE device_id=? AND session_id=?",
                                       (s["device_id"], s["session_id"])).fetchone()
                if row is None or row["input_hash"] != s["input_hash"] or row["status"] != "completed":
                    raise ValueError("已完成报告与数据库状态不一致")
                logging.info("续跑跳过已完成session=%s", s["session_id"])
                continue
            logging.info("用户=%s Session=%s/%s 处理session=%s turns=%s",
                         config["device_id"], session_index, len(sessions),
                         s["session_id"], len(s["turns"]))
            with session_usage(out / "model_calls.jsonl", config["device_id"], s["session_id"]) as usage_state:
                result = execute_session(agent, s, out, config, client)
                usage_state["status"] = result.get("execution_status") or ("completed_with_pending_extraction" if result.get("needs_attention") else result.get("status", "unknown"))
                results.append(result)
            store.export_book(out / "记忆书页.md", registry)
    except KeyboardInterrupt:
        failed = "interrupted: 用户或运行器中断，未完成链路"
        logging.error(failed)
    except Exception as exc:
        failed = f"{type(exc).__name__}: {exc}"
        logging.error("运行未完成：%s", failed)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        all_calls = []
        if (out / "model_calls.jsonl").exists():
            all_calls = [json.loads(line) for line in (out / "model_calls.jsonl").read_text(encoding="utf-8").splitlines()]
        total_wall = previous_report.get("elapsed_seconds", 0) + time.perf_counter() - started
        peak = max(peak, previous_report.get("python_traced_peak_bytes", 0))
        report = {"status": "failed" if failed else ("completed" if all(r["status"] == "completed" and not r.get("needs_attention") for r in results) else "partial"),
                  "extraction_pending": sum(len(r.get("extraction_issues", [])) for r in results),
                  "flow_version": config.get("flow_version", "legacy"),
                  "retryable_tasks": sum(len(r.get("retry_tasks", [])) for r in results),
                  "committed_sessions": sum(r["status"] == "completed" for r in results),
                  "error": failed, "device_id": config["device_id"], "processed_sessions": len(results),
                  "selected_sessions": len(sessions), "results": results,
                  "model_calls": len(all_calls), "tokens": sum(int(c.get("usage", {}).get("total_tokens", 0)) for c in all_calls),
                  "llm_seconds": round(sum(c.get("elapsed_seconds", 0) for c in all_calls), 3),
                  "elapsed_seconds": round(total_wall, 3),
                  "attempts": previous_report.get("attempts", 1 if previous_report else 0) + 1,
                  "python_traced_peak_bytes": peak, "parameters_flops": "远端不可获取",
                  "source_scope": "dialogue_and_photo" if photo_options(config) else "dialogue_only", "review_version": policy_version(config),
                  "photo_sessions": sum(s.get("source_kind") == "photo" for s in sessions),
                  "photo_completed": sum(r.get("source_kind") == "photo" and r["status"] == "completed" for r in results),
                  "batch_metrics": {role: sum(r.get("batch_metrics", {}).get(role, 0) for r in results)
                                    for role in sorted({k for r in results for k in r.get("batch_metrics", {})})},
                  "actual_batches": {role: sum(c.get("role") == role for c in all_calls)
                                     for role in ("source_reviewer", "history_compare", "source_relations")},
                  "actual_calls_by_role": {role: sum(c.get("role") == role for c in all_calls)
                                           for role in sorted({c.get("role", "unknown") for c in all_calls})},
                  "metrics_note": "batch_metrics为已返回Session的流程计数（含缓存标记）；actual_calls_by_role来自实际调用日志，包含失败重试。",
                  "recovery_note": previous_report.get("recovery_note"),
                  "quality": "未经过独立人工标注，非106维度质量验收"}
        report.update(summarize_results(results,len(sessions)))
        report["actual_stage_metrics"]=summarize_calls(all_calls,results)
        if not failed and (report["technical_pending_sessions"] or report["unvisited_sessions"]):
            report["status"]="partial"
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        lines = ["# 单用户记忆Agent运行报告", "",
                 f"- 用户设备：{config['device_id']}", f"- 状态：{report['status']}",
                 f"- 会话：{len(results)}/{len(sessions)}；模型调用：{report['model_calls']}；token：{report['tokens']}",
                 f"- 耗时：{report['elapsed_seconds']}秒；Python追踪峰值：{peak}字节（不含远端模型）",
                 f"- 会话选择：{config['max_sessions']}；实际选中{len(sessions)}个处理单元（图片一张一个Session）；图片完成{report['photo_completed']}/{report['photo_sessions']}；定位、通话和统计聚合未处理，向量disabled。",
                 "- 正式记忆均通过逐条Review；不是人工确认或外部事实核实。旧候选不自动升级。", ""]
        if failed:
            lines.append(f"失败原因：{failed}")
        for field, label in (("candidate_outcomes","候选去向"),("admitted_candidate_retention","已入库候选类型（含强化）"),("processing_versions","处理版本/Session数")):
            lines.append(f"- {label}：{json.dumps(report[field],ensure_ascii=False)}")
        if report["mixed_versions"]:
            lines.append("- 本报告含旧已提交与新恢复结果；旧结果没有重新审核或重新计数。")
        for r in results:
            lines += [f"## {r['session_id']}", "", f"话轮：{r['turns']}；状态：{r['status']}；正式入库：{len(r['approved'])}", ""]
            for review in r.get("reviews", []):
                lines.append(f"- Review {review['candidate_id']}：{review['decision']}；{review['reason']}")
            for c in r["approved"]:
                lines.append(f"- {c['dimension_id']} {c['claim'].replace(chr(10), ' ')}（{c['operation']}）")
            lines.append("")
        (out / "运行报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        store.close()
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE / "python_app/config/memory_agent_generic_flow.json")
    parser.add_argument("--mode", choices=("prepare", "live"), default="prepare")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true", help="校验相同输入后跳过已完成会话，继续未提交会话")
    args = parser.parse_args()
    sys.exit(run(args.config, args.mode, args.output, args.resume))
