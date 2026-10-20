"""单文件HTML渲染与导出；数据和旧结果兼容统一由memory_view_data提供。"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
from urllib.parse import quote
from agent_data import Registry
from memory_view_data import build_data, build_report_user
BASE = Path(__file__).resolve().parents[2]


def render(data: dict, template: str) -> str:
    """JSON转义阻止原文中的HTML结束标签逸出数据脚本。"""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    if template.count("__MEMORY_DATA__") != 1:
        raise ValueError("模板必须恰好有一个数据占位符")
    if data.get("internal_view") or any(u.get("internal_view") for u in data.get("users", [])):
        template = template.replace("<body>", '<body><div style="padding:12px 24px;background:#fff3cd;color:#614600">内部查看版：完整显示候选主体、内容、引文及审核原因；审核状态与入库结果不变。</div>')
        template = template.replace("候选的隐私检查未通过或不确定时，汇报文件不嵌入其正文、关键词及原话。这里不是人工隐私审计，请勿公开传播。", "内部查看版保留正文、主体、关键词及原话，隐私检查结果仍显示，不代表审核通过。")
        template = template.replace("受限候选已最小化展示", "本版完整展示受限候选供内部核验")
    template = template.replace("__MEMORY_DETAILS_JS__", (BASE / "python_app/assets/memory_details.js").read_text(encoding="utf-8"))
    return template.replace("__MEMORY_DATA__", payload)


def export_user(folder: Path, output: Path | None = None, rules: Path | None = None, *, internal_view: bool = True) -> Path:
    """从用户独立结果目录离线导出，不修改数据库，也不调用模型。"""
    data = build_data(folder, rules or BASE / "J20核心用户记忆维度与提取存储规划.md", internal_view=internal_view)
    name = "reviewed_dashboard.html" if data["reviewed"] else "memory_dashboard.html"
    template = (BASE / "python_app/assets" / name).read_text(encoding="utf-8")
    output = output or folder / "用户记忆维度展示.html"
    if data["reviewed"]:
        report_path = export_report([build_report_user(folder, data)], output)
        data["report_href"] = quote(report_path.name)
    output.write_text(render(data, template), encoding="utf-8", newline="\n")
    return output


def export_report(users: list[dict], formal_path: Path) -> Path:
    """与正式版同目录输出独立汇报HTML；只读数据、不调用模型。"""
    output = formal_path.with_name(formal_path.stem + "_report.html")
    data = {"users": users, "formal_href": quote(formal_path.name),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    template = (BASE / "python_app/assets/memory_report.html").read_text(encoding="utf-8")
    output.write_text(render(data, template), encoding="utf-8", newline="\n")
    return output


def export_index(folder: Path, rows: list[dict], rules: Path | None = None,
                 output: Path | None = None, *, internal_view: bool = True) -> Path:
    """把全部用户结果嵌入一个HTML；用户切换不请求其他文件或服务。"""
    import re
    rules = rules or BASE / "J20核心用户记忆维度与提取存储规划.md"
    manifest = json.loads((folder / "batch_manifest.json").read_text(encoding="utf-8"))
    devices = manifest["device_ids"]
    if not devices or any(not isinstance(d, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", d) for d in devices):
        raise ValueError("批次用户ID非法或为空")
    statuses = {r["device_id"]: r for r in rows}
    dimensions = [dict(id=k, name=v["name"], rules=v["rules"], coverage={})
                  for k, v in Registry(rules).dimensions.items()]
    users, report_users = [], []
    for device in devices:
        child = folder / device
        row = statuses.get(device, {})
        if device in statuses and (child / "report.json").exists():
            data = build_data(child, rules, internal_view=internal_view)
            if data["device_id"] != device or not data["reviewed"]:
                raise ValueError("汇总页面仅接受同设备的新版Review结果：" + device)
                report_users.append(build_report_user(child, data))
        else:
            data = dict(device_id=device, reviewed=True, dimensions=dimensions,
                        memories=[], sessions=[], reviews=[], selected_sessions=0,
                        run_status=row.get("status", "pending"),
                        run_error=row.get("error") or "此用户尚无可导出的运行报告")
            report_users.append(dict(device_id=device, run_status=data["run_status"],
                                     selected_sessions=0, committed_sessions=0, sessions=[], cards=[],
                                     dimensions=dimensions, counts={}, source_scope="unknown"))
        users.append(data)
    template = (BASE / "python_app/assets/reviewed_dashboard.html").read_text(encoding="utf-8")
    output = output or folder / "index.html"
    report_path = export_report(report_users, output)
    output.write_text(render({"users": users, "report_href": quote(report_path.name)}, template),
                      encoding="utf-8", newline="\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "output/memory_agent_demo_v1")
    parser.add_argument("--output", type=Path)
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--internal", dest="internal", action="store_true", help="内部完整展示（默认），不改变审核和入库状态")
    visibility.add_argument("--redacted", dest="internal", action="store_false", help="显式隐藏受限候选正文及原话")
    parser.set_defaults(internal=True)
    args = parser.parse_args()
    if (args.input / "batch_manifest.json").exists():
        report = json.loads((args.input / "batch_report.json").read_text(encoding="utf-8"))
        output = export_index(args.input, report["users"], output=args.output, internal_view=args.internal)
    else:
        output = export_user(args.input, args.output, internal_view=args.internal)
    report_output = output.with_name(output.stem + "_report.html")
    print(json.dumps({"output": str(output),
                      "report_output": str(report_output) if report_output.exists() else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
