"""从 data/1017 原始 xlsx 中提取指定用户的记录，产出 memory_agent 可直接读取的 candidate_data。

只读取原始 xlsx，只写出目标用户的 jsonl.gz；不修改原始数据，不调用模型。

产出的每个记录带 disposition/flags，与 core_users/**/candidate_data 格式一致：
  {source_kind, source_file, source_relative_path, sheet, row, disposition, flags,
   occurred_at, device_ids, data, record_key, content_fingerprint, identity_method, batch}

用法：
  python extract_1017_users.py                    # 默认目标用户与输出目录
  python extract_1017_users.py --kinds dialogue photo
  python extract_1017_users.py --force            # 忽略缓存重新扫描

设计说明：
  - 复用仓库自身的 StreamBook（流式 XML，不整表载入）、classify、devices_for、identity，
    保证 disposition 口径与 core_users 既有数据完全一致。
  - 大文件按 kind 分别扫描；每个 kind 的结果落盘为缓存，重复运行可秒级跳过。
  - max_date 默认取导出日期，避免 1017 数据被 core_user_audit.json 的 2026-09-07 截断。
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "python_app" / "src"))

from core_user_audit import StreamBook, classify, devices_for  # noqa: E402
from prepare_user_batches import encoded, identity  # noqa: E402

LOG = logging.getLogger("extract_1017")

FILES = {
    "dialogue": "智能体对话全部数据20261014.xlsx",
    "photo": "拍照图片全部数据20261015.xlsx",
    "call": "通话记录全部数据20261014.xlsx",
    "location": "定位全部数据20261014.xlsx",
}

CORE = [
    "12354838B667950A", "12354838B66792A5", "12354838B6679432",
    "12354838B667A7F0", "12354838B667A861",
]
SEED = [
    "12354838B6678E41", "12354838B6678E87", "12354838B6678EAE", "12354838B6678EAF",
    "12354838B6678EB7", "12354838B6678EB9", "12354838B6678EBC", "12354838B6678EC0",
]


def audit_config(max_date: str) -> dict:
    """与 python_app/config/core_user_audit.json 同口径，仅放开 max_date。"""
    return {
        "timezone_offset": 8,
        "min_date": "2020-01-01",
        "max_date": max_date,
        "device_pattern": "[0-9A-Fa-f]{16}",
        "dialogue_statuses": ["0"],
        "system_texts": ["[心跳]", "[系统欢迎语]", "heartbeat"],
        "test_text_pattern": "测试测试|测试一下|功能测试|压力测试|接口测试|自动化测试|调试|test[ _-]?(?:case|ing)|hello world",
        "asset_url_pattern": "(?:icon_toycloud|album_enter_|playing_bg|recording_bg|/assets/)",
        "accuracy_warning": 500,
    }


def scan_kind(kind: str, path: Path, targets: set[str], config: dict, scratch: Path,
              cache: Path, force: bool) -> list[dict]:
    """扫描单个 xlsx，返回目标用户的记录包（含 disposition）。结果落盘缓存。"""
    signature = f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}:{config['max_date']}"
    if cache.is_file() and not force:
        head = cache.open(encoding="utf-8").readline().strip()
        if head == signature:
            rows = [json.loads(line) for line in cache.open(encoding="utf-8").readlines()[1:]]
            LOG.info("[%s] 命中缓存 %s 条", kind, f"{len(rows):,}")
            return rows
        LOG.info("[%s] 缓存签名变化，重新扫描", kind)

    LOG.info("[%s] 开始扫描 %s (%.1f MiB)", kind, path.name, path.stat().st_size / 1048576)
    started = time.perf_counter()
    book = StreamBook(path, scratch / f"strings_{kind}.sqlite3")
    out: list[dict] = []
    scanned = 0
    try:
        for sheet, number, row in book.rows():
            scanned += 1
            devs = {d.upper() for d in devices_for(kind, row) if d}
            hit = targets & devs
            if not hit:
                continue
            disposition, flags, at = classify(kind, row, config)
            key, fp, method = identity(kind, row)
            for device in sorted(hit):
                out.append({
                    "source_kind": kind,
                    "source_file": path.name,
                    "source_relative_path": f"1017/{path.name}",
                    "sheet": sheet,
                    "row": number,
                    "disposition": disposition,
                    "flags": flags,
                    "occurred_at": at.isoformat() if at else None,
                    "device_ids": sorted(devs),
                    "data": row,
                    "record_key": key,
                    "content_fingerprint": fp,
                    "identity_method": method,
                    "batch": "1017",
                })
    finally:
        book.close()
    elapsed = time.perf_counter() - started
    dist = Counter(r["disposition"] for r in out)
    LOG.info("[%s] 完成 扫描=%s 命中=%s 耗时=%.1fs %s",
             kind, f"{scanned:,}", f"{len(out):,}", elapsed, dict(dist))

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8") as stream:
        stream.write(signature + "\n")
        for record in out:
            stream.write(encoded(record) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=BASE / "data/1017")
    parser.add_argument("--output", type=Path, default=BASE / "data/1017")
    parser.add_argument("--scratch", type=Path, default=None, help="共享字符串缓存目录")
    parser.add_argument("--max-date", default="2026-10-15", help="审计时间上限，默认导出日期")
    parser.add_argument("--kinds", nargs="+", default=list(FILES), choices=list(FILES))
    parser.add_argument("--force", action="store_true", help="忽略缓存重新扫描")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    targets = {d.upper() for d in CORE + SEED}
    config = audit_config(args.max_date)
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="extract1017_"))
    scratch.mkdir(parents=True, exist_ok=True)
    cache_dir = scratch / "cache"
    LOG.info("目标用户 %s 个；scratch=%s", len(targets), scratch)

    by_device: dict[str, dict[str, list[dict]]] = {d: {} for d in sorted(targets)}
    stats: dict[str, dict] = {}
    for kind in args.kinds:
        path = args.data / FILES[kind]
        if not path.is_file():
            LOG.error("[%s] 缺少源文件 %s，跳过", kind, path)
            continue
        records = scan_kind(kind, path, targets, config, scratch, cache_dir / f"{kind}.jsonl", args.force)
        for record in records:
            for device in targets & {d.upper() for d in record["device_ids"]}:
                by_device[device].setdefault(kind, []).append(record)
        stats[kind] = {
            "source_file": path.name,
            "records": len(records),
            "eligible": sum(1 for r in records if r["disposition"] == "eligible"),
            "dispositions": dict(Counter(r["disposition"] for r in records)),
        }

    written = Counter()
    for device, kinds in sorted(by_device.items()):
        directory = args.output / "candidate_data" / device
        directory.mkdir(parents=True, exist_ok=True)
        for kind, records in kinds.items():
            records.sort(key=lambda r: ((r["occurred_at"] or ""), r["record_key"]))
            with gzip.open(directory / f"{kind}.jsonl.gz", "wt", encoding="utf-8", newline="\n") as stream:
                for record in records:
                    stream.write(encoded(record) + "\n")
            written[(device, kind)] = len(records)
    LOG.info("写出 %s 个用户目录", len({d for d, _ in written}))

    summary = {
        "source": "data/1017",
        "batch": "1017",
        "max_date": args.max_date,
        "targets": {"core": CORE, "seed": SEED},
        "kinds": stats,
        "per_user": {
            device: {kind: count for (d, kind), count in sorted(written.items()) if d == device}
            for device in sorted(by_device)
        },
    }
    (args.output / "extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    LOG.info("完成：%s", args.output / "extraction_summary.json")


if __name__ == "__main__":
    main()
