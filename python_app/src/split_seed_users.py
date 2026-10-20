"""将已验证的种子用户分包移到独立目录，原分包仅保留既有候选。"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import gzip
import json
from pathlib import Path
import tempfile
import time

from prepare_user_batches import BASE, KINDS, dump, encoded, file_sha, peak_memory, validate, write_csv

STAGES = ("baseline_0907", "increment_0929", "revisions_0929")


def read_counts(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in row:
            if key not in ("device_id", "source_kind", "source_0929"):
                row[key] = int(row[key])
    return rows


def metadata(root: Path, original: dict, rows: list[dict], users: list[dict], cohort: str) -> dict:
    m = copy.deepcopy(original)
    m["users"] = users
    m["cohort"] = cohort
    m["source_scan_scope"] = "files内扫描统计沿用原108设备联合扫描；本目录数量以totals和user_source_counts.csv为准"
    fields = ("baseline_rows", "baseline_eligible", "snapshot_0929_rows", "new_rows", "new_eligible", "overlap", "batch_duplicate", "revised", "within_batch_conflict", "extends_existing_session_rows")
    m["totals"] = {key: sum(r[key] for r in rows) for key in fields}
    selected = {u["device_id"] for u in users}
    m["seed_reference_counts"] = {kind: {d: n for d, n in values.items() if d in selected} for kind, values in m.get("seed_reference_counts", {}).items()}
    m["policy"]["scope"] = cohort
    dump(root / "manifest.json", m)
    write_csv(root / "users.csv", users)
    write_csv(root / "user_source_counts.csv", rows)
    for stage in STAGES:
        write_csv(root / stage / "users.csv", users)
    return m


def report(root: Path, manifest: dict, verification: dict, seconds: float) -> None:
    seed = manifest["cohort"] == "seed_only"
    title = "8个种子用户独立数据" if seed else "原100个候选用户数据（不含种子）"
    t = Counter(manifest["totals"])
    lines = [f"# {title}", "", f"本目录包含 {len(manifest['users'])} 个设备；核心50成员标记数：{sum(u['is_core_50'] for u in manifest['users'])}。种子和原候选已物理分开，不共用数据包。", "",
        "| 分区 | 记录份数 | 含义 |", "| --- | ---: | --- |",
        f"| baseline_0907 | {t['baseline_rows']} | 0907完整基线，保留异常状态标记 |",
        f"| increment_0929 | {t['new_rows']} | 排除基线及旧core_users后的新增记录 |",
        f"| revisions_0929 | {t['revised'] + t['within_batch_conflict']} | 同ID实质变化，单独处理 |",
        f"| overlap_0929.jsonl.gz | {t['overlap'] + t['batch_duplicate']} | 重叠来源索引，不重复导入 |", "",
        "## 使用与字段", "", "每个分区的 candidate_data/<device_id>/ 内按 dialogue/photo/call/location 保存原始JSONL.gz。先处理基线，再处理新增；修订不直接当新增。现有Agent的 --resume 不能通过改data_root来实现跨批追加。本次未改配置、未调用模型、未写记忆库。", "",
        "users.csv：device_id是设备标识；is_seed、is_existing_candidate、is_core_50分别为种子、原候选、核心50标记（1是/0否）。",
        "user_source_counts.csv：source_kind为来源；baseline_rows/baseline_eligible为基线总量/可用量；snapshot_0929_rows为新快照总量；new_rows/new_eligible为新增总量/可用量；overlap为重叠；batch_duplicate为批内重复；revised为修订；within_batch_conflict为批内主键冲突；extends_existing_session_rows为延续旧chat_id的新增行数；source_0929表示来源是否提供。",
        "记录字段：data保持原始值；source_file/source_relative_path/sheet/row定位原Excel行；device_ids为归属设备；record_key为主键；content_fingerprint为规范化比较指纹；identity_method为主键或内容识别方式；batch为来源批次；occurred_at为UTC+8观察时间；disposition/base_disposition为当前/隔离前状态；flags为风险标记；delta_class为新增或修订分类；previous_fingerprints为旧内容指纹；session_relation为新会话或旧会话延续。",
        "时间按毫秒、数值按12位有效数字比较，忽略导出浮点尾差但不改原文。0929没有照片文件，不能解释为照片零新增；基线记录在新快照缺失不代表删除。计数是记录份数，不是Session数。", "",
        "manifest.json中users/totals仅对应本组；files和legacy_rows_used_for_dedup保留联合扫描/旧库比对溯源范围，不是本组记录数量。validation.json为本组落盘校验。", "",
        "## 校验", "", f"状态：{verification['status']}；目录用户隔离、文件哈希及各分区计数已核对。拆分耗时约{seconds:.2f}秒；进程峰值工作集{peak_memory()}字节。无模型参数、FLOPs或推理费用。原core_users_stage1_20260917及源Excel不变。", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def split(source: Path, target: Path) -> dict:
    start = time.perf_counter()
    source, target = source.resolve(), target.resolve()
    if target.exists() or source.parent != target.parent or source == target:
        raise ValueError("新目录须为同一父目录下尚不存在的独立目录")
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    rows = read_counts(source / "user_source_counts.csv")
    seeds = [u for u in original["users"] if u["is_seed"]]
    existing = [u for u in original["users"] if not u["is_seed"]]
    if not seeds or not existing or any(u["is_existing_candidate"] for u in seeds) or not all(u["is_existing_candidate"] for u in existing):
        raise ValueError("分组为空、种子与候选重叠或候选标记异常，停止移动")
    ids = {u["device_id"] for u in seeds}
    moves, fingerprints = [], {}
    for stage in STAGES:
        for device in ids:
            old = (source / stage / "candidate_data" / device).resolve()
            new = (target / stage / "candidate_data" / device).resolve()
            if not old.is_relative_to(source) or not new.is_relative_to(target) or not old.is_dir():
                raise ValueError("移动路径不在已确认数据目录内")
            files = list(old.iterdir())
            if {p.name for p in files} != {k + ".jsonl.gz" for k in KINDS}:
                raise ValueError("设备目录含未知文件或缺失文件")
            for p in files:
                fingerprints[p.relative_to(source)] = file_sha(p)
            moves.append((old, new))
    # 保存元数据用于异常回滚；不会删除或重写原始记录文件。
    meta_paths = [source / name for name in ("manifest.json", "users.csv", "user_source_counts.csv", "README.md", "validation.json")]
    meta_paths += [source / stage / "users.csv" for stage in STAGES]
    backups = {p: p.read_bytes() for p in meta_paths if p.exists()}
    moved = []
    target.mkdir()
    with tempfile.TemporaryDirectory(prefix="cohort_split_", dir=source.parent) as temporary:
        scratch = Path(temporary)
        old_ledger = scratch / "original_overlap.jsonl.gz"
        try:
            with gzip.open(source / "overlap_0929.jsonl.gz", "rt", encoding="utf-8") as reader, gzip.open(target / "overlap_0929.jsonl.gz", "wt", encoding="utf-8", newline="\n") as seed_writer, gzip.open(scratch / "remaining.jsonl.gz", "wt", encoding="utf-8", newline="\n") as remaining_writer:
                all_ids = {u["device_id"] for u in original["users"]}
                for line in reader:
                    p = json.loads(line)
                    if p["device_id"] not in all_ids:
                        raise ValueError("重叠索引包含未知设备")
                    (seed_writer if p["device_id"] in ids else remaining_writer).write(line)
            for old, new in moves:
                new.parent.mkdir(parents=True, exist_ok=True)
                old.rename(new)
                moved.append((old, new))
            for relative, expected in fingerprints.items():
                if file_sha(target / relative) != expected:
                    raise ValueError("移动前后文件哈希不一致")
            (source / "overlap_0929.jsonl.gz").rename(old_ledger)
            (scratch / "remaining.jsonl.gz").rename(source / "overlap_0929.jsonl.gz")
            seed_meta = metadata(target, original, [r for r in rows if r["device_id"] in ids], seeds, "seed_only")
            old_meta = metadata(source, original, [r for r in rows if r["device_id"] not in ids], existing, "existing_candidates_only")
            result = {}
            for root, m in ((target, seed_meta), (source, old_meta)):
                v = validate(root)
                v["seed_candidate_separation"] = True
                v["moved_file_hashes_unchanged"] = True
                dump(root / "validation.json", v)
                report(root, m, v, time.perf_counter() - start)
                result[m["cohort"]] = dict(path=str(root), users=len(m["users"]), totals=m["totals"])
            return result
        except Exception:
            for old, new in reversed(moved):
                new.rename(old)
            for path, content in backups.items():
                path.write_bytes(content)
            if old_ledger.exists():
                old_ledger.replace(source / "overlap_0929.jsonl.gz")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "core_users/user_batches_0907_0929")
    parser.add_argument("--target", type=Path, default=BASE / "core_users/seed_users_8_0907_0929")
    args = parser.parse_args()
    print(encoded(split(args.source, args.target)), flush=True)
