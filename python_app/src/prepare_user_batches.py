"""流式筛选种子与既有候选用户，隔离0907基线、0929新增和修订。只整理数据，不调用模型。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from typing import Iterator
import xml.etree.ElementTree as ET

from core_user_audit import StreamBook, NS, classify, devices_for, dump, write_csv, parse_time

BASE = Path(__file__).resolve().parents[2]
KINDS = {"dialogue": "*对话*.xlsx", "photo": "*拍照*.xlsx", "call": "*通话*.xlsx", "location": "*定位*.xlsx"}
PRIMARY = {"dialogue": "dialog_id", "photo": "picture_id", "call": "video_id", "location": "id"}
LOG = logging.getLogger(__name__)


def peak_memory() -> int | None:
    """Windows进程峰值工作集，含原生缓存；不启用逐分配追踪拖慢全量XML扫描。"""
    import ctypes
    from ctypes import wintypes
    if not hasattr(ctypes, "WinDLL"):
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
            ("peak", "working", "peak_pool", "pool", "peak_nonpool", "nonpool", "page", "peak_page")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    result = Counters()
    result.cb = ctypes.sizeof(result)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(result), result.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return result.peak


def encoded(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value: object) -> str:
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


def identity(kind: str, row: dict) -> tuple[str, str, str]:
    """新增user_id不是新对话。共同字段变化则归修订；缺主键仅按完整内容去重。"""
    common = {k: v for k, v in row.items() if v not in (None, "") and not (kind == "dialogue" and k == "user_id")}
    for field, value in list(common.items()):
        if field.endswith("_at") or field in ("create_time", "update_time", "validity_time", "timestamp"):
            dt, _ = parse_time(value, 8)
            if dt is not None:
                common[field] = {"timestamp_ms": round(dt.timestamp() * 1000)}
        elif field in ("lat", "lon", "accuracy", "score", "duration"):
            try:
                number = float(value)
                if math.isfinite(number):
                    common[field] = {"number_12sig": format(number, ".12g")}
            except (ValueError, TypeError):
                pass
    fingerprint = sha(common)
    primary = row.get(PRIMARY[kind])
    if primary not in (None, ""):
        key = str(primary).strip()
        if re.fullmatch(r"\d+\.0+", key):
            key = key.split(".")[0]
        return "id:" + key, fingerprint, "primary_key"
    return "hash:" + fingerprint, fingerprint, "content_hash"


def column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def filtered_rows(book: StreamBook, kind: str, targets: set[str]) -> Iterator[tuple[str, int, dict | None]]:
    """先解码设备列再解码命中行，所有工作表均扫描，非目标原文不进缓存。"""
    def value(cell: ET.Element) -> str | None:
        raw = cell.findtext(NS + "v")
        if cell.get("t") == "s" and raw is not None:
            return book.string(int(raw))
        if cell.get("t") == "inlineStr":
            return "".join(t.text or "" for t in cell.iter(NS + "t"))
        return raw
    for sheet, target in book.sheets():
        headers = None
        parent = None
        with book.zip.open(target) as stream:
            for event, element in ET.iterparse(stream, events=("start", "end")):
                if event == "start" and element.tag == NS + "sheetData":
                    parent = element
                if event != "end" or element.tag != NS + "row":
                    continue
                cells = {re.sub(r"\d+$", "", c.attrib["r"]): c for c in element if c.tag == NS + "c"}
                number = int(element.attrib["r"])
                if headers is None:
                    headers = {col: value(c) for col, c in cells.items()}
                    if not headers or any(not h for h in headers.values()) or len(set(headers.values())) != len(headers):
                        raise ValueError(f"非法表头 {book.path.name}/{sheet}")
                    reverse = {name: col for col, name in headers.items()}
                    needed = ("caller_type", "caller_id", "callee_type", "callee_id") if kind == "call" else ("device_id",)
                    if not set(needed + (PRIMARY[kind],)).issubset(reverse):
                        raise ValueError(f"缺少设备或主键字段 {book.path.name}/{sheet}")
                    LOG.info("表头 %s/%s: %s", book.path.name, sheet, list(headers.values()))
                else:
                    if set(cells) - set(headers):
                        raise ValueError(f"表头以外字段 {book.path.name}/{sheet}:{number}")
                    selected = {name: value(cells[reverse[name]]) if reverse[name] in cells else None for name in needed}
                    row = None
                    if targets.intersection(d.upper() for d in devices_for(kind, selected)):
                        row = {name: value(cells[col]) if col in cells else None for col, name in headers.items()}
                    yield sheet, number, row
                if parent is not None:
                    parent.clear()


class BatchIndex:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA cache_size=-16384;
        CREATE TABLE records(seq INTEGER PRIMARY KEY,batch TEXT,kind TEXT,device TEXT,rkey TEXT,fp TEXT,package TEXT);
        CREATE TABLE versions(batch TEXT,kind TEXT,device TEXT,rkey TEXT,fp TEXT,first_seq INTEGER,n INTEGER,
          PRIMARY KEY(batch,kind,device,rkey,fp)) WITHOUT ROWID;
        CREATE TABLE identities(batch TEXT,kind TEXT,device TEXT,rkey TEXT,n INTEGER,
          PRIMARY KEY(batch,kind,device,rkey)) WITHOUT ROWID;
        CREATE TABLE known(kind TEXT,device TEXT,rkey TEXT,fp TEXT,in0907 INTEGER,inlegacy INTEGER,
          PRIMARY KEY(kind,device,rkey,fp)) WITHOUT ROWID;
        CREATE TABLE chats(device TEXT,chat_id TEXT,PRIMARY KEY(device,chat_id)) WITHOUT ROWID;
        """)

    def remember(self, kind: str, device: str, row: dict, origin: str) -> None:
        key, fp, _ = identity(kind, row)
        self.db.execute("INSERT INTO known VALUES(?,?,?,?,?,?) ON CONFLICT(kind,device,rkey,fp) DO UPDATE SET in0907=max(in0907,excluded.in0907),inlegacy=max(inlegacy,excluded.inlegacy)",
                        (kind, device, key, fp, int(origin == "0907"), int(origin == "legacy")))
        if kind == "dialogue" and row.get("chat_id"):
            self.db.execute("INSERT OR IGNORE INTO chats VALUES(?,?)", (device, row["chat_id"]))

    def add(self, batch: str, kind: str, device: str, package: dict) -> None:
        key, fp, method = identity(kind, package["data"])
        package.update(record_key=key, content_fingerprint=fp, identity_method=method, batch=batch)
        cursor = self.db.execute("INSERT INTO records(batch,kind,device,rkey,fp,package) VALUES(?,?,?,?,?,?)", (batch, kind, device, key, fp, encoded(package)))
        first = self.db.execute("INSERT OR IGNORE INTO versions VALUES(?,?,?,?,?,?,1)", (batch, kind, device, key, fp, cursor.lastrowid)).rowcount
        if first:
            self.db.execute("INSERT INTO identities VALUES(?,?,?,?,1) ON CONFLICT(batch,kind,device,rkey) DO UPDATE SET n=n+1", (batch, kind, device, key))
        else:
            self.db.execute("UPDATE versions SET n=n+1 WHERE batch=? AND kind=? AND device=? AND rkey=? AND fp=?", (batch, kind, device, key, fp))
        if batch == "0907":
            self.remember(kind, device, package["data"], "0907")

    def relation(self, kind: str, device: str, key: str, fp: str, variants: int) -> str:
        if variants > 1:
            return "within_batch_conflict"
        if self.db.execute("SELECT 1 FROM known WHERE kind=? AND device=? AND rkey=? AND fp=?", (kind, device, key, fp)).fetchone():
            return "overlap"
        if self.db.execute("SELECT 1 FROM known WHERE kind=? AND device=? AND rkey=?", (kind, device, key)).fetchone():
            return "revised"
        return "new"


def source_file(directory: Path, kind: str, optional: bool = False) -> Path | None:
    paths = sorted(directory.glob(KINDS[kind]))
    if optional and not paths:
        return None
    if len(paths) != 1:
        raise ValueError(f"{directory}/{kind} 应恰好一个文件，实际{len(paths)}")
    return paths[0]


def seed_ids(directory: Path, scratch: Path) -> tuple[set[str], dict]:
    sets, summary = {}, {}
    for kind in KINDS:
        path = source_file(directory, kind)
        book = StreamBook(path, scratch / ("seed_" + kind + ".sqlite3"))
        counts = Counter()
        try:
            for _, _, row in book.rows():
                counts.update(d.upper() for d in devices_for(kind, row))
        finally:
            book.close()
        sets[kind] = set(counts)
        summary[kind] = dict(counts)
    direct = sets["dialogue"] | sets["photo"] | sets["location"]
    if any(not re.fullmatch(r"[0-9A-F]{16}", d) for d in direct):
        raise ValueError("种子设备ID格式异常")
    if sets["call"] - direct:
        LOG.warning("通话对端有额外设备，不自动扩展种子名单：%s", sorted(sets["call"] - direct))
    if not direct:
        raise ValueError("种子名单为空")
    return direct, summary


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scan_file(index: BatchIndex, path: Path, batch: str, kind: str, targets: set[str], scratch: Path, config: dict) -> dict:
    started = time.perf_counter()
    signature = (path.stat().st_size, path.stat().st_mtime_ns)
    LOG.info("开始 %s/%s 大小=%.1fMiB", batch, kind, signature[0] / 1048576)
    book = StreamBook(path, scratch / f"strings_{batch}_{kind}.sqlite3")
    total = matched = copies = 0
    sheets = Counter()
    try:
        for sheet, number, row in filtered_rows(book, kind, targets):
            total += 1
            sheets[sheet] += 1
            if row is not None:
                matched += 1
                devices = sorted({d.upper() for d in devices_for(kind, row)})
                disposition, flags, at = classify(kind, row, config)
                package = {"source_kind": kind, "source_file": path.name, "source_relative_path": f"{batch}/{path.name}",
                    "sheet": sheet, "row": number, "disposition": disposition, "flags": flags,
                    "occurred_at": at.isoformat() if at else None, "device_ids": devices, "data": row}
                for device in targets.intersection(devices):
                    index.add(batch, kind, device, package)
                    copies += 1
            if total % 100000 == 0:
                index.db.commit()
                LOG.info("%s/%s 已扫描=%s 命中源行=%s 用户归属份数=%s 耗时=%.1fs", batch, kind, total, matched, copies, time.perf_counter() - started)
        index.db.commit()
    finally:
        book.close()
    if signature != (path.stat().st_size, path.stat().st_mtime_ns):
        raise ValueError("扫描期间源文件变化")
    result = {"path": str(path), "bytes": signature[0], "sha256": file_sha(path), "rows_scanned": total,
              "matched_source_rows": matched, "matched_device_rows": copies, "sheets": dict(sheets), "seconds": round(time.perf_counter()-started, 3)}
    LOG.info("完成 %s/%s %s", batch, kind, {k: result[k] for k in ("rows_scanned", "matched_source_rows", "seconds")})
    return result


def load_legacy(index: BatchIndex, directory: Path, targets: set[str]) -> dict:
    counts = Counter()
    for device in sorted(targets):
        for path in sorted((directory / "candidate_data" / device).glob("*.jsonl.gz")):
            kind = path.name.split(".")[0]
            if kind not in KINDS:
                continue
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    package = json.loads(line)
                    if device not in {d.upper() for d in devices_for(kind, package["data"])}:
                        raise ValueError("旧分包设备归属不符")
                    index.remember(kind, device, package["data"], "legacy")
                    counts[kind] += 1
        index.db.commit()
    return dict(counts)


def export(index: BatchIndex, output: Path, users: list[dict]) -> tuple[list[dict], dict]:
    counts, totals = [], Counter()
    stages = {"baseline": "baseline_0907", "new": "increment_0929", "revised": "revisions_0929"}
    for folder in stages.values():
        (output / folder / "candidate_data").mkdir(parents=True)
        write_csv(output / folder / "users.csv", users)
    with gzip.open(output / "overlap_0929.jsonl.gz", "wt", encoding="utf-8", newline="\n") as overlaps:
        for user in users:
            device = user["device_id"]
            for kind in KINDS:
                metrics = Counter()
                writers = {}
                for name, folder in stages.items():
                    directory = output / folder / "candidate_data" / device
                    directory.mkdir(exist_ok=True)
                    writers[name] = gzip.open(directory / (kind + ".jsonl.gz"), "wt", encoding="utf-8", newline="\n", compresslevel=3)
                try:
                    sql = """SELECT r.seq,r.batch,r.rkey,r.fp,r.package,v.first_seq,i.n FROM records r
                        JOIN versions v ON (r.batch=v.batch AND r.kind=v.kind AND r.device=v.device AND r.rkey=v.rkey AND r.fp=v.fp)
                        JOIN identities i ON (r.batch=i.batch AND r.kind=i.kind AND r.device=i.device AND r.rkey=i.rkey)
                        WHERE r.device=? AND r.kind=? ORDER BY r.seq"""
                    for seq, batch, key, fp, raw, first, variants in index.db.execute(sql, (device, kind)):
                        p = json.loads(raw)
                        if batch == "0907":
                            metrics["baseline_rows"] += 1
                            p["base_disposition"] = p["disposition"]
                            if variants > 1:
                                p["disposition"] = "key_conflict"
                            elif seq != first:
                                p["disposition"] = "duplicate"
                            metrics["baseline_eligible"] += p["disposition"] == "eligible"
                            writers["baseline"].write(encoded(p) + "\n")
                            continue
                        metrics["snapshot_0929_rows"] += 1
                        relation = index.relation(kind, device, key, fp, variants)
                        if seq != first:
                            relation = "batch_duplicate"
                        if relation in ("overlap", "batch_duplicate"):
                            metrics[relation] += 1
                            overlaps.write(encoded({"device_id": device, "source_kind": kind, "record_key": key,
                                "content_fingerprint": fp, "relation": relation, "source_file": p["source_relative_path"],
                                "sheet": p["sheet"], "row": p["row"], "extra_fields": {"user_id": p["data"].get("user_id")} if kind == "dialogue" else {}}) + "\n")
                            continue
                        p["delta_class"] = relation
                        if relation == "new":
                            metrics["new_rows"] += 1
                            metrics["new_eligible"] += p["disposition"] == "eligible"
                            if kind == "dialogue" and p["data"].get("chat_id"):
                                extends = bool(index.db.execute("SELECT 1 FROM chats WHERE device=? AND chat_id=?", (device, p["data"]["chat_id"])).fetchone())
                                p["session_relation"] = "extends_existing_session" if extends else "new_session"
                                metrics["extends_existing_session_rows"] += extends
                            writers["new"].write(encoded(p) + "\n")
                        else:
                            metrics[relation] += 1
                            p["base_disposition"] = p["disposition"]
                            p["disposition"] = "snapshot_revision_pending"
                            p["previous_fingerprints"] = [x[0] for x in index.db.execute("SELECT fp FROM known WHERE kind=? AND device=? AND rkey=?", (kind, device, key))]
                            writers["revised"].write(encoded(p) + "\n")
                finally:
                    for writer in writers.values():
                        writer.close()
                assert metrics["snapshot_0929_rows"] == sum(metrics[k] for k in ("new_rows", "overlap", "batch_duplicate", "revised", "within_batch_conflict"))
                fields = ("baseline_rows", "baseline_eligible", "snapshot_0929_rows", "new_rows", "new_eligible", "overlap", "batch_duplicate", "revised", "within_batch_conflict", "extends_existing_session_rows")
                counts.append(dict(user, source_kind=kind, **{k: metrics[k] for k in fields}, source_0929="not_provided" if kind == "photo" else "provided"))
                totals.update(metrics)
    return counts, dict(totals)


def validate(output: Path) -> dict:
    """重新读取落盘包，校验身份、行数、状态与新增主键不出现在0907/旧库中。"""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    users = {u["device_id"] for u in manifest["users"]}
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix="memory_batch_validate_") as tmp:
        db = sqlite3.connect(Path(tmp) / "keys.sqlite3")
        db.execute("CREATE TABLE baseline(kind TEXT,device TEXT,k TEXT,PRIMARY KEY(kind,device,k)) WITHOUT ROWID")
        db.execute("CREATE TABLE added(kind TEXT,device TEXT,k TEXT,PRIMARY KEY(kind,device,k)) WITHOUT ROWID")
        legacy = Path(manifest["legacy_root"]) / "candidate_data"
        for device in users:
            for path in (legacy / device).glob("*.jsonl.gz"):
                kind = path.name.split(".")[0]
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        key, _, _ = identity(kind, json.loads(line)["data"])
                        db.execute("INSERT OR IGNORE INTO baseline VALUES(?,?,?)", (kind, device, key))
        for stage in ("baseline_0907", "increment_0929", "revisions_0929"):
            paths = list((output / stage / "candidate_data").glob("*/*.jsonl.gz"))
            if len(paths) != len(users) * len(KINDS):
                raise ValueError("用户数据文件数量不完整")
            for path in sorted((output / stage / "candidate_data").glob("*/*.jsonl.gz")):
                device, kind = path.parent.name, path.name.split(".")[0]
                if device not in users:
                    raise ValueError("导出包包含未授权设备")
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        p = json.loads(line)
                        key, fp, _ = identity(kind, p["data"])
                        if device not in p["device_ids"] or key != p["record_key"] or fp != p["content_fingerprint"]:
                            raise ValueError("设备归属或内容指纹不一致")
                        if stage == "baseline_0907":
                            db.execute("INSERT OR IGNORE INTO baseline VALUES(?,?,?)", (kind, device, key))
                        if stage == "increment_0929" and db.execute("SELECT 1 FROM baseline WHERE kind=? AND device=? AND k=?", (kind, device, key)).fetchone():
                            raise ValueError("新增包混入已存在的记录ID")
                        if stage == "increment_0929":
                            db.execute("INSERT INTO added VALUES(?,?,?)", (kind, device, key))
                        if stage == "revisions_0929" and p["disposition"] != "snapshot_revision_pending":
                            raise ValueError("修订未隔离")
                        counts[stage] += 1
        db.close()
    with gzip.open(output / "overlap_0929.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            p = json.loads(line)
            if p["device_id"] not in users or p["relation"] not in ("overlap", "batch_duplicate"):
                raise ValueError("重复记录索引无效")
            counts["overlap_0929"] += 1
    expected = manifest["totals"]
    assert counts["baseline_0907"] == expected.get("baseline_rows", 0)
    assert counts["increment_0929"] == expected.get("new_rows", 0)
    assert counts["revisions_0929"] == expected.get("revised", 0) + expected.get("within_batch_conflict", 0)
    assert counts["overlap_0929"] == expected.get("overlap", 0) + expected.get("batch_duplicate", 0)
    return {"status": "passed", "users": len(users), "counts": dict(counts), "increment_keys_disjoint_from_baseline_and_legacy": True}


def run(data: Path, legacy: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("输出目录已存在，请用新目录，不覆盖旧分包")
    config = json.loads((BASE / "python_app/config/core_user_audit.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="memory_batch_prepare_") as temporary:
        scratch = Path(temporary)
        seeds, seed_summary = seed_ids(data / "种子", scratch)
        old_ids = {p.name for p in (legacy / "candidate_data").iterdir() if p.is_dir()}
        with (legacy / "core_50_provisional.csv").open(encoding="utf-8-sig", newline="") as stream:
            core = {r["device_id"] for r in csv.DictReader(stream)}
        if not core.issubset(old_ids) or any(not re.fullmatch(r"[0-9A-F]{16}", d) for d in old_ids):
            raise ValueError("旧用户名单或目录格式异常")
        targets = seeds | old_ids
        users = [dict(device_id=d, is_seed=int(d in seeds), is_existing_candidate=int(d in old_ids), is_core_50=int(d in core)) for d in sorted(targets)]
        LOG.info("种子=%s 既有候选=%s 核心=%s 合并去重用户=%s", len(seeds), len(old_ids), len(core), len(targets))
        index = BatchIndex(scratch / "index.sqlite3")
        try:
            legacy_counts = load_legacy(index, legacy, old_ids)
            files = {}
            for batch in ("0907", "0929"):
                batch_config = dict(config, max_date="2026-09-07" if batch == "0907" else "2026-09-29")
                for kind in KINDS:
                    path = source_file(data / batch, kind, optional=batch == "0929" and kind == "photo")
                    files[batch + "/" + kind] = scan_file(index, path, batch, kind, targets, scratch, batch_config) if path else {"status": "not_provided"}
            index.db.execute("CREATE INDEX export_lookup ON records(device,kind,seq)")
            index.db.commit()
            counts, totals = export(index, output, users)
            peak = peak_memory()
            manifest = {"schema_version": "user-data-batches-v2", "users": users, "seed_reference_counts": seed_summary,
                "legacy_root": str(legacy.resolve()), "legacy_rows_used_for_dedup": legacy_counts,
                "files": files, "totals": totals, "seconds": round(time.perf_counter()-started, 3), "process_peak_working_set_bytes": peak,
                "policy": {"scope": "8_seed_plus_all_saved_candidates", "dialogue_extra_column": "user_id保存但不作为新对话依据",
                           "comparison_precision": "时间毫秒；数值字段12位有效数字；原data不变",
                           "no_model_calls": True, "old_outputs_unchanged": True}}
            dump(output / "manifest.json", manifest)
            write_csv(output / "users.csv", users)
            write_csv(output / "user_source_counts.csv", counts)
        finally:
            index.db.close()
    verification = validate(output)
    dump(output / "validation.json", verification)
    report(output, manifest, counts, verification)
    LOG.info("完成：%s", encoded({"output": str(output), "totals": totals, "validation": verification}))


def normalize_existing(output: Path) -> None:
    """修正本脚本v1输出的导出精度误报。只替换本次生成文件，不重新读取大表或改原文。"""
    started = time.perf_counter()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != "user-data-batches-v1":
        raise ValueError("只允许对本脚本v1输出执行一次精度修正")
    if manifest["totals"].get("within_batch_conflict", 0) or manifest["totals"].get("batch_duplicate", 0):
        raise ValueError("存在批内冲突/重复，需重新执行完整扫描")
    with (output / "user_source_counts.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for k in row:
            if k not in ("device_id", "source_kind", "source_0929"):
                row[k] = int(row[k])
        row["revised"] = 0
        row["overlap"] = 0
    metrics = {(r["device_id"], r["source_kind"]): r for r in rows}
    with tempfile.TemporaryDirectory(prefix="memory_precision_rebuild_", dir=output.parent) as tmp:
        staging = Path(tmp)
        index = BatchIndex(staging / "known.sqlite3")
        index.db.execute("CREATE TABLE fp_map(kind TEXT,device TEXT,old_fp TEXT,new_fp TEXT,PRIMARY KEY(kind,device,old_fp)) WITHOUT ROWID")
        def remember(kind: str, device: str, p: dict, origin: str) -> None:
            index.remember(kind, device, p["data"], origin)
            raw = {k: v for k, v in p["data"].items() if v not in (None, "") and not (kind == "dialogue" and k == "user_id")}
            index.db.execute("INSERT OR REPLACE INTO fp_map VALUES(?,?,?,?)", (kind, device, sha(raw), identity(kind, p["data"])[1]))
        try:
            for user in manifest["users"]:
                device = user["device_id"]
                for path in (Path(manifest["legacy_root"]) / "candidate_data" / device).glob("*.jsonl.gz"):
                    with gzip.open(path, "rt", encoding="utf-8") as stream:
                        for p in map(json.loads, stream):
                            remember(path.name.split(".")[0], device, p, "legacy")
                index.db.commit()
            for stage in ("baseline_0907", "increment_0929", "revisions_0929"):
                (staging / stage).mkdir()
                write_csv(staging / stage / "users.csv", manifest["users"])
            ledger = staging / "overlap_0929.jsonl.gz"
            with gzip.open(ledger, "wt", encoding="utf-8", newline="\n") as overlap:
                for stage in ("baseline_0907", "increment_0929", "revisions_0929"):
                    for path in sorted((output / stage / "candidate_data").glob("*/*.jsonl.gz")):
                        kind, device = path.name.split(".")[0], path.parent.name
                        target = staging / path.relative_to(output)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with gzip.open(path, "rt", encoding="utf-8") as src, gzip.open(target, "wt", encoding="utf-8", newline="\n", compresslevel=3) as dst:
                            for p in map(json.loads, src):
                                if p["disposition"] == "key_conflict":
                                    raise ValueError("基线有主键冲突，需重新完整扫描")
                                key, fp, _ = identity(kind, p["data"])
                                if stage == "baseline_0907":
                                    remember(kind, device, p, "0907")
                                p["content_fingerprint"] = fp
                                if stage == "revisions_0929":
                                    relation = index.relation(kind, device, key, fp, 1)
                                    if relation == "overlap":
                                        metrics[(device, kind)]["overlap"] += 1
                                        overlap.write(encoded(dict(device_id=device, source_kind=kind, record_key=key, content_fingerprint=fp,
                                            relation="overlap", source_file=p["source_relative_path"], sheet=p["sheet"], row=p["row"],
                                            extra_fields={"user_id": p["data"].get("user_id")} if kind == "dialogue" else {})) + "\n")
                                        continue
                                    if relation != "revised":
                                        raise ValueError("修订主键范围改变，不能仅做精度重分类")
                                    metrics[(device, kind)]["revised"] += 1
                                    p["previous_fingerprints"] = [r[0] for r in index.db.execute("SELECT fp FROM known WHERE kind=? AND device=? AND rkey=?", (kind, device, key))]
                                dst.write(encoded(p) + "\n")
                        index.db.commit()
                with gzip.open(output / "overlap_0929.jsonl.gz", "rt", encoding="utf-8") as src:
                    for p in map(json.loads, src):
                        result = index.db.execute("SELECT new_fp FROM fp_map WHERE kind=? AND device=? AND old_fp=?", (p["source_kind"], p["device_id"], p["content_fingerprint"])).fetchone()
                        if result is None:
                            raise ValueError("原重叠记录无法映射新指纹")
                        p["content_fingerprint"] = result[0]
                        metrics[(p["device_id"], p["source_kind"])]["overlap"] += 1
                        overlap.write(encoded(p) + "\n")
            manifest["totals"]["revised"] = sum(r["revised"] for r in rows)
            manifest["totals"]["overlap"] = sum(r["overlap"] for r in rows)
            manifest["schema_version"] = "user-data-batches-v2"
            manifest["precision_reclassification_seconds"] = round(time.perf_counter() - started, 3)
            manifest["policy"]["comparison_precision"] = "时间毫秒；数值字段12位有效数字；原data不变"
            for row in rows:
                assert row["snapshot_0929_rows"] == sum(row[k] for k in ("new_rows", "overlap", "revised", "batch_duplicate", "within_batch_conflict"))
            dump(staging / "manifest.json", manifest)
            write_csv(staging / "users.csv", manifest["users"])
            write_csv(staging / "user_source_counts.csv", rows)
            verification = validate(staging)
            dump(staging / "validation.json", verification)
            report(staging, manifest, rows, verification)
            # 先完整校验临时结果，再替换本次生成的已知文件；无目录删除，无旧库写入。
            for source in staging.rglob("*"):
                if source.is_file() and source.suffix != ".sqlite3":
                    source.replace(output / source.relative_to(staging))
        finally:
            index.db.close()
    LOG.info("精度修正完成 %s", encoded(manifest["totals"]))


def report(output: Path, manifest: dict, counts: list[dict], verification: dict) -> None:
    t = Counter(manifest["totals"])
    lines = ["# 0907基线与0929追加测试数据", "", "只整理原始数据，不调用模型，不修改已有记忆库或Agent配置。", "",
        f"范围：种子{sum(u['is_seed'] for u in manifest['users'])}个，既有候选{sum(u['is_existing_candidate'] for u in manifest['users'])}个（保留核心50标记），合并去重后{len(manifest['users'])}个设备。", "",
        "| 分区 | 记录份数 | 说明 |", "| --- | ---: | --- |",
        f"| baseline_0907 | {t['baseline_rows']} | 0907中目标用户全部原始行，含无效/重复/冲突标记 |",
        f"| increment_0929 | {t['new_rows']} | 0929中新记录ID，排除0907和既有core_users已保存记录 |",
        f"| revisions_0929 | {t['revised'] + t['within_batch_conflict']} | 同ID内容变化或0929批内冲突，不混入新增 |",
        f"| overlap_0929.jsonl.gz | {t['overlap'] + t['batch_duplicate']} | 已知重叠或批内重复的来源索引，不重复导入 |", "",
        "## 使用顺序", "", "1. 先使用 baseline_0907 做基线。已有core_users原包不变，可对照其原有结果。",
        "2. 再使用 increment_0929 测试追加；保留原data字段和证据ID，不把重叠记录重复当新支持。",
        "3. revisions_0929 是原日志修订，不等同于用户在新对话里更正事实；需单独处理，不能直接追加有效记忆。",
        "4. 现有Agent的 --resume 会校验配置与输入不变，不能用改 data_root 后 --resume 冒充追加。本次未改该门禁，也未启动记忆提取。",
        "", "0929未提供照片文件：照片来源状态为 not_provided，不等于没有新照片。", "",
        "某条0907记录未出现在0929快照中，不作为删除依据；基线仍保留，不执行记忆删除。", "",
        "通话可能同时属于两个目标设备，按用户分包时各保存一份，因此份数不等于去重通话数。非正常状态、空文本等保留原行并标记，不静默丢弃。", "",
        "## 文件和字段", "", "各分区 candidate_data/<device_id>/ 下按 dialogue/photo/call/location 保存 UTF-8 JSONL.gz。",
        "users.csv 的 is_seed / is_existing_candidate / is_core_50 分别表示种子、原100候选、原核心50成员；0/1为否/是，不重新排名。",
        "record_key为来源类型内主键；content_fingerprint为共同字段指纹；identity_method说明按主键或缺主键时按内容识别。",
        "比较时以设备+来源类型+主键隔离：时间统一到毫秒，坐标/精度/分数/时长统一12位有效数字，忽略Excel导出的浮点尾差。原data值保持不变；超过该精度的实质差异仍进入修订区。",
        "data为原字段；source_file/source_relative_path/sheet/row为可回溯位置；occurred_at按既有口径转UTC+8；disposition是是否适于后续提取的状态，不删除原文。",
        "base_disposition保留隔离之前的状态；delta_class为new/revised/within_batch_conflict；previous_fingerprints是旧内容指纹，不声称旧版已被覆盖。",
        "dialogue中新加的user_id列保留在新记录和重叠索引extra_fields中；不会仅因为多了一列就当成新增对话。",
        "session_relation为new_session或extends_existing_session，后者提示同一chat_id在基线已存在，追加处理需保留前文，不应覆盖已提交Session。",
        "user_source_counts.csv逐用户逐来源核对基线、0929总量、新增、重叠、修订和可用量；manifest.json记录输入哈希和完整口径；validation.json记录落盘复核。", "",
        "统计列：baseline_rows为0907原始份数，baseline_eligible为其中可用量；snapshot_0929_rows为0929命中总份数；new_rows/new_eligible为新增总量/可用量。",
        "overlap为旧数据重叠量；batch_duplicate为同批重复；revised为同ID实质变化；within_batch_conflict为同批同ID多内容；extends_existing_session_rows为新增但延续旧chat_id的对话行数。",
        "source_0929说明来源是否提供；flags为辅助风险标记，不等于删除；device_ids记录原日志关联设备；batch为来源批次。所有计数均为记录行/用户归属份数，不是Session数量。", "",
        "## 种子用户明细", "", "| 设备 | 0907对话 | 0907照片 | 0907通话 | 0907定位 | 0929新增对话 | 新增通话 | 新增定位 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for u in manifest["users"]:
        if not u["is_seed"]:
            continue
        rows = {r["source_kind"]: r for r in counts if r["device_id"] == u["device_id"]}
        numbers = [rows[k]["baseline_rows"] for k in KINDS] + [rows[k]["new_rows"] for k in ("dialogue", "call", "location")]
        lines.append("| " + u["device_id"] + " | " + " | ".join(map(str, numbers)) + " |")
    missing = [u["device_id"] for u in manifest["users"] if not any(r["baseline_rows"] for r in counts if r["device_id"] == u["device_id"])]
    lines += ["", "## 校验与性能", "", f"落盘复核：{verification['status']}。新增主键与0907及旧core_users互斥，分区数量核对通过。",
        f"扫描及导出耗时：{manifest['seconds']}秒；进程峰值工作集：{manifest['process_peak_working_set_bytes']}字节（含原生缓存）。无模型推理、参数量及FLOPs不适用。",
        f"额外精度重分类耗时：{manifest.get('precision_reclassification_seconds', 0)}秒。统计时间不包含最后的独立落盘复核。",
        "0907完全未出现的目标设备：" + ("、".join(missing) if missing else "无"), ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE / "date")
    parser.add_argument("--legacy", type=Path, default=BASE / "core_users/core_users_stage1_20260917")
    parser.add_argument("--output", type=Path, default=BASE / "core_users/user_batches_0907_0929")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--normalize-existing", action="store_true")
    args = parser.parse_args()
    if args.normalize_existing:
        normalize_existing(args.output)
    elif args.validate_only:
        print(encoded(validate(args.output)))
    else:
        run(args.data, args.legacy, args.output)
