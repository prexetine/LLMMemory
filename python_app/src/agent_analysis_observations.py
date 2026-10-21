"""将行为日志聚合为可追溯观察，不把统计数值直接伪装为儿童偏好。"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import gzip
import json
import re
from pathlib import Path
from typing import Any

from agent_data import digest


def _coarse_region(address: object) -> str:
    """仅保留城市/区县；绝不返回门牌、道路、坐标或完整地址。"""
    if not isinstance(address, str):
        return ""
    city = re.search(r"([^省市]{2,16}市)", address)
    # 在已识别城市后的文本中找区县，避免把省、市一起误当成区县。
    tail = address[city.end():] if city else address
    district = re.search(r"([^区县]{2,16}(?:区|县))", tail)
    return " ".join(value.group(1) for value in (city, district) if value).strip()


def load_candidate_events(root: Path, device_id: str, max_location_accuracy_m: float = 200.0) -> list[dict[str, Any]]:
    """读取已清洗 candidate_data，转换为统计层事件；不读取原始 Excel。"""
    if max_location_accuracy_m <= 0:
        raise ValueError("max_location_accuracy_m必须为正数")
    directory = root / "candidate_data" / device_id
    output: list[dict[str, Any]] = []
    for source_type in ("photo", "location", "call"):
        path = directory / (source_type + ".jsonl.gz")
        if not path.is_file():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("disposition") != "eligible" or row.get("source_kind") != source_type:
                    continue
                data = row.get("data", {})
                event = {"event_id": row.get("record_key", ""), "device_id": device_id, "source_type": source_type,
                         "occurred_at": row["occurred_at"], "source_ref": {"source_file": row.get("source_file"), "sheet": row.get("sheet"), "row": row.get("row")}}
                if source_type == "location":
                    try:
                        accuracy = float(data.get("accuracy", "inf"))
                    except (TypeError, ValueError):
                        continue
                    region = _coarse_region(data.get("address"))
                    if accuracy > max_location_accuracy_m or not region:
                        continue
                    event["coarse_region"] = region
                output.append(event)
    return output


def aggregate(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按设备、来源和日期聚合；仅跨两个自然日生成 exploring 观察。"""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if not isinstance(event.get("device_id"), str) or not isinstance(event.get("occurred_at"), str):
            raise ValueError("分析事件缺少device_id或occurred_at")
        if event.get("source_type") not in {"photo", "location", "call", "product"}:
            raise ValueError("未知分析来源")
        datetime.fromisoformat(event["occurred_at"])
        groups[(event["device_id"], event["source_type"])].append(event)
    output = []
    for (device_id, source_type), rows in groups.items():
        dates = sorted({row["occurred_at"][:10] for row in rows})
        if len(dates) < 2:
            continue
        if source_type == "photo":
            label, kind = "持续拍摄行为模式", "photo_activity_pattern"
        elif source_type == "location":
            regions = sorted({str(row.get("coarse_region", "")) for row in rows if row.get("coarse_region")})
            if not regions:
                continue
            label, kind = "跨日期常活动区域", "coarse_activity_region"
        elif source_type == "call":
            label, kind = "跨日期通话使用模式", "call_activity_pattern"
        else:
            label, kind = "跨日期产品使用模式", "product_activity_pattern"
        output.append({"observation_id": digest(["analysis-v1", device_id, kind, dates, len(rows)])[:32],
                       "device_id": device_id, "observation_type": kind, "claim": label,
                       "state": "exploring", "period_start": min(row["occurred_at"] for row in rows),
                       "period_end": max(row["occurred_at"] for row in rows), "sample_count": len(rows),
                       "distinct_dates": len(dates), "source_event_ids": [str(row.get("event_id", index)) for index, row in enumerate(rows)],
                       "inference_limit": "统计行为模式，不等同于自述偏好；位置仅为粗粒度活动区域"})
    return output
