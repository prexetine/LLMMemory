"""根据 extraction_summary.json 生成运行所需的名单与清单元数据。

memory_agent 读取 data_root 下的 users.csv（load_user_roster 的兜底名单），
并在缺 core_50_provisional.csv 时使用它。本脚本同时产出 manifest.json 便于溯源。

用法：
  python build_1017_roster.py --root data/1017
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CORE = {
    "12354838B667950A", "12354838B66792A5", "12354838B6679432",
    "12354838B667A7F0", "12354838B667A861",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data/1017")
    args = parser.parse_args()
    root: Path = args.root

    summary = json.loads((root / "extraction_summary.json").read_text(encoding="utf-8"))
    per_user = summary["per_user"]

    rows = []
    for device in sorted(per_user):
        counts = per_user[device]
        rows.append({
            "device_id": device,
            "is_seed": int(device not in CORE),
            "is_existing_candidate": 0,
            "is_core_50": int(device in CORE),
            "dialogue_records": counts.get("dialogue", 0),
            "photo_records": counts.get("photo", 0),
            "call_records": counts.get("call", 0),
            "location_records": counts.get("location", 0),
        })

    fields = ["device_id", "is_seed", "is_existing_candidate", "is_core_50",
              "dialogue_records", "photo_records", "call_records", "location_records"]
    with (root / "users.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": "user-data-batches-v2",
        "batch": "1017",
        "data_root": str(root),
        "source_directory": "data/1017",
        "source_files": summary["kinds"],
        "max_date": summary["max_date"],
        "totals": {
            kind: sum(counts.get(kind, 0) for counts in per_user.values())
            for kind in ("dialogue", "photo", "call", "location")
        },
        "users": rows,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(f"写出 {root / 'users.csv'}（{len(rows)} 用户）")
    print(f"写出 {root / 'manifest.json'}  合计 {manifest['totals']}")


if __name__ == "__main__":
    main()
