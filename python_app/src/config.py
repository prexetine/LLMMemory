"""UTF-8 配置加载与字段校验。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """加载 JSON 配置并验证核心参数。"""

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    behavior = config.get("behavior")
    weights = config.get("trigger_weights")
    recommendation = config.get("recommendation")
    if not isinstance(behavior, dict) or not isinstance(weights, dict):
        raise ValueError("配置缺少 behavior 或 trigger_weights 对象")
    if not isinstance(recommendation, dict):
        raise ValueError("配置缺少 recommendation 对象")
    threshold = float(behavior.get("positive_threshold", 0))
    if threshold <= 0:
        raise ValueError("behavior.positive_threshold 必须大于 0")
    weekly_limit = int(behavior.get("weekly_pending_limit", 0))
    batch_limit = int(behavior.get("batch_candidate_limit", 0))
    if weekly_limit <= 0 or batch_limit <= 0:
        raise ValueError("待确认频控参数必须大于 0")
    for name in ("active_request", "accepted_recommendation", "idle_push"):
        weight = float(weights.get(name, -1))
        if not 0 <= weight <= 1:
            raise ValueError(f"trigger_weights.{name} 必须在 0 到 1 之间")
    return config


def resolve_database_path(config_path: Path, configured_path: str) -> Path:
    """从项目根目录解析数据库路径，不依赖当前工作目录。"""

    project_root = config_path.resolve().parents[2]
    candidate = Path(configured_path)
    return candidate if candidate.is_absolute() else project_root / candidate

