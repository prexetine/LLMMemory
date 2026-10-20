"""ToyMemory 领域枚举和数据传输对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryStatus(str, Enum):
    """记忆生命周期状态。"""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    OBSERVING = "observing"
    CONFLICT = "conflict"
    IGNORED = "ignored"
    STOPPED = "stopped"


class MemoryGroup(str, Enum):
    """需求定义的四类记忆。"""

    PROFILE = "profile"
    EXPERIENCE = "experience"
    ABILITY_GRAPH = "ability_graph"
    CHILD_GROWTH = "child_growth"


class Sensitivity(str, Enum):
    """记忆敏感级别。"""

    NORMAL = "normal"
    SECRET = "secret"
    FEAR = "fear"
    EMOTION = "emotion"
    FAMILY_EVENT = "family_event"
    FRIEND_REAL_NAME = "friend_real_name"


class BehaviorType(str, Enum):
    """设备端可上报的六类行为事件。"""

    COMPLETE_LISTEN = "complete_listen"
    PLAY = "play"
    SKIP = "skip"
    ACCEPT_RECOMMENDATION = "accept_recommendation"
    REJECT = "reject"
    NEVER_RECOMMEND = "never_recommend"


class TriggerType(str, Enum):
    """行为触发方式。"""

    ACTIVE_REQUEST = "active_request"
    ACCEPTED_RECOMMENDATION = "accepted_recommendation"
    IDLE_PUSH = "idle_push"


@dataclass(frozen=True)
class BehaviorEventInput:
    """行为事件写入参数。"""

    event_id: str
    child_id: str
    content_id: str
    event_type: BehaviorType
    trigger_type: TriggerType
    progress: float | None = None
    occurred_at: str | None = None
    source_device: str = ""
    source_agent: str = ""
    interaction_id: str = ""


@dataclass(frozen=True)
class RecommendationContext:
    """推荐请求上下文。"""

    child_id: str
    conversation_entities: tuple[str, ...] = ()
    child_initiated_sensitive_topic: bool = False
    limit: int | None = None


@dataclass(frozen=True)
class RecommendationItem:
    """带解释依据的推荐结果。"""

    content_id: str
    title: str
    entity: str
    memory_id: str
    memory_status: str
    confidence: float
    reason: str
    speech_mode: str
    rank_factors: dict[str, Any] = field(default_factory=dict)

