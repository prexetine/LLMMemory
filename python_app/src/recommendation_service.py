"""可审计的内容召回、硬过滤和确定性排序。"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from memory_service import utc_now
from models import MemoryStatus, RecommendationContext, RecommendationItem, Sensitivity
from storage import Database
from taxonomy import Taxonomy


BLOCKED_SENSITIVITIES = {
    Sensitivity.FEAR.value,
    Sensitivity.EMOTION.value,
    Sensitivity.FAMILY_EVENT.value,
}


class RecommendationService:
    """执行规则召回，并保存每次推荐的记忆依据。"""

    def __init__(
        self,
        database: Database,
        config: dict[str, Any],
        taxonomy: Taxonomy,
    ) -> None:
        self._database = database
        self._config = config["recommendation"]
        self._taxonomy = taxonomy

    def recommend(self, context: RecommendationContext) -> list[RecommendationItem]:
        """为孩子生成最多 2 至 3 条带依据的推荐。"""

        limit = context.limit or int(self._config["default_limit"])
        if not 1 <= limit <= 3:
            raise ValueError("语音场景单次推荐数量必须在 1 到 3 之间")
        conversation_entities = {
            self._taxonomy.normalize(value) for value in context.conversation_entities
        }
        with self._database.connect() as connection:
            child = connection.execute(
                "SELECT * FROM children WHERE child_id=?", (context.child_id,)
            ).fetchone()
            if not child:
                raise KeyError(f"孩子不存在: {context.child_id}")
            age = self._calculate_age(child["birth_date"])
            blacklist = {
                row["entity"]
                for row in connection.execute(
                    "SELECT entity FROM recommendation_blacklist WHERE child_id=?",
                    (context.child_id,),
                ).fetchall()
            }
            forbidden = {
                row["entity"]
                for row in connection.execute(
                    "SELECT entity FROM forbidden_topics WHERE child_id=?",
                    (context.child_id,),
                ).fetchall()
            }
            memories = connection.execute(
                """
                SELECT * FROM memories
                WHERE child_id=? AND status IN ('pending', 'confirmed')
                """,
                (context.child_id,),
            ).fetchall()
            contents = connection.execute(
                """
                SELECT c.*, ce.entity AS content_entity,
                       COALESCE(s.complete_count, 0) AS complete_count,
                       COALESCE(s.max_progress, 0) AS max_progress,
                       COALESCE(s.reject_count, 0) AS reject_count
                FROM content_items c
                JOIN content_entities ce ON ce.content_id=c.content_id
                LEFT JOIN listening_stats s
                  ON s.content_id=c.content_id AND s.child_id=?
                """,
                (context.child_id,),
            ).fetchall()
            candidates: dict[str, tuple[tuple[Any, ...], RecommendationItem]] = {}
            recent_cutoff = (
                datetime.now(timezone.utc)
                - timedelta(days=int(self._config["recent_days"]))
            ).isoformat()
            for memory in memories:
                if not self._memory_is_usable(
                    memory, child, conversation_entities,
                    context.child_initiated_sensitive_topic,
                ):
                    continue
                for content in contents:
                    content_entity = self._taxonomy.normalize(content["content_entity"])
                    if not self._taxonomy.matches(memory["entity"], content_entity):
                        continue
                    if not self._content_is_allowed(
                        content, content_entity, age, blacklist, forbidden
                    ):
                        continue
                    recent = connection.execute(
                        """
                        SELECT 1 FROM behavior_events e
                        JOIN content_entities ce ON ce.content_id=e.content_id
                        WHERE e.child_id=? AND ce.entity=? AND e.occurred_at>=?
                        LIMIT 1
                        """,
                        (context.child_id, content_entity, recent_cutoff),
                    ).fetchone()
                    conversation_match = int(
                        any(
                            self._taxonomy.matches(value, content_entity)
                            for value in conversation_entities
                        )
                    )
                    confirmed = int(memory["status"] == MemoryStatus.CONFIRMED.value)
                    unheard = int(content["complete_count"] == 0 and content["max_progress"] == 0)
                    unfinished = int(
                        content["complete_count"] == 0 and 0 < content["max_progress"] < 0.9
                    )
                    not_rejected = int(content["reject_count"] == 0)
                    rank = (
                        conversation_match,
                        confirmed,
                        float(memory["confidence"]),
                        int(bool(recent)),
                        unheard,
                        unfinished,
                        -int(content["complete_count"]),
                        not_rejected,
                        content["title"],
                    )
                    speech_mode = "natural_assertion" if confirmed else "tentative_question"
                    status_text = "家长已确认" if confirmed else "待家长确认"
                    reason = (
                        f"依据记忆“{memory['title']}”（{status_text}，"
                        f"置信度 {memory['confidence']:.0f}），匹配实体“{content_entity}”"
                    )
                    item = RecommendationItem(
                        content_id=content["content_id"],
                        title=content["title"],
                        entity=content_entity,
                        memory_id=memory["memory_id"],
                        memory_status=memory["status"],
                        confidence=float(memory["confidence"]),
                        reason=reason,
                        speech_mode=speech_mode,
                        rank_factors={
                            "conversation_match": bool(conversation_match),
                            "confirmed": bool(confirmed),
                            "recent_signal": bool(recent),
                            "unheard": bool(unheard),
                            "unfinished": bool(unfinished),
                            "complete_count": int(content["complete_count"]),
                            "not_rejected": bool(not_rejected),
                        },
                    )
                    current = candidates.get(content["content_id"])
                    if current is None or rank > current[0]:
                        candidates[content["content_id"]] = (rank, item)
            selected = [
                value[1]
                for value in sorted(candidates.values(), key=lambda value: value[0], reverse=True)
            ][:limit]
            run_id = f"run_{uuid4().hex}"
            now = utc_now()
            connection.execute(
                """
                INSERT INTO recommendation_runs(
                    run_id, child_id, context_json, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    context.child_id,
                    json.dumps(
                        {
                            "conversation_entities": list(context.conversation_entities),
                            "child_initiated_sensitive_topic": (
                                context.child_initiated_sensitive_topic
                            ),
                            "limit": limit,
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        [item.__dict__ for item in selected], ensure_ascii=False
                    ),
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO memory_usage_logs(
                    usage_id, memory_id, child_id, content_id, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f"use_{uuid4().hex}", item.memory_id, context.child_id,
                        item.content_id, item.reason, now,
                    )
                    for item in selected
                ],
            )
            return selected

    def set_forbidden_topic(self, child_id: str, entity: str) -> None:
        """由家长设置禁止推荐主题。"""

        normalized = self._taxonomy.normalize(entity)
        with self._database.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO forbidden_topics(child_id, entity, created_at)
                VALUES (?, ?, ?)
                """,
                (child_id, normalized, utc_now()),
            )

    def remove_forbidden_topic(self, child_id: str, entity: str) -> bool:
        """由家长解除一个禁止主题。"""

        normalized = self._taxonomy.normalize(entity)
        with self._database.connect() as connection:
            result = connection.execute(
                "DELETE FROM forbidden_topics WHERE child_id=? AND entity=?",
                (child_id, normalized),
            )
            return result.rowcount > 0

    def _memory_is_usable(
        self,
        memory: Any,
        child: Any,
        conversation_entities: set[str],
        child_initiated_sensitive_topic: bool,
    ) -> bool:
        sensitivity = memory["sensitivity"]
        if sensitivity == Sensitivity.SECRET.value:
            return False
        if sensitivity in BLOCKED_SENSITIVITIES:
            return (
                child_initiated_sensitive_topic
                and self._taxonomy.normalize(memory["entity"]) in conversation_entities
            )
        if sensitivity == Sensitivity.FRIEND_REAL_NAME.value:
            return bool(child["friend_name_enabled"])
        return True

    def _content_is_allowed(
        self,
        content: Any,
        content_entity: str,
        age: int | None,
        blacklist: set[str],
        forbidden: set[str],
    ) -> bool:
        if content["status"] != "active":
            return False
        content_family = self._taxonomy.ancestors(content_entity)
        if blacklist.intersection(content_family) or forbidden.intersection(content_family):
            return False
        has_age_constraint = content["min_age"] is not None or content["max_age"] is not None
        if age is None and has_age_constraint:
            return False
        if age is not None:
            if content["min_age"] is not None and age < content["min_age"]:
                return False
            if content["max_age"] is not None and age > content["max_age"]:
                return False
        return True

    @staticmethod
    def _calculate_age(birth_date: str | None) -> int | None:
        if not birth_date:
            return None
        born = date.fromisoformat(birth_date)
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

