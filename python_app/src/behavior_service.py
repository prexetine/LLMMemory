"""设备行为采集、幂等计分和记忆回流。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from memory_service import utc_now
from models import BehaviorEventInput, BehaviorType, MemoryStatus, TriggerType
from storage import Database
from taxonomy import Taxonomy


class BehaviorService:
    """处理六类行为事件并将偏好信号回流为候选记忆。"""

    def __init__(
        self,
        database: Database,
        config: dict[str, Any],
        taxonomy: Taxonomy,
    ) -> None:
        self._database = database
        self._behavior_config = config["behavior"]
        self._trigger_weights = config["trigger_weights"]
        self._taxonomy = taxonomy

    def ingest(self, event: BehaviorEventInput) -> bool:
        """写入一个事件；重复 event_id 返回 False，不重复入库。"""

        if not event.event_id.strip() or not event.child_id.strip():
            raise ValueError("event_id 和 child_id 不能为空")
        if event.progress is not None and not 0 <= event.progress <= 1:
            raise ValueError("progress 必须在 0 到 1 之间")
        if event.event_type in {BehaviorType.PLAY, BehaviorType.SKIP}:
            if event.progress is None:
                raise ValueError("播放和跳过事件必须提供 progress")
        occurred_at = event.occurred_at or utc_now()
        with self._database.connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO behavior_events(
                        event_id, child_id, content_id, event_type, trigger_type,
                        progress, occurred_at, source_device, source_agent,
                        interaction_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id, event.child_id, event.content_id,
                        event.event_type.value, event.trigger_type.value,
                        event.progress, occurred_at, event.source_device,
                        event.source_agent, event.interaction_id,
                    ),
                )
            except Exception as error:
                if "UNIQUE constraint failed: behavior_events.event_id" in str(error):
                    return False
                raise
        return True

    def process_pending(self) -> dict[str, int]:
        """原子处理未回流事件，并按频控创建候选记忆。"""

        processed = 0
        skipped_duplicate_evidence = 0
        with self._database.connect() as connection:
            events = connection.execute(
                "SELECT * FROM behavior_events WHERE processed_at IS NULL ORDER BY occurred_at, event_id"
            ).fetchall()
            for event in events:
                evidence_key = (
                    f"interaction:{event['interaction_id']}"
                    if event["interaction_id"]
                    else f"event:{event['event_id']}"
                )
                already_used = connection.execute(
                    "SELECT 1 FROM processed_evidence WHERE child_id=? AND evidence_key=?",
                    (event["child_id"], evidence_key),
                ).fetchone()
                if already_used:
                    connection.execute(
                        "UPDATE behavior_events SET processed_at=? WHERE event_id=?",
                        (utc_now(), event["event_id"]),
                    )
                    skipped_duplicate_evidence += 1
                    continue
                entities = [
                    self._taxonomy.normalize(row["entity"])
                    for row in connection.execute(
                        "SELECT entity FROM content_entities WHERE content_id=?",
                        (event["content_id"],),
                    ).fetchall()
                ]
                self._apply_event(connection, event, entities)
                connection.execute(
                    """
                    INSERT INTO processed_evidence(child_id, evidence_key, pipeline, created_at)
                    VALUES (?, ?, 'behavior', ?)
                    """,
                    (event["child_id"], evidence_key, utc_now()),
                )
                connection.execute(
                    "UPDATE behavior_events SET processed_at=? WHERE event_id=?",
                    (utc_now(), event["event_id"]),
                )
                processed += 1
            created = self._materialize_candidates(connection)
        return {
            "processed_events": processed,
            "duplicate_evidence_skipped": skipped_duplicate_evidence,
            "created_candidates": created,
        }

    def _apply_event(self, connection: Any, event: Any, entities: list[str]) -> None:
        weight = float(self._trigger_weights[event["trigger_type"]])
        score = self._base_score(event) * weight
        self._update_listening_stats(connection, event)
        for entity in entities:
            if event["event_type"] == BehaviorType.NEVER_RECOMMEND.value:
                connection.execute(
                    """
                    INSERT INTO recommendation_blacklist(
                        child_id, entity, reason_event_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(child_id, entity) DO UPDATE SET
                        reason_event_id=excluded.reason_event_id,
                        created_at=excluded.created_at
                    """,
                    (event["child_id"], entity, event["event_id"], event["occurred_at"]),
                )
                continue
            if score == 0:
                continue
            memory = connection.execute(
                """
                SELECT memory_id, status, confidence FROM memories
                WHERE child_id=? AND entity=? ORDER BY created_at DESC LIMIT 1
                """,
                (event["child_id"], entity),
            ).fetchone()
            if memory:
                self._update_existing_memory(connection, memory, event, score, weight)
            else:
                positive = max(score, 0)
                negative = abs(min(score, 0))
                connection.execute(
                    """
                    INSERT INTO entity_signals(
                        child_id, entity, positive_score, negative_score, last_event_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(child_id, entity) DO UPDATE SET
                        positive_score=positive_score + excluded.positive_score,
                        negative_score=negative_score + excluded.negative_score,
                        last_event_at=excluded.last_event_at
                    """,
                    (event["child_id"], entity, positive, negative, event["occurred_at"]),
                )

    def _base_score(self, event: Any) -> float:
        event_type = event["event_type"]
        progress = event["progress"] or 0.0
        if event_type in {
            BehaviorType.COMPLETE_LISTEN.value,
            BehaviorType.ACCEPT_RECOMMENDATION.value,
        }:
            return 1.0
        if event_type == BehaviorType.PLAY.value:
            return 1.0 if progress >= 0.9 else 0.0
        if event_type == BehaviorType.SKIP.value:
            return (
                float(self._behavior_config["skip_negative_score"])
                if progress >= 0.5
                else 0.0
            )
        if event_type == BehaviorType.REJECT.value:
            return -1.0
        return 0.0

    def _update_existing_memory(
        self, connection: Any, memory: Any, event: Any, score: float, weight: float
    ) -> None:
        if memory["status"] == MemoryStatus.PENDING.value:
            step = float(self._behavior_config["confidence_step"])
            if score > 0:
                confidence = min(
                    float(self._behavior_config["pending_confidence_cap"]),
                    memory["confidence"] + step * weight,
                )
            else:
                confidence = max(
                    float(self._behavior_config["pending_confidence_floor"]),
                    memory["confidence"] - step * abs(score),
                )
            connection.execute(
                "UPDATE memories SET confidence=?, updated_at=? WHERE memory_id=?",
                (confidence, utc_now(), memory["memory_id"]),
            )
        elif memory["status"] == MemoryStatus.CONFIRMED.value and score > 0:
            connection.execute(
                """
                INSERT INTO memory_usage_logs(
                    usage_id, memory_id, child_id, content_id, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"use_{uuid4().hex}", memory["memory_id"], event["child_id"],
                    event["content_id"], "行为正信号命中已确认记忆", utc_now(),
                ),
            )

    def _update_listening_stats(self, connection: Any, event: Any) -> None:
        complete = int(
            event["event_type"] == BehaviorType.COMPLETE_LISTEN.value
            or (
                event["event_type"] == BehaviorType.PLAY.value
                and (event["progress"] or 0) >= 0.9
            )
        )
        reject = int(event["event_type"] == BehaviorType.REJECT.value)
        progress = float(event["progress"] or complete)
        connection.execute(
            """
            INSERT INTO listening_stats(
                child_id, content_id, complete_count, max_progress,
                reject_count, last_event_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(child_id, content_id) DO UPDATE SET
                complete_count=complete_count + excluded.complete_count,
                max_progress=MAX(max_progress, excluded.max_progress),
                reject_count=reject_count + excluded.reject_count,
                last_event_at=excluded.last_event_at
            """,
            (
                event["child_id"], event["content_id"], complete, progress,
                reject, event["occurred_at"],
            ),
        )

    def _materialize_candidates(self, connection: Any) -> int:
        threshold = float(self._behavior_config["positive_threshold"])
        week_start = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        weekly_limit = int(self._behavior_config["weekly_pending_limit"])
        batch_limit = int(self._behavior_config["batch_candidate_limit"])
        qualifying = connection.execute(
            """
            SELECT s.* FROM entity_signals s
            LEFT JOIN memories m ON m.child_id=s.child_id AND m.entity=s.entity
            LEFT JOIN recommendation_blacklist b
                ON b.child_id=s.child_id AND b.entity=s.entity
            WHERE s.positive_score >= ? AND m.memory_id IS NULL AND b.entity IS NULL
            ORDER BY s.positive_score DESC, s.last_event_at DESC, s.entity
            """,
            (threshold,),
        ).fetchall()
        created = 0
        for signal in qualifying:
            if created >= batch_limit:
                break
            child_id = signal["child_id"]
            existing_this_week = connection.execute(
                """
                SELECT COUNT(*) AS count FROM memories
                WHERE child_id=? AND source_type='behavior'
                    AND created_at>=?
                """,
                (child_id, week_start),
            ).fetchone()["count"]
            if existing_this_week >= weekly_limit:
                continue
            memory_id = f"mem_{uuid4().hex}"
            now = utc_now()
            entity = signal["entity"]
            source = connection.execute(
                """
                SELECT e.source_device, e.source_agent
                FROM behavior_events e
                JOIN content_entities ce ON ce.content_id=e.content_id
                WHERE e.child_id=? AND ce.entity=?
                ORDER BY e.occurred_at DESC LIMIT 1
                """,
                (child_id, entity),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO memories(
                    memory_id, child_id, title, summary, memory_group, dimension,
                    entity, status, confidence, sensitivity, source_type,
                    source_device, source_agent, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'profile', '偏好-喜欢的主题', ?, 'pending', ?,
                          'normal', 'behavior', ?, ?, ?, ?)
                """,
                (
                    memory_id, child_id, f"对{entity}持续感兴趣",
                    f"多次正向收听或接受了{entity}相关内容", entity,
                    float(self._behavior_config["initial_confidence"]),
                    source["source_device"] if source else "",
                    source["source_agent"] if source else "",
                    now, now,
                ),
            )
            evidence_rows = connection.execute(
                """
                SELECT DISTINCT e.event_id, e.event_type, e.occurred_at, c.title
                FROM behavior_events e
                JOIN content_items c ON c.content_id=e.content_id
                JOIN content_entities ce ON ce.content_id=e.content_id
                WHERE e.child_id=? AND ce.entity=?
                  AND (e.event_type IN ('complete_listen', 'accept_recommendation')
                       OR (e.event_type='play' AND e.progress>=0.9))
                ORDER BY e.occurred_at DESC LIMIT 20
                """,
                (child_id, entity),
            ).fetchall()
            connection.executemany(
                """
                INSERT INTO memory_evidence(
                    evidence_id, memory_id, evidence_type, reference_id,
                    excerpt, occurred_at
                ) VALUES (?, ?, 'behavior', ?, ?, ?)
                """,
                [
                    (
                        f"evd_{uuid4().hex}", memory_id, row["event_id"],
                        f"{row['event_type']}：《{row['title']}》", row["occurred_at"],
                    )
                    for row in evidence_rows
                ],
            )
            created += 1
        return created

    def remove_blacklist(self, child_id: str, entity: str) -> bool:
        """由家长解除一个实体的“不再推荐”。"""

        normalized = self._taxonomy.normalize(entity)
        with self._database.connect() as connection:
            result = connection.execute(
                "DELETE FROM recommendation_blacklist WHERE child_id=? AND entity=?",
                (child_id, normalized),
            )
            return result.rowcount > 0

