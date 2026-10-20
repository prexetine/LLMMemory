"""记忆生命周期、证据和家长治理。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from models import MemoryGroup, MemoryStatus, Sensitivity
from storage import Database


def utc_now() -> str:
    """返回可排序的 UTC ISO 时间。"""

    return datetime.now(timezone.utc).isoformat()


class MemoryService:
    """管理记忆创建、审核、停用和彻底删除。"""

    def __init__(self, database: Database) -> None:
        self._database = database

    def create_memory(
        self,
        *,
        child_id: str,
        title: str,
        summary: str,
        memory_group: MemoryGroup,
        dimension: str,
        entity: str,
        status: MemoryStatus,
        confidence: float,
        source_type: str,
        sensitivity: Sensitivity = Sensitivity.NORMAL,
        source_device: str = "",
        source_agent: str = "",
        evidence: list[dict[str, str]] | None = None,
    ) -> str:
        """创建一条带直接证据的记忆。"""

        if not entity.strip() or not title.strip() or not summary.strip():
            raise ValueError("记忆标题、摘要和实体不能为空")
        if not 0 <= confidence <= 100:
            raise ValueError("记忆置信度必须在 0 到 100 之间")
        if not evidence:
            raise ValueError("记忆必须至少包含一条直接证据")
        memory_id = f"mem_{uuid4().hex}"
        now = utc_now()
        with self._database.connect() as connection:
            duplicate = connection.execute(
                "SELECT memory_id FROM memories WHERE child_id=? AND entity=?",
                (child_id, entity),
            ).fetchone()
            if duplicate:
                raise ValueError(f"孩子已有实体 {entity} 的记忆")
            connection.execute(
                """
                INSERT INTO memories(
                    memory_id, child_id, title, summary, memory_group, dimension,
                    entity, status, confidence, sensitivity, source_type,
                    source_device, source_agent, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id, child_id, title.strip(), summary.strip(),
                    memory_group.value, dimension.strip(), entity.strip(),
                    status.value, confidence, sensitivity.value, source_type,
                    source_device, source_agent, now, now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO memory_evidence(
                    evidence_id, memory_id, evidence_type, reference_id,
                    excerpt, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f"evd_{uuid4().hex}", memory_id, item["evidence_type"],
                        item["reference_id"], item["excerpt"], item["occurred_at"],
                    )
                    for item in evidence
                ],
            )
            if source_type == "dialogue":
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO processed_evidence(
                        child_id, evidence_key, pipeline, created_at
                    ) VALUES (?, ?, 'dialogue', ?)
                    """,
                    [
                        (child_id, f"interaction:{item['reference_id']}", now)
                        for item in evidence
                    ],
                )
        return memory_id

    def parent_review(self, memory_id: str, action: str) -> str:
        """执行家长审核操作并返回新状态。"""

        transitions: dict[tuple[str, str], str] = {
            (MemoryStatus.PENDING.value, "confirm"): MemoryStatus.CONFIRMED.value,
            (MemoryStatus.PENDING.value, "ignore"): MemoryStatus.IGNORED.value,
            (MemoryStatus.CONFLICT.value, "confirm"): MemoryStatus.CONFIRMED.value,
            (MemoryStatus.CONFLICT.value, "ignore"): MemoryStatus.IGNORED.value,
            (MemoryStatus.IGNORED.value, "restore"): MemoryStatus.PENDING.value,
            (MemoryStatus.CONFIRMED.value, "stop"): MemoryStatus.STOPPED.value,
            (MemoryStatus.STOPPED.value, "restore"): MemoryStatus.CONFIRMED.value,
        }
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT status FROM memories WHERE memory_id=?", (memory_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"记忆不存在: {memory_id}")
            new_status = transitions.get((row["status"], action))
            if not new_status:
                raise ValueError(f"状态 {row['status']} 不允许执行 {action}")
            connection.execute(
                "UPDATE memories SET status=?, updated_at=? WHERE memory_id=?",
                (new_status, utc_now(), memory_id),
            )
            return new_status

    def permanently_delete(self, memory_id: str, *, confirmed: bool) -> None:
        """彻底删除记忆及证据、调用记录和关联数据。"""

        if not confirmed:
            raise ValueError("永久删除必须显式二次确认")
        with self._database.connect() as connection:
            result = connection.execute(
                "DELETE FROM memories WHERE memory_id=?", (memory_id,)
            )
            if result.rowcount == 0:
                raise KeyError(f"记忆不存在: {memory_id}")

    def list_memories(self, child_id: str) -> list[dict[str, Any]]:
        """返回孩子全部记忆及其证据。"""

        with self._database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories WHERE child_id=? ORDER BY created_at DESC",
                (child_id,),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                evidence = connection.execute(
                    """
                    SELECT evidence_type, reference_id, excerpt, occurred_at
                    FROM memory_evidence WHERE memory_id=? ORDER BY occurred_at
                    """,
                    (row["memory_id"],),
                ).fetchall()
                item["evidence"] = [dict(value) for value in evidence]
                result.append(item)
            return result

