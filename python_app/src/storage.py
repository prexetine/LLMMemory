"""SQLite 持久化结构与基础写入接口。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS children (
    child_id TEXT PRIMARY KEY,
    nickname TEXT NOT NULL,
    birth_date TEXT,
    friend_name_enabled INTEGER NOT NULL DEFAULT 0 CHECK(friend_name_enabled IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    child_id TEXT NOT NULL REFERENCES children(child_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    memory_group TEXT NOT NULL,
    dimension TEXT NOT NULL,
    entity TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 100),
    sensitivity TEXT NOT NULL DEFAULT 'normal',
    source_type TEXT NOT NULL,
    source_device TEXT NOT NULL DEFAULT '',
    source_agent TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_child_entity ON memories(child_id, entity);

CREATE TABLE IF NOT EXISTS memory_evidence (
    evidence_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
    evidence_type TEXT NOT NULL,
    reference_id TEXT NOT NULL,
    excerpt TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    UNIQUE(memory_id, evidence_type, reference_id)
);

CREATE TABLE IF NOT EXISTS content_items (
    content_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    genre TEXT NOT NULL DEFAULT '',
    ability_dimension TEXT NOT NULL DEFAULT '',
    language_region TEXT NOT NULL DEFAULT '',
    min_age INTEGER,
    max_age INTEGER,
    status TEXT NOT NULL DEFAULT 'active',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS content_entities (
    content_id TEXT NOT NULL REFERENCES content_items(content_id) ON DELETE CASCADE,
    entity TEXT NOT NULL,
    PRIMARY KEY(content_id, entity)
);
CREATE INDEX IF NOT EXISTS idx_content_entities_entity ON content_entities(entity);

CREATE TABLE IF NOT EXISTS behavior_events (
    event_id TEXT PRIMARY KEY,
    child_id TEXT NOT NULL REFERENCES children(child_id) ON DELETE CASCADE,
    content_id TEXT NOT NULL REFERENCES content_items(content_id),
    event_type TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    progress REAL CHECK(progress IS NULL OR (progress BETWEEN 0 AND 1)),
    occurred_at TEXT NOT NULL,
    source_device TEXT NOT NULL DEFAULT '',
    source_agent TEXT NOT NULL DEFAULT '',
    interaction_id TEXT NOT NULL DEFAULT '',
    processed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_behavior_unprocessed ON behavior_events(processed_at, occurred_at);

CREATE TABLE IF NOT EXISTS processed_evidence (
    child_id TEXT NOT NULL,
    evidence_key TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(child_id, evidence_key)
);

CREATE TABLE IF NOT EXISTS entity_signals (
    child_id TEXT NOT NULL REFERENCES children(child_id) ON DELETE CASCADE,
    entity TEXT NOT NULL,
    positive_score REAL NOT NULL DEFAULT 0,
    negative_score REAL NOT NULL DEFAULT 0,
    last_event_at TEXT NOT NULL,
    PRIMARY KEY(child_id, entity)
);

CREATE TABLE IF NOT EXISTS recommendation_blacklist (
    child_id TEXT NOT NULL REFERENCES children(child_id) ON DELETE CASCADE,
    entity TEXT NOT NULL,
    reason_event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(child_id, entity)
);

CREATE TABLE IF NOT EXISTS forbidden_topics (
    child_id TEXT NOT NULL REFERENCES children(child_id) ON DELETE CASCADE,
    entity TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(child_id, entity)
);

CREATE TABLE IF NOT EXISTS listening_stats (
    child_id TEXT NOT NULL REFERENCES children(child_id) ON DELETE CASCADE,
    content_id TEXT NOT NULL REFERENCES content_items(content_id) ON DELETE CASCADE,
    complete_count INTEGER NOT NULL DEFAULT 0,
    max_progress REAL NOT NULL DEFAULT 0,
    reject_count INTEGER NOT NULL DEFAULT 0,
    last_event_at TEXT NOT NULL,
    PRIMARY KEY(child_id, content_id)
);

CREATE TABLE IF NOT EXISTS memory_usage_logs (
    usage_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
    child_id TEXT NOT NULL REFERENCES children(child_id) ON DELETE CASCADE,
    content_id TEXT NOT NULL REFERENCES content_items(content_id),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_runs (
    run_id TEXT PRIMARY KEY,
    child_id TEXT NOT NULL REFERENCES children(child_id) ON DELETE CASCADE,
    context_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Database:
    """管理 SQLite 连接和原子事务。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        """创建数据库目录和全部表。"""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """返回开启外键约束的行对象连接。"""

        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def upsert_child(
        self,
        child_id: str,
        nickname: str,
        birth_date: str | None = None,
        friend_name_enabled: bool = False,
    ) -> None:
        """新增或更新孩子基础资料。"""

        if not child_id.strip() or not nickname.strip():
            raise ValueError("child_id 和 nickname 不能为空")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO children(child_id, nickname, birth_date, friend_name_enabled)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(child_id) DO UPDATE SET
                    nickname=excluded.nickname,
                    birth_date=excluded.birth_date,
                    friend_name_enabled=excluded.friend_name_enabled
                """,
                (child_id.strip(), nickname.strip(), birth_date, int(friend_name_enabled)),
            )

    def upsert_content(
        self,
        content_id: str,
        title: str,
        entities: list[str],
        *,
        genre: str = "",
        ability_dimension: str = "",
        language_region: str = "",
        min_age: int | None = None,
        max_age: int | None = None,
        status: str = "active",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """写入内容快照及实体标签。"""

        if not content_id.strip() or not title.strip() or not entities:
            raise ValueError("content_id、title 和至少一个实体标签为必填项")
        if min_age is not None and max_age is not None and min_age > max_age:
            raise ValueError("min_age 不能大于 max_age")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO content_items(
                    content_id, title, genre, ability_dimension, language_region,
                    min_age, max_age, status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_id) DO UPDATE SET
                    title=excluded.title, genre=excluded.genre,
                    ability_dimension=excluded.ability_dimension,
                    language_region=excluded.language_region,
                    min_age=excluded.min_age, max_age=excluded.max_age,
                    status=excluded.status, metadata_json=excluded.metadata_json
                """,
                (
                    content_id.strip(), title.strip(), genre, ability_dimension,
                    language_region, min_age, max_age, status,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            connection.execute(
                "DELETE FROM content_entities WHERE content_id = ?", (content_id,)
            )
            connection.executemany(
                "INSERT INTO content_entities(content_id, entity) VALUES (?, ?)",
                [(content_id, entity) for entity in sorted(set(entities))],
            )

