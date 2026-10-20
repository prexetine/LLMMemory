"""ToyMemory 行为、记忆与推荐闭环测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from behavior_service import BehaviorService  # noqa: E402
from config import load_config  # noqa: E402
from memory_service import MemoryService  # noqa: E402
from models import (  # noqa: E402
    BehaviorEventInput,
    BehaviorType,
    MemoryGroup,
    MemoryStatus,
    RecommendationContext,
    Sensitivity,
    TriggerType,
)
from recommendation_service import RecommendationService  # noqa: E402
from storage import Database  # noqa: E402
from taxonomy import Taxonomy  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.json"


class PipelineTestCase(unittest.TestCase):
    """为每个测试创建隔离的 SQLite 数据库。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temp_dir.name) / "test.sqlite3")
        self.database.initialize()
        self.config = load_config(CONFIG_PATH)
        self.taxonomy = Taxonomy()
        self.memories = MemoryService(self.database)
        self.behaviors = BehaviorService(self.database, self.config, self.taxonomy)
        self.recommendations = RecommendationService(
            self.database, self.config, self.taxonomy
        )
        self.database.upsert_child("child", "小宇", "2021-05-20")
        self.database.upsert_content(
            "dolphin_story", "会唱歌的小海豚", ["海豚"],
            genre="故事", min_age=3, max_age=8,
        )
        self.database.upsert_content(
            "dolphin_science", "海豚为什么会跳跃", ["海豚"],
            genre="知识", min_age=4, max_age=9,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _complete_event(
        self,
        event_id: str,
        *,
        content_id: str = "dolphin_story",
        interaction_id: str = "",
        trigger: TriggerType = TriggerType.ACTIVE_REQUEST,
    ) -> BehaviorEventInput:
        return BehaviorEventInput(
            event_id=event_id,
            child_id="child",
            content_id=content_id,
            event_type=BehaviorType.COMPLETE_LISTEN,
            trigger_type=trigger,
            interaction_id=interaction_id,
        )


class BehaviorFlowTests(PipelineTestCase):
    def test_three_positive_points_create_auditable_pending_memory(self) -> None:
        for index in range(3):
            self.assertTrue(self.behaviors.ingest(self._complete_event(f"event_{index}")))
        self.assertFalse(self.behaviors.ingest(self._complete_event("event_0")))
        result = self.behaviors.process_pending()
        self.assertEqual(3, result["processed_events"])
        self.assertEqual(1, result["created_candidates"])
        memory = self.memories.list_memories("child")[0]
        self.assertEqual("海豚", memory["entity"])
        self.assertEqual(MemoryStatus.PENDING.value, memory["status"])
        self.assertEqual(65.0, memory["confidence"])
        self.assertEqual(3, len(memory["evidence"]))
        second = self.behaviors.process_pending()
        self.assertEqual(0, second["processed_events"])
        self.assertEqual(0, second["created_candidates"])

    def test_pending_confidence_respects_cap_and_floor(self) -> None:
        for index in range(3):
            self.behaviors.ingest(self._complete_event(f"seed_{index}"))
        self.behaviors.process_pending()
        for index in range(10):
            self.behaviors.ingest(self._complete_event(f"positive_{index}"))
        self.behaviors.process_pending()
        self.assertEqual(84.0, self.memories.list_memories("child")[0]["confidence"])
        for index in range(20):
            self.behaviors.ingest(
                BehaviorEventInput(
                    event_id=f"reject_{index}", child_id="child",
                    content_id="dolphin_story", event_type=BehaviorType.REJECT,
                    trigger_type=TriggerType.ACTIVE_REQUEST,
                )
            )
        self.behaviors.process_pending()
        memory = self.memories.list_memories("child")[0]
        self.assertEqual(40.0, memory["confidence"])
        self.assertEqual(MemoryStatus.PENDING.value, memory["status"])

    def test_dialogue_and_behavior_share_one_evidence_budget(self) -> None:
        memory_id = self.memories.create_memory(
            child_id="child", title="喜欢海豚", summary="对话中反复提到海豚",
            memory_group=MemoryGroup.PROFILE, dimension="偏好-喜欢的主题",
            entity="海豚", status=MemoryStatus.PENDING, confidence=60,
            source_type="dialogue",
            evidence=[{
                "evidence_type": "dialogue", "reference_id": "same_interaction",
                "excerpt": "我还想听海豚。",
                "occurred_at": "2026-08-27T08:00:00+00:00",
            }],
        )
        self.behaviors.ingest(
            self._complete_event("behavior_copy", interaction_id="same_interaction")
        )
        result = self.behaviors.process_pending()
        self.assertEqual(1, result["duplicate_evidence_skipped"])
        memory = next(
            item for item in self.memories.list_memories("child")
            if item["memory_id"] == memory_id
        )
        self.assertEqual(60.0, memory["confidence"])

    def test_never_recommend_blacklists_without_changing_memory(self) -> None:
        for index in range(3):
            self.behaviors.ingest(self._complete_event(f"seed_black_{index}"))
        self.behaviors.process_pending()
        memory = self.memories.list_memories("child")[0]
        self.memories.parent_review(memory["memory_id"], "confirm")
        self.behaviors.ingest(
            BehaviorEventInput(
                event_id="never", child_id="child", content_id="dolphin_story",
                event_type=BehaviorType.NEVER_RECOMMEND,
                trigger_type=TriggerType.ACTIVE_REQUEST,
            )
        )
        self.behaviors.process_pending()
        self.assertEqual([], self.recommendations.recommend(RecommendationContext("child")))
        self.assertEqual(
            MemoryStatus.CONFIRMED.value,
            self.memories.list_memories("child")[0]["status"],
        )
        self.assertTrue(self.behaviors.remove_blacklist("child", "海豚"))
        self.assertTrue(self.recommendations.recommend(RecommendationContext("child")))

    def test_idle_push_counts_half_weight(self) -> None:
        for index in range(5):
            self.behaviors.ingest(
                self._complete_event(f"idle_{index}", trigger=TriggerType.IDLE_PUSH)
            )
        self.behaviors.process_pending()
        self.assertEqual([], self.memories.list_memories("child"))
        self.behaviors.ingest(
            self._complete_event("idle_6", trigger=TriggerType.IDLE_PUSH)
        )
        self.behaviors.process_pending()
        self.assertEqual(1, len(self.memories.list_memories("child")))


class MemoryGovernanceTests(PipelineTestCase):
    def test_parent_controls_lifecycle_and_delete_requires_confirmation(self) -> None:
        memory_id = self.memories.create_memory(
            child_id="child", title="喜欢海豚", summary="有直接对话证据",
            memory_group=MemoryGroup.PROFILE, dimension="偏好",
            entity="海豚", status=MemoryStatus.PENDING, confidence=70,
            source_type="dialogue",
            evidence=[{
                "evidence_type": "dialogue", "reference_id": "dialogue_1",
                "excerpt": "我最喜欢海豚。", "occurred_at": "2026-08-27T08:00:00+00:00",
            }],
        )
        self.assertEqual("confirmed", self.memories.parent_review(memory_id, "confirm"))
        self.assertEqual("stopped", self.memories.parent_review(memory_id, "stop"))
        self.assertEqual("confirmed", self.memories.parent_review(memory_id, "restore"))
        with self.assertRaises(ValueError):
            self.memories.permanently_delete(memory_id, confirmed=False)
        self.memories.permanently_delete(memory_id, confirmed=True)
        self.assertEqual([], self.memories.list_memories("child"))


class RecommendationTests(PipelineTestCase):
    def _create_memory(
        self,
        *,
        status: MemoryStatus = MemoryStatus.CONFIRMED,
        sensitivity: Sensitivity = Sensitivity.NORMAL,
        entity: str = "海豚",
    ) -> str:
        return self.memories.create_memory(
            child_id="child", title=f"喜欢{entity}", summary=f"对{entity}感兴趣",
            memory_group=MemoryGroup.PROFILE, dimension="偏好",
            entity=entity, status=status, confidence=75, source_type="dialogue",
            sensitivity=sensitivity,
            evidence=[{
                "evidence_type": "dialogue", "reference_id": f"dialogue_{entity}",
                "excerpt": f"我喜欢{entity}。", "occurred_at": "2026-08-27T08:00:00+00:00",
            }],
        )

    def test_confirmed_and_pending_memories_have_different_speech_modes(self) -> None:
        self._create_memory(status=MemoryStatus.PENDING)
        items = self.recommendations.recommend(RecommendationContext("child", limit=1))
        self.assertEqual("tentative_question", items[0].speech_mode)
        self.assertIn("待家长确认", items[0].reason)

    def test_parent_forbidden_topic_is_hard_filter(self) -> None:
        self._create_memory()
        self.recommendations.set_forbidden_topic("child", "海豚")
        self.assertEqual([], self.recommendations.recommend(RecommendationContext("child")))
        self.assertTrue(self.recommendations.remove_forbidden_topic("child", "海豚"))
        self.assertTrue(self.recommendations.recommend(RecommendationContext("child")))

    def test_sensitive_memory_requires_child_initiated_context(self) -> None:
        self._create_memory(sensitivity=Sensitivity.FEAR)
        self.assertEqual([], self.recommendations.recommend(RecommendationContext("child")))
        allowed = self.recommendations.recommend(
            RecommendationContext(
                "child", conversation_entities=("海豚",),
                child_initiated_sensitive_topic=True, limit=1,
            )
        )
        self.assertEqual(1, len(allowed))

    def test_secret_memory_is_never_used(self) -> None:
        self._create_memory(sensitivity=Sensitivity.SECRET)
        result = self.recommendations.recommend(
            RecommendationContext(
                "child", conversation_entities=("海豚",),
                child_initiated_sensitive_topic=True,
            )
        )
        self.assertEqual([], result)

    def test_age_and_content_status_are_hard_filters(self) -> None:
        self._create_memory()
        self.database.upsert_content(
            "adult_dolphin", "成人海豚纪录片", ["海豚"], min_age=18, max_age=99
        )
        self.database.upsert_content(
            "blocked_dolphin", "已屏蔽海豚内容", ["海豚"], status="blocked"
        )
        items = self.recommendations.recommend(RecommendationContext("child"))
        self.assertNotIn("adult_dolphin", {item.content_id for item in items})
        self.assertNotIn("blocked_dolphin", {item.content_id for item in items})

    def test_recommendation_keeps_memory_usage_and_run_log(self) -> None:
        self._create_memory()
        items = self.recommendations.recommend(RecommendationContext("child", limit=1))
        self.assertEqual(1, len(items))
        with self.database.connect() as connection:
            self.assertEqual(
                1, connection.execute("SELECT COUNT(*) FROM recommendation_runs").fetchone()[0]
            )
            self.assertEqual(
                1, connection.execute("SELECT COUNT(*) FROM memory_usage_logs").fetchone()[0]
            )


class ValidationTests(PipelineTestCase):
    def test_invalid_progress_and_empty_content_entities_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.behaviors.ingest(
                BehaviorEventInput(
                    event_id="bad", child_id="child", content_id="dolphin_story",
                    event_type=BehaviorType.PLAY,
                    trigger_type=TriggerType.ACTIVE_REQUEST, progress=1.5,
                )
            )
        with self.assertRaises(ValueError):
            self.database.upsert_content("bad_content", "无标签", [])

    def test_taxonomy_supports_synonym_and_hierarchy(self) -> None:
        self.assertEqual("恐龙", self.taxonomy.normalize("大恐龙"))
        self.assertTrue(self.taxonomy.matches("恐龙", "霸王龙"))
        self.assertFalse(self.taxonomy.matches("海豚", "霸王龙"))


if __name__ == "__main__":
    unittest.main()

