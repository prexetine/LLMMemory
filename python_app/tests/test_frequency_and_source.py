"""候选频控与行为来源继承测试。"""

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
from models import BehaviorEventInput, BehaviorType, TriggerType  # noqa: E402
from storage import Database  # noqa: E402
from taxonomy import Taxonomy  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.json"


class FrequencyAndSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temp_dir.name) / "test.sqlite3")
        self.database.initialize()
        self.database.upsert_child("child", "小宇", "2021-05-20")
        self.behaviors = BehaviorService(
            self.database, load_config(CONFIG_PATH), Taxonomy()
        )
        self.memories = MemoryService(self.database)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _add_three_listens(self, entity: str) -> None:
        content_id = f"content_{entity}"
        self.database.upsert_content(content_id, f"{entity}故事", [entity])
        for index in range(3):
            self.behaviors.ingest(
                BehaviorEventInput(
                    event_id=f"{entity}_{index}",
                    child_id="child",
                    content_id=content_id,
                    event_type=BehaviorType.COMPLETE_LISTEN,
                    trigger_type=TriggerType.ACTIVE_REQUEST,
                    source_device="蛋宝设备",
                    source_agent="故事智能体",
                )
            )

    def test_behavior_candidate_inherits_latest_source(self) -> None:
        self._add_three_listens("海豚")
        self.behaviors.process_pending()
        memory = self.memories.list_memories("child")[0]
        self.assertEqual("蛋宝设备", memory["source_device"])
        self.assertEqual("故事智能体", memory["source_agent"])

    def test_weekly_limit_is_not_released_after_confirmation(self) -> None:
        for entity in ("海豚", "恐龙", "消防车", "小动物"):
            self._add_three_listens(entity)
        first = self.behaviors.process_pending()
        self.assertEqual(3, first["created_candidates"])
        current = self.memories.list_memories("child")
        self.memories.parent_review(current[0]["memory_id"], "confirm")
        second = self.behaviors.process_pending()
        self.assertEqual(0, second["created_candidates"])
        self.assertEqual(3, len(self.memories.list_memories("child")))


if __name__ == "__main__":
    unittest.main()

