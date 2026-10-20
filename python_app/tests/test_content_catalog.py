"""内容混合标签拆分测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from content_catalog import ContentCatalogService  # noqa: E402
from storage import Database  # noqa: E402
from taxonomy import Taxonomy  # noqa: E402


class ContentCatalogTests(unittest.TestCase):
    def test_mixed_tags_are_split_and_unknown_tags_fall_back_to_entity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "catalog.sqlite3")
            database.initialize()
            service = ContentCatalogService(database, Taxonomy())
            result = service.upsert_from_mixed_tags(
                "content_1",
                "恐龙安全绘本",
                ["恐龙", "绘本", "安全自护", "中文", "火山岛"],
                min_age=4,
                max_age=8,
            )
            self.assertEqual(["恐龙", "火山岛"], result["entities"])
            self.assertEqual(["故事类"], result["genres"])
            self.assertEqual(["安全自护"], result["ability_dimensions"])
            self.assertEqual(["中文"], result["metadata_tags"])
            self.assertEqual(["火山岛"], result["fallback_entities"])
            with database.connect() as connection:
                content = connection.execute(
                    "SELECT * FROM content_items WHERE content_id='content_1'"
                ).fetchone()
                self.assertEqual("故事类", content["genre"])
                self.assertEqual("安全自护", content["ability_dimension"])

    def test_tags_require_at_least_one_entity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "catalog.sqlite3")
            database.initialize()
            service = ContentCatalogService(database, Taxonomy())
            with self.assertRaises(ValueError):
                service.classify_tags(["绘本", "中文", "安全自护"])


if __name__ == "__main__":
    unittest.main()

