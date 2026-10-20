"""内容混合标签拆分与本地快照适配。"""

from __future__ import annotations

from typing import Any

from storage import Database
from taxonomy import Taxonomy


GENRE_MAP: dict[str, str] = {
    "故事": "故事类",
    "绘本": "故事类",
    "冒险": "故事类",
    "科普": "知识类",
    "知识": "知识类",
    "儿歌": "音乐类",
    "音乐": "音乐类",
}

ABILITY_TAGS: set[str] = {
    "安全自护",
    "社交能力",
    "语言能力",
    "情绪管理",
    "表达交流",
    "社会适应",
    "科学探究",
    "数学认知",
    "表现与创造",
    "人际交往",
    "感受与欣赏",
}

METADATA_TAGS: set[str] = {"中文", "英文", "中国", "欧美"}


class ContentCatalogService:
    """把内容平台混合标签转换为统一标签语言。"""

    def __init__(self, database: Database, taxonomy: Taxonomy) -> None:
        self._database = database
        self._taxonomy = taxonomy

    def classify_tags(self, tags: list[str]) -> dict[str, list[str]]:
        """拆分实体、能力、体裁和元数据，未知标签按实体兜底。"""

        if not tags:
            raise ValueError("内容标签不能为空")
        entities: set[str] = set()
        genres: set[str] = set()
        abilities: set[str] = set()
        metadata: set[str] = set()
        fallback_entities: set[str] = set()
        for raw_tag in tags:
            tag = raw_tag.strip()
            if not tag:
                raise ValueError("内容标签不能包含空字符串")
            if tag in GENRE_MAP:
                genres.add(GENRE_MAP[tag])
            elif tag in ABILITY_TAGS:
                abilities.add(tag)
            elif tag in METADATA_TAGS:
                metadata.add(tag)
            else:
                normalized = self._taxonomy.normalize(tag)
                entities.add(normalized)
                if tag not in self._taxonomy.known_entities:
                    fallback_entities.add(normalized)
        if not entities:
            raise ValueError("混合标签中至少需要一个实体标签")
        return {
            "entities": sorted(entities),
            "genres": sorted(genres),
            "ability_dimensions": sorted(abilities),
            "metadata_tags": sorted(metadata),
            "fallback_entities": sorted(fallback_entities),
        }

    def upsert_from_mixed_tags(
        self,
        content_id: str,
        title: str,
        tags: list[str],
        *,
        min_age: int | None = None,
        max_age: int | None = None,
        status: str = "active",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        """拆分标签并写入内容快照，返回分类结果供运营检查。"""

        classified = self.classify_tags(tags)
        stored_metadata = dict(metadata or {})
        stored_metadata["raw_tags"] = tags
        stored_metadata["fallback_entities"] = classified["fallback_entities"]
        self._database.upsert_content(
            content_id,
            title,
            classified["entities"],
            genre=",".join(classified["genres"]),
            ability_dimension=",".join(classified["ability_dimensions"]),
            language_region=",".join(classified["metadata_tags"]),
            min_age=min_age,
            max_age=max_age,
            status=status,
            metadata=stored_metadata,
        )
        return classified

