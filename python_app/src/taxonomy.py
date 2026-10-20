"""实体同义归一与层级展开。"""

from __future__ import annotations


DEFAULT_SYNONYMS: dict[str, str] = {
    "大恐龙": "恐龙",
    "dinosaur": "恐龙",
    "霸王龙": "霸王龙",
    "t-rex": "霸王龙",
    "海豚鱼": "海豚",
}

DEFAULT_PARENTS: dict[str, str] = {
    "霸王龙": "恐龙",
    "三角龙": "恐龙",
    "腕龙": "恐龙",
    "消防车": "车辆",
    "救护车": "车辆",
}


class Taxonomy:
    """轻量实体词表，可由内容平台全量词表替换。"""

    def __init__(
        self,
        synonyms: dict[str, str] | None = None,
        parents: dict[str, str] | None = None,
    ) -> None:
        self._synonyms = {**DEFAULT_SYNONYMS, **(synonyms or {})}
        self._parents = {**DEFAULT_PARENTS, **(parents or {})}

    @property
    def known_entities(self) -> set[str]:
        """返回当前词表已登记的实体，供未知标签运营提示使用。"""

        return (
            set(self._synonyms)
            | set(self._synonyms.values())
            | set(self._parents)
            | set(self._parents.values())
        )

    def normalize(self, entity: str) -> str:
        """清理空白并执行同义归一。"""

        normalized = entity.strip().lower()
        if not normalized:
            raise ValueError("实体不能为空")
        return self._synonyms.get(normalized, normalized)

    def ancestors(self, entity: str) -> set[str]:
        """返回实体本身及其全部上位实体，并防止循环词表。"""

        current = self.normalize(entity)
        result: set[str] = set()
        while current not in result:
            result.add(current)
            parent = self._parents.get(current)
            if not parent:
                break
            current = self.normalize(parent)
        return result

    def matches(self, memory_entity: str, content_entity: str) -> bool:
        """判断内容实体是否等于记忆实体或属于其下位实体。"""

        memory = self.normalize(memory_entity)
        return memory in self.ancestors(content_entity)

