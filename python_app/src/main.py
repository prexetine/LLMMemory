"""ToyMemory 规则核心命令行入口。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from behavior_service import BehaviorService
from config import load_config, resolve_database_path
from memory_service import MemoryService
from models import (
    BehaviorEventInput,
    BehaviorType,
    RecommendationContext,
    TriggerType,
)
from recommendation_service import RecommendationService
from storage import Database
from taxonomy import Taxonomy


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.json"


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="ToyMemory 记忆-内容-推荐规则核心")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, help="覆盖配置中的数据库路径")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="初始化 SQLite 数据库")
    subparsers.add_parser("demo", help="运行三次收听到推荐的完整演示")
    subparsers.add_parser("process", help="处理尚未回流的行为事件")

    list_parser = subparsers.add_parser("list-memories", help="列出孩子的记忆")
    list_parser.add_argument("--child-id", required=True)

    review_parser = subparsers.add_parser("review", help="执行家长审核")
    review_parser.add_argument("--memory-id", required=True)
    review_parser.add_argument(
        "--action", required=True, choices=["confirm", "ignore", "restore", "stop"]
    )

    recommend_parser = subparsers.add_parser("recommend", help="生成推荐候选")
    recommend_parser.add_argument("--child-id", required=True)
    recommend_parser.add_argument("--entities", default="")
    recommend_parser.add_argument("--limit", type=int, default=None)
    recommend_parser.add_argument("--child-initiated-sensitive-topic", action="store_true")
    return parser


def create_services(args: argparse.Namespace) -> tuple[
    Database, MemoryService, BehaviorService, RecommendationService
]:
    """加载配置并创建服务对象。"""

    config = load_config(args.config)
    database_path = args.database or resolve_database_path(
        args.config, str(config["database_path"])
    )
    database = Database(database_path)
    database.initialize()
    taxonomy = Taxonomy()
    return (
        database,
        MemoryService(database),
        BehaviorService(database, config, taxonomy),
        RecommendationService(database, config, taxonomy),
    )


def run_demo(
    database: Database,
    memories: MemoryService,
    behaviors: BehaviorService,
    recommendations: RecommendationService,
) -> dict[str, object]:
    """执行需求文档中的核心现场演示路径。"""

    child_id = "demo_child"
    database.upsert_child(child_id, "小宇", "2021-05-20")
    database.upsert_content(
        "dolphin_001", "会唱歌的小海豚", ["海豚"], genre="故事", min_age=3, max_age=8
    )
    database.upsert_content(
        "dolphin_002", "海豚为什么会跳跃", ["海豚"], genre="知识", min_age=4, max_age=9
    )
    database.upsert_content(
        "dinosaur_001", "霸王龙吃饭记", ["霸王龙"], genre="故事", min_age=3, max_age=8
    )
    for index in range(3):
        behaviors.ingest(
            BehaviorEventInput(
                event_id=f"demo_listen_{index + 1}",
                child_id=child_id,
                content_id="dolphin_001",
                event_type=BehaviorType.COMPLETE_LISTEN,
                trigger_type=TriggerType.ACTIVE_REQUEST,
                source_device="演示设备",
                source_agent="故事智能体",
                interaction_id=f"demo_interaction_{index + 1}",
            )
        )
    process_result = behaviors.process_pending()
    current = memories.list_memories(child_id)
    dolphin = next((item for item in current if item["entity"] == "海豚"), None)
    if dolphin and dolphin["status"] == "pending":
        memories.parent_review(dolphin["memory_id"], "confirm")
    items = recommendations.recommend(
        RecommendationContext(child_id=child_id, conversation_entities=("海豚",))
    )
    return {
        "behavior_result": process_result,
        "memories": memories.list_memories(child_id),
        "recommendations": [asdict(item) for item in items],
    }


def main() -> int:
    """解析命令并输出 UTF-8 JSON 结果。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    database, memories, behaviors, recommendations = create_services(args)
    result: object
    if args.command == "init-db":
        result = {"database": str(database.path), "initialized": True}
    elif args.command == "demo":
        result = run_demo(database, memories, behaviors, recommendations)
    elif args.command == "process":
        result = behaviors.process_pending()
    elif args.command == "list-memories":
        result = memories.list_memories(args.child_id)
    elif args.command == "review":
        result = {
            "memory_id": args.memory_id,
            "status": memories.parent_review(args.memory_id, args.action),
        }
    elif args.command == "recommend":
        entities = tuple(value.strip() for value in args.entities.split(",") if value.strip())
        items = recommendations.recommend(
            RecommendationContext(
                child_id=args.child_id,
                conversation_entities=entities,
                child_initiated_sensitive_topic=args.child_initiated_sensitive_topic,
                limit=args.limit,
            )
        )
        result = [asdict(item) for item in items]
    else:
        raise RuntimeError(f"未处理的命令: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

