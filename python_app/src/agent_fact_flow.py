"""fact-driven-v1：事实先行，维度仅作为消费投影。"""
from __future__ import annotations

import copy
import logging
import json
from pathlib import Path
from collections import Counter
from typing import Any

from agent_context import ContextCapacityError, windows
from agent_data import digest
from agent_fact_model import RELATIONS, content_key, deterministic_relation, fact_text
from agent_fact_policy import deduplicate_candidates, latest_candidates
from agent_fact_store import FactStore
from agent_flow_data import materialize_references
from agent_generic_flow import GenericPipeline
from agent_llm import TransientModelError
from agent_retrieval import RetrievalError, Retriever
from agent_data import SKILL_NAMES, exact_keys
from agent_prompts import ROUTER
from agent_session_control import PROMPT as CONTROL_PROMPT, controls, validate_controls


RELATION_PROMPT = (Path(__file__).resolve().parents[1] / "prompts/fact_relations_v2.txt").read_text(encoding="utf-8")


class FactDrivenPipeline(GenericPipeline):
    """复用已验证的分窗、路由、提取与来源审核；不调用旧维护、分类或维度门禁。"""

    FLOW = "fact-driven-v1"

    def __init__(self, agent: Any, session: dict[str, Any]) -> None:
        super().__init__(agent, session)
        self.flow = self.FLOW
        self.fact_store = FactStore(agent.store.db)
        self.fact_store.ensure_protocol()
        self.processing_revision = "session-reliable-2026-10-20-2"
        self.fact_tasks: list[dict[str, Any]] = []
        self.retry_tasks = self.fact_tasks

    def task(self, stage, ids, reason, code=None, evidence_ids=()):
        self._task(stage, ids, reason, code or "model_protocol")

    def _control_window(self, window):
        self.current_window_id = digest(window)[:20]
        signals = []
        if window.get("source_kind") != "photo":
            try:
                raw = self.ask("source_control", CONTROL_PROMPT,
                               {"session": dict(window, turns=[dict(t, assistant_text="") for t in window["turns"]])},
                               lambda value: validate_controls(value, window))
                signals = controls(raw, window)
            except (ValueError, TypeError, KeyError, TransientModelError, ContextCapacityError) as exc:
                self._task("source_control", [], str(exc))
        return signals

    def extract_window(self, window, depth=0):
        """路由失败后继续提取；使用完整Session的控制线索。"""
        self.current_window_id = digest(window)[:20]
        signals = getattr(self, "session_signals", None)
        if signals is None:
            signals = self._control_window(window)
        def validate_route(value):
            exact_keys(value, {"skills", "reason", "signals"})
            if not isinstance(value["skills"], list) or any(s not in SKILL_NAMES for s in value["skills"]):
                raise ValueError("未知Skill")
            if not isinstance(value["reason"], str) or len(set(value["skills"])) != len(value["skills"]):
                raise ValueError("路由字段非法")
        try:
            route = self.ask("source_router", ROUTER, {"session": window,
                             "catalog": [{"skill_id": k, "name": v} for k,v in SKILL_NAMES.items()]}, validate_route)
            skills = route["skills"]
        except (ValueError, TypeError, KeyError, TransientModelError, ContextCapacityError) as exc:
            skills = list(SKILL_NAMES)
            if window.get("source_kind") == "photo":
                skills = ["events_plans", "product_interaction"]
            self.stats["router_fallback"] += 1
            logging.warning("Session=%s 路由失败，逐Skill兜底：%s", self.s["session_id"], exc)
        self.shared = []
        found = []
        for skill in skills:
            try:
                self.shared = []  # 独立Skill不依赖前一个Skill的成功或失败产物。
                rows = self.extract_skill_window(window, skill, signals)
                found.extend(rows)
                self.shared.extend(rows)
            except (ValueError, TypeError, KeyError, TransientModelError, ContextCapacityError) as exc:
                self._task("source_extractor:" + skill, [], str(exc))
        return found, signals

    def _task(self, stage: str, candidate_ids: list[str], reason: str, code: str = "model_protocol", unit_id=None) -> None:
        row = {"task_id": digest([self.s["session_id"], stage, sorted(candidate_ids), unit_id or getattr(self,"current_window_id",None)])[:32],
               "stage": stage, "candidate_ids": sorted(candidate_ids), "state": "technical_retry",
               "reason": str(reason), "code": code, "session_id":self.s["session_id"],
               "window_id":getattr(self,"current_window_id",None), "input_fingerprint": self.s["input_hash"],
               "dependency_fingerprint":digest(sorted(candidate_ids)),
               "work_records":[dict(record) for record in self.runtime.records.values() if record["stage"] == stage]}
        if not any(item["task_id"] == row["task_id"] for item in self.fact_tasks):
            self.fact_tasks.append(row)
            with self.a.store.db:
                self.a.store.db.execute("INSERT OR REPLACE INTO fact_tasks VALUES(?,?,?,?,?,?,?)",
                    (row["task_id"],self.s["device_id"],self.s["session_id"],stage,
                     "technical_retry",str(reason),json.dumps(row,ensure_ascii=False)))

    def _collect(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        candidates: dict[str, dict[str, Any]] = {}
        signals: dict[str, dict[str, Any]] = {}
        count = 0
        char_budget = self.a.config.get("session_chunk_chars")
        from agent_context import setting
        window_list = list(windows(self.s, char_budget if char_budget is not None else setting(self.a.config, "session_chunk_tokens"),
                                  unit="chars" if char_budget is not None else "bytes"))
        self.session_signals = [signal for window in window_list for signal in self._control_window(window)]
        for window in window_list:
            rows, window_signals = self.extract_window(window)
            count += 1
            for candidate in rows:
                candidate["session_id"] = self.s["session_id"]
                candidate["origin_session_id"] = self.s["session_id"]
                candidates[candidate["candidate_id"]] = candidate
            self.fact_store.save_candidates(self.s, rows)
            for signal in window_signals:
                signals[digest(signal)] = signal
        return list(candidates.values()), list(signals.values()), count

    def _review(self, candidates: list[dict[str, Any]], signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        proof_rows: list[dict[str, Any]] = []
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for candidate in candidates:
            grouped.setdefault(tuple(candidate["source_windows"]), []).append(candidate)
        limit = self.a.config.get("source_review_batch_size", 8)
        for group in grouped.values():
            for start in range(0, len(group), limit):
                batch = group[start:start + limit]
                related = signals
                proof_rows.extend(self.review(batch, related))
        return proof_rows

    def _recall(self, active: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        if not active:
            return {}
        all_history: dict[str, dict[str, Any]] = {}
        if not hasattr(self.a, "fact_history_cache"):
            self.a.fact_history_cache = {}
        for device, subject in sorted({(c["device_id"], c["subject_key"]) for c in active}):
            for item in self.fact_store.history_incremental(device, subject, self.a.fact_history_cache):
                all_history[item["fact_id"]] = dict(item, memory_id=item["fact_id"],
                    revision=item["target_version"], state="conflicted" if item["state"] == "conflicted" else "reviewed")
        groups = {}
        for row in self.a.store.db.execute("SELECT relation_id,left_fact_id,right_fact_id FROM fact_relations WHERE relation='contradictory' AND state='active'"):
            members = [self.fact_store.canonical(row[k]) for k in ("left_fact_id", "right_fact_id")]
            groups[row["relation_id"]] = {"status": "open", "members": [{"memory_id": mid} for mid in members]}
        state = {"memories": all_history, "groups": groups}
        try:
            retriever = getattr(self.a, "fact_retriever", None)
            if retriever is None or retriever.db is not self.a.store.db:
                retriever = Retriever(self.a.store.db, self.a.config["retrieval"], self.stats)
                self.a.fact_retriever = retriever
            recalled = retriever.recall(active, state)
            from agent_retrieval import compatible, structured
            from agent_flow_data import source_pairs
            _, vectors = retriever.vectors(active)
            self.batch_pairs = set()
            for i, left in enumerate(active):
                pool = [(sum(a*b for a,b in zip(vectors[i], vectors[j])), j)
                        for j, right in enumerate(active) if i != j and compatible(left, right)]
                pool.sort(key=lambda x: (-x[0], active[x[1]]["candidate_id"]))
                chosen = {j for _, j in pool[:self.a.config["retrieval"].get("batch_top_k", 8)]}
                chosen.update(j for _, j in pool if structured(left, active[j]))
                self.batch_pairs.update(tuple(sorted((left["candidate_id"], active[j]["candidate_id"]))) for j in chosen)
            self.batch_pairs.update(tuple(sorted((l["candidate_id"], r["candidate_id"]))) for l,r in source_pairs(active))
            return {candidate_id: [dict(row, fact_id=row["memory_id"]) for row in rows] for candidate_id, rows in recalled.items()}
        except RetrievalError as exc:
            self._task("semantic_recall", [row["candidate_id"] for row in active], str(exc), getattr(exc, "code", "retrieval_unavailable"))
            return {row["candidate_id"]: [] for row in active}

    @staticmethod
    def _pair_id(left: str, right: str) -> str:
        return digest(["fact-relation-v1", left, right])[:32]

    def _relation_tasks(self, active: list[dict[str, Any]], recalled: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], dict[str, dict[str, Any]], dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
        pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        immediate: dict[str, dict[str, Any]] = {}
        # pair_index 同时保存程序直接判定和模型判定的配对，保证直接等价
        # 关系也参与维护决策及关系凭证落库，而不是被静默跳过。
        pair_index: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for candidate in active:
            candidate["content_key"] = content_key(candidate)
            for history in recalled.get(candidate["candidate_id"], []):
                key = self._pair_id("c:" + candidate["candidate_id"], "h:" + history["fact_id"])
                pair_index[key] = (candidate, history)
                relation = deterministic_relation(candidate, history)
                if relation:
                    immediate[key] = {"pair_id": key, "relation": relation, "reason": "结构化事实字段可直接证明", "basis_refs": []}
                else:
                    pairs[key] = (candidate, history)
        for index, left in enumerate(active):
            for right in active[:index]:
                if left.get("subject_key") != right.get("subject_key") or not left.get("subject_key"):
                    continue
                if tuple(sorted((left["candidate_id"], right["candidate_id"]))) not in getattr(self, "batch_pairs", set()):
                    continue
                key = self._pair_id("c:" + left["candidate_id"], "c:" + right["candidate_id"])
                pair_index[key] = (left, right)
                relation = deterministic_relation(left, right)
                if relation:
                    immediate[key] = {"pair_id": key, "relation": relation, "reason": "同批结构化字段可直接证明", "basis_refs": []}
                else:
                    pairs[key] = (left, right)
        return pairs, immediate, pair_index

    def _judge_relations(self, pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]], signals: list[dict[str, Any]], batch_limit=None) -> dict[str, dict[str, Any]]:
        from agent_fact_relations import judge
        self.stats["relation_tasks"] += len(pairs)
        if len(pairs) >= self.a.config.get("relation_warn_tasks", 2000):
            logging.warning("Session=%s 关系任务量预警：%s", self.s["session_id"], len(pairs))
        return judge(self, pairs, signals, RELATION_PROMPT)

    @staticmethod
    def _validate_relation_envelope(raw: dict[str, Any], values: list[tuple[str, tuple[dict[str, Any], dict[str, Any]]]]) -> None:
        if set(raw) != {"results"} or not isinstance(raw["results"], list):
            raise ValueError("事实关系返回封套非法")
        expected = {pair_id for pair_id, _ in values}
        found = {row.get("pair_id") for row in raw["results"] if isinstance(row, dict)}
        if found != expected or len(raw["results"]) != len(expected):
            raise ValueError("事实关系结果缺失、重复或额外ID")
        for row in raw["results"]:
            if set(row) != {"pair_id", "relation", "reason", "basis_refs", "direction"} or not isinstance(row["reason"], str) or not isinstance(row["basis_refs"], list):
                raise ValueError("事实关系字段非法")
            if row["relation"] not in RELATIONS:
                raise ValueError("事实关系类型非法")
            if row["direction"] not in ("none", "left_replaces_right", "right_replaces_left"):
                raise ValueError("更新方向非法")
            if (row["relation"] == "changed_with_direction") != (row["direction"] != "none"):
                raise ValueError("关系与更新方向不一致")

    def _decisions(self, candidates: list[dict[str, Any]], proofs: list[dict[str, Any]], relations: dict[str, dict[str, Any]], pair_index: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        proof_by_id = {row["candidate_id"]: row for row in proofs}
        decisions: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            blocked = [t for t in self.fact_tasks if candidate["candidate_id"] in t.get("candidate_ids", [])
                       or not t.get("candidate_ids") and t["stage"] in ("source_control", "semantic_recall", "semantic_relation", "source_reviewer")]
            missing = [p for p, ends in pair_index.items() if candidate in ends and p not in relations]
            if blocked or missing:
                decisions[candidate["candidate_id"]] = {"action": "provisional", "reason": "技术工作未完成，禁止提升有效记忆",
                                                        "technical_pending": True}
                continue
            proof = proof_by_id.get(candidate["candidate_id"], {})
            if proof.get("decision") != "pass":
                decisions[candidate["candidate_id"]] = {"action": "provisional", "reason": proof.get("reason", "来源未通过")}
                continue
            if not candidate.get("subject_key"):
                decisions[candidate["candidate_id"]] = {"action": "provisional", "reason": "主体尚未确定"}
                continue
            linked = [(pair_id, row) for pair_id, row in relations.items() if pair_id in pair_index and candidate in pair_index[pair_id]]
            same_source_conflict = any(row["relation"] == "contradictory" and
                {e["evidence_id"] for e in pair_index[pid][0].get("evidence", [])} &
                {e["evidence_id"] for e in pair_index[pid][1].get("evidence", [])}
                for pid,row in linked)
            if same_source_conflict:
                decisions[candidate["candidate_id"]] = {"action": "provisional", "reason": "同一来源被提取成矛盾结论，需要整理，不作为真实历史冲突保存"}
            elif any(row["relation"] == "insufficient_evidence" for _, row in linked):
                decisions[candidate["candidate_id"]] = {"action": "provisional", "reason": "关系不足以安全归并"}
            elif any(row["relation"] == "contradictory" for _, row in linked):
                decisions[candidate["candidate_id"]] = {"action": "conflict", "reason": "存在同条件互斥事实"}
            elif any(row["relation"] == "equivalent" for _, row in linked):
                decisions[candidate["candidate_id"]] = {"action": "reinforce", "reason": "与召回事实等价"}
            elif any(row["relation"] == "changed_with_direction" for _, row in linked):
                decisions[candidate["candidate_id"]] = {"action": "supersede", "reason": "当前会话存在明确变化依据"}
            else:
                decisions[candidate["candidate_id"]] = {"action": "new", "reason": "没有等价、更新或冲突关系"}
        return decisions

    def _run_core(self) -> dict[str, Any]:
        if self.fact_store.session_completed(self.s):
            return self.fact_store.session_result(self.s)
        self.fact_store.save_receipt(self.s)
        candidates, signals, window_count = self._collect()
        before = len(candidates)
        candidates = deduplicate_candidates(candidates)
        self.stats["session_exact_duplicates"] += before - len(candidates)
        done = self.fact_store.completed_candidates(self.s)
        candidates = [c for c in candidates if c["candidate_id"] not in done]
        proofs = self._review(candidates, signals)
        candidates, proofs = self.repair_once(candidates, proofs, signals, self.a.config.get("source_review_batch_size", 8))
        candidates = latest_candidates(candidates)
        self.fact_store.reconcile_candidates(self.s, candidates)
        valid_ids = {candidate["candidate_id"] for candidate in candidates}
        proofs = [proof for proof in proofs if proof["candidate_id"] in valid_ids]
        active = [candidate for candidate in candidates if next((proof for proof in proofs if proof["candidate_id"] == candidate["candidate_id"]), {}).get("decision") == "pass" and candidate.get("subject_key")]
        recalled = self._recall(active)
        pairs, relations, pair_index = self._relation_tasks(active, recalled)
        relations.update(self._judge_relations(pairs, signals))
        for issue in self.extraction_issues:
            self._task("source_extractor:" + issue.get("skill", ""), [], issue["reason"], issue.get("code", "model_protocol"))
        decisions = self._decisions(candidates, proofs, relations, pair_index)
        from agent_fact_relations import unsafe_equivalence
        for cid in unsafe_equivalence(pair_index, relations):
            decisions[cid] = {"action": "provisional", "reason": "等价组缺少代表项直接验证，保留语义待确认"}
        from agent_maintenance import storage_control
        for candidate in candidates:
            control = storage_control(candidate, signals)
            if control["status"] != "allowed":
                decisions[candidate["candidate_id"]] = {"action": "provisional", "reason": control["reason"],
                                                        "storage_control": control}
            elif candidate.get("retention") == "discard":
                decisions[candidate["candidate_id"]] = {"action": "provisional", "reason": "无新增记忆价值", "discard": True}
        relation_records = []
        for pair_id, relation in relations.items():
            left, right = pair_index.get(pair_id, (None, None))
            if left is None:
                continue
            if any(decisions.get(end.get("candidate_id"), {}).get("action") == "provisional"
                   for end in (left, right)):
                continue
            relation_records.append({
                "left_candidate_id": left["candidate_id"] if "candidate_id" in left else "",
                "right_candidate_id": right["candidate_id"] if "candidate_id" in right else "",
                "left_fact_id": left.get("fact_id", ""), "right_fact_id": right.get("fact_id", ""),
                "left_version": left.get("target_version"), "right_version": right.get("target_version"),
                "relation": relation["relation"], "reason": relation["reason"], "pair_id": pair_id,
                "direction": relation.get("direction", "none"), "basis": relation.get("basis", []),
            })
        mapping_tasks = []
        if self.a.config.get("dimension_mapping") == "fact-dimensions-106-v1":
            mapping_tasks = [dict(task_id=digest(["dimension_mapping", self.s["input_hash"]]),
                stage="dimension_mapping", candidate_ids=[], state="technical_retry",
                reason="事实已提交，106维投影尚未收口", code="projection_incomplete",
                session_id=self.s["session_id"])]
        receipt = self.fact_store.commit_groups(self.s, candidates, proofs, decisions, signals, self.fact_tasks,
                                               relation_records, final_tasks=mapping_tasks)
        reviews = [dict(proof, source_decision=proof["decision"], candidate=next(c for c in candidates if c["candidate_id"] == proof["candidate_id"])) for proof in proofs]
        approved = [dict(candidate, operation=decisions[candidate["candidate_id"]]["action"]) for candidate in candidates if decisions[candidate["candidate_id"]]["action"] in {"new", "reinforce", "supersede", "conflict"}]
        return {"session_id": self.s["session_id"], "turns": len(self.s["turns"]), "status": "completed", "policy": "fact-driven-v1",
                "candidates": candidates, "approved": approved, "reviews": reviews,
                "maintenance": [dict(candidate_id=key, **value) for key, value in decisions.items()], "signals": signals,
                "commit": receipt, "window_count": window_count, "batch_metrics": dict(self.stats),
                "processing_revision": self.processing_revision, "flow_version": self.flow,
                "retry_tasks": self.fact_tasks, "needs_attention": bool(self.fact_tasks),
                "data_completeness": self.s.get("source_kind", "dialogue") + "_only"}

    def run(self) -> dict[str, Any]:
        result = self._run_core()
        if self.a.config.get("dimension_mapping") == "fact-dimensions-106-v1":
            from agent_dimension_mapping import project_session
            project_session(self)
            result["retry_tasks"] = self.fact_tasks
            result["batch_metrics"] = dict(self.stats)
        stored = self.fact_store.session_result(self.s)
        for field in ("conclusion", "session_summary", "approved", "candidates", "reviews", "maintenance"):
            result[field] = stored[field]
        result["execution_status"] = "completed" if not result.get("retry_tasks") else "completed_with_retry_tasks"
        result["commit_status"] = "committed" if result.get("commit", {}).get("status") in {"committed", "already_committed"} else "not_committed"
        return result
