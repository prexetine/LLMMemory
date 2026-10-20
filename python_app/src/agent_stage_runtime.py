"""阶段检查点与有界调用调度；网络和协议错误共享同一重试预算。"""
from __future__ import annotations

import copy
import json
import logging
import time
from collections import Counter

from agent_data import digest
from agent_context import ContextCapacityError
from agent_llm import TransientModelError
from agent_v3_contracts import ModelOutputError

STAGE_VERSION = "stage-runtime-1"
STAGE_VERSIONS = {name: 1 for name in ("source_router", "source_extractor", "source_reviewer", "source_repair", "source_consistency", "source_classifier", "history_compare")}
STAGE_VERSIONS.update(source_router=3, source_reviewer=3, history_compare=2, source_extractor=3, source_repair=4, source_relations=3, source_classifier=3)


def stage_version(role: str) -> int:
    return STAGE_VERSIONS.get(role.split(":")[0], 1)


def model_signature(config: dict) -> dict:
    """只纳入会改变模型语义输出的设置，不绑定调度或产品目录。"""
    return {k: config.get(k) for k in ("provider", "model_alias", "thinking", "max_output_tokens")}


class StageRuntime:
    def __init__(self, pipeline):
        self.p = pipeline
        # Session异常恢复沿用工作单元预算，避免外层重试把模型三次预算翻倍。
        self.attempts = getattr(pipeline.a, "session_attempt_budget", None)
        if self.attempts is None:
            self.attempts = Counter()
        self.records = {}
        self.invalidations = []

    def invalidate(self, key, role, cached, error):
        """失效记录保留旧产物供追溯；不迁移旧通过结果。"""
        issue = dict(stage=role, session_id=self.p.s["session_id"], code="invalid_cache",
                     reason=str(error), cache_key=key, retryable=True)
        self.invalidations.append(issue)
        self.p.stats[role + ":cache_invalidated"] += 1
        self.p.a.store.save_work(key, self.p.s, dict(status="invalidated", previous=cached, issues=[issue]))
        logging.warning("Session=%s stage=%s 缓存失效，将重新处理：%s", self.p.s["session_id"], role, error)

    def fingerprint(self, role: str, prompt: str, data: dict) -> str:
        return digest([STAGE_VERSION, self.p.s["device_id"], self.p.s["input_hash"],
                       model_signature(self.p.a.config), getattr(self.p.a.client.client, "model", None),
                       stage_version(role), role, prompt, data])

    def load_item(self, role: str, prompt: str, data: dict, validate=None) -> tuple[str, dict | None]:
        key = self.fingerprint(role + ":item", prompt, data)
        cached = self.p.a.store.load_work(key)
        if cached and cached.get("status") == "completed":
            try:
                value = copy.deepcopy(cached["output"])
                if validate is not None:
                    validate(value)
            except (ValueError, TypeError, KeyError) as exc:
                self.invalidate(key, role, cached, exc)
            else:
                self.p.stats[role + ":item_cached"] += 1
                return key, value
        return key, None

    def save_item(self, key: str, role: str, data: dict, output: dict) -> None:
        self.p.a.store.save_work(key, self.p.s, dict(stage=role, stage_version=STAGE_VERSION,
            input_fingerprint=digest(data), dependency_fingerprint=key, status="completed",
            output=output, issues=[], metrics={}))

    def discard_unit_cache(self, unit: str, reason: str) -> None:
        """封套成功但逐项失败时，禁止下一次重试复用同一坏批次。"""
        for key, record in self.records.items():
            if record.get('work_id') == unit:
                self.p.a.store.save_work(key, self.p.s, dict(status='invalidated', reason=reason))

    def ask(self, role, prompt, data, validate, *, attempt_limit=None, validation_label="valid", unit_key=None):
        from agent_contracts import VERSION
        from agent_prompts import PHOTO_NOTICE
        data = dict(data, policy=VERSION)
        if self.p.s.get("source_kind") == "photo" and role not in ("history_compare", "source_classifier", "source_relations"):
            prompt += PHOTO_NOTICE
        key = self.fingerprint(role, prompt, data)
        unit = unit_key or key
        cached = self.p.a.store.load_work(key)
        record = dict(stage=role, work_id=unit, stage_version=stage_version(role), input_fingerprint=digest(data),
                      dependency_fingerprint=key, status="running", issues=[],
                      metrics=dict(attempts=0, calls=0, cache_hits=0, elapsed_seconds=0.0, known_tokens=0, usage_missing=0))
        self.records[key] = record
        if cached and cached.get("status") == "completed":
            try:
                value = copy.deepcopy(cached["output"])
                validate(value)
            except (ValueError, TypeError, KeyError) as exc:
                self.invalidate(key, role, cached, exc)
            else:
                record.update(status="completed", output=value)
                record["metrics"]["cache_hits"] = 1
                self.p.stats[role + ":cached"] += 1
                return value
        maximum = self.p.a.config.get("stage_max_attempts", 3)
        if type(maximum) is not int or not 1 <= maximum <= 3:
            raise ValueError("stage_max_attempts必须为1至3的整数")
        remaining = maximum - self.attempts[unit]
        if attempt_limit is not None:
            remaining = min(remaining, attempt_limit)
        request, failure = data, ModelOutputError("工作单元重试预算耗尽")
        for _ in range(max(0, remaining)):
            self.attempts[unit] += 1
            self.p.stats[role] += 1
            record["metrics"]["attempts"] += 1
            started = time.perf_counter()
            client = self.p.a.client.client
            prior = getattr(client,"last_call_record",None)
            client.execution_context = dict(session_id=self.p.s["session_id"], device_id=self.p.s["device_id"],
                                            work_id=unit, stage_version=stage_version(role), attempt=self.attempts[unit])
            from agent_flow_data import PROCESSING_REVISION
            client.execution_context['processing_revision'] = getattr(self.p,'processing_revision',PROCESSING_REVISION)
            try:
                value = self.p.a.client.ask(role, prompt, request)
                try:
                    validate(value)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ModelOutputError(str(exc)) from exc
            except ContextCapacityError as exc:
                # 容量不是协议重试；交调用方拆分输入，不能重复同一超限请求。
                record["status"] = "capacity_blocked"
                record["issues"] = [dict(code="context_capacity", reason=str(exc), retryable=True)]
                raise
            except (TransientModelError, json.JSONDecodeError, ModelOutputError) as exc:
                failure = exc
                request = dict(data, validation_feedback=str(exc))
                self.p.record_validation(role, self.attempts[unit], "retrying", str(exc))
            else:
                record.update(status="completed", output=value)
                record["metrics"]["elapsed_seconds"] += time.perf_counter() - started
                self.p.record_validation(role, self.attempts[unit], validation_label, "")
                return value
            finally:
                client.execution_context = {}
                call = getattr(client,"last_call_record",None)
                if call is not None and call is not prior:
                    record["metrics"]["calls"] += 1
                    total = (call.get("usage") or {}).get("total_tokens")
                    if type(total) is int:
                        record["metrics"]["known_tokens"] += total
                    else:
                        record["metrics"]["usage_missing"] += 1
                if record["status"] != "completed":
                    record["metrics"]["elapsed_seconds"] += time.perf_counter() - started
                if record["status"] in ("completed", "capacity_blocked"):
                    self.p.a.store.save_work(key, self.p.s, record)
        error = failure if isinstance(failure, (TransientModelError, ModelOutputError)) else ModelOutputError(str(failure))
        code = "transient_service" if isinstance(error, TransientModelError) else "model_protocol"
        record.update(status="retryable", issues=[dict(code=code, reason=str(error), retryable=True)])
        self.p.a.store.save_work(key, self.p.s, record)
        logging.warning("Session=%s stage=%s 本次工作调用未完成 attempts=%s/%s code=%s", self.p.s["session_id"], role, self.attempts[unit], maximum, code)
        raise error
