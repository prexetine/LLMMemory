"""仅允许用户授权的豆包/DeepSeek，真实网络与离线测试分离。"""
from __future__ import annotations

import json
import http.client
import logging
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from agent_schema import response_schema
from agent_context import check_request, ContextCapacityError, input_budget, request_metrics
from agent_v3_contracts import ModelOutputError
try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

LOGGER = logging.getLogger(__name__)


class TransientModelError(RuntimeError):
    """重试耗尽的临时服务异常；可以隔离当前Session，不能吞掉鉴权/配置错误。"""


class JsonClient:
    """通过配置中的兼容chat接口请求JSON，限额、超时并记录真实响应。"""

    def __init__(self, config: dict, credential_path: Path, log_path: Path):
        import yaml  # 现有Python环境已安装；不读取其他模型配置。
        source = yaml.safe_load(credential_path.read_text(encoding="utf-8"))
        provider, alias = config["provider"], config["model_alias"]
        if provider not in ("Doubao", "Deepseek"):
            raise ValueError("只允许Doubao或Deepseek")
        if not alias.lower().startswith(("doubao-", "deepseek-")):
            raise ValueError("不允许调用其他模型")
        section = source[provider]
        base = section["base_url"].rstrip("/")
        parsed = urllib.parse.urlsplit(base)
        allowed = {"Doubao": "ark.cn-beijing.volces.com", "Deepseek": "api.deepseek.com"}
        if parsed.scheme != "https" or parsed.hostname != allowed[provider] or parsed.username or parsed.query:
            raise ValueError("服务地址不符合授权服务白名单")
        self.url = base + "/chat/completions"
        self.key = section["api_key"]
        self.model = section["model_map"][alias]
        self.config, self.log_path = config, log_path
        self.calls, self.input_chars, self.total_tokens = 0, 0, 0
        self.elapsed = 0.0
        # 禁止自动重定向，防止鉴权头被转发给其他服务。
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        self.opener = urllib.request.build_opener(NoRedirect())

    def ask(self, role: str, instruction: str, data: Any) -> dict:
        """对临时网络故障作有界重试，不改变供应商或模型。"""
        retries = 0 if self.config.get("flow_version") in ("generic-flow-v2", "generic-flow-v3-relations") else self.config.get("network_retries", 1)
        if type(retries) is not int or not 0 <= retries <= 10:
            raise ValueError("network_retries必须为0至10的整数")
        for attempt in range(retries + 1):
            try:
                return self._ask_once(role, instruction, data)
            except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, ConnectionError, TransientModelError) as exc:
                if attempt == retries:
                    detail = str(exc) if isinstance(exc, TransientModelError) else type(exc).__name__
                    raise TransientModelError("模型临时请求失败，重试耗尽：" + detail) from exc
                LOGGER.warning("临时网络故障，将重试同一模型 attempt=%s", attempt + 1)
        raise RuntimeError("重试未返回结果")

    def _ask_once(self, role: str, instruction: str, data: Any) -> dict:
        """一次实际调用；失败明确返回异常，不静默切换模型或本地规则。"""
        from agent_parallel import stop_requested
        if stop_requested():
            raise KeyboardInterrupt()
        estimated_tokens = check_request(self.config, role, instruction, data)
        from agent_photos import image_content, options
        images = image_content(self, data)
        if images:
            settings = options(self.config)
            if settings is None:
                raise ValueError("图片请求未启用视觉配置")
            estimated_tokens += settings["input_token_reserve"]
            if estimated_tokens > input_budget(self.config):
                raise ContextCapacityError("含图片预留的请求超出上下文容量")
        user = json.dumps(data, ensure_ascii=False)
        chars = len(instruction) + len(user)
        call_limit = self.config["max_calls"]
        # max_request_chars为旧配置兼容字段，不再限制或截断单次输入。
        if call_limit is not None and self.calls >= call_limit:
            raise RuntimeError("模型调用次数预算不足")
        input_limit = self.config["max_total_input_chars"]
        if input_limit is not None and self.input_chars + chars > input_limit:
            raise RuntimeError("累计输入预算不足")
        self.calls += 1
        self.input_chars += chars
        payload = {"model": self.model, "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": ([{"type": "text", "text": user}] + images) if images else user}],
            "response_format": {"type": "json_object"}}
        if "max_output_tokens" in self.config:
            limit = self.config["max_output_tokens"]
            if type(limit) is not int or limit <= 0:
                raise ValueError("max_output_tokens必须为正整数；不设置上限时删除该字段")
            payload["max_tokens"] = limit
        if self.config["provider"] == "Doubao":
            thinking = self.config.get("thinking", "disabled")
            if thinking not in ("disabled", "enabled", "auto"):
                raise ValueError("非法thinking配置")
            payload["thinking"] = {"type": thinking}
            schema = response_schema(role, data)
            if schema is not None:
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "memory_agent_response", "strict": True, "schema": schema}}
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + self.key,
                                                  "Content-Type": "application/json"})
        started = time.perf_counter()
        record: dict[str, Any] = {"call": self.calls, "role": role, "input_chars": chars,
                                  "estimated_input_tokens": estimated_tokens,
                                  **request_metrics(self.config, role, instruction, data),
                                  "request_id": uuid.uuid4().hex,
                                  "model": self.model, "provider": self.config["provider"]}
        record.update(getattr(self, "execution_context", {}))
        record['estimated_input_tokens'] = estimated_tokens
        record['image_token_reserve'] = settings['input_token_reserve'] if images else 0
        record["image_count"] = sum(x["type"] == "image_url" for x in images)
        LOGGER.info('模型请求 call=%s role=%s input_estimate=%s input_budget=%s request_bytes=%s estimator=%s',
                    self.calls,role,estimated_tokens,input_budget(self.config),record['request_utf8_bytes'],record['token_estimator'])
        try:
            with self.opener.open(request, timeout=self.config["timeout_seconds"]) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body["choices"][0]["message"]
            raw = message.get("content") or ""
            record.update(raw_output=raw, usage=body.get("usage", {}),
                          finish_reason=body["choices"][0].get("finish_reason"))
            self.total_tokens += int(body.get("usage", {}).get("total_tokens", 0))
            if record["finish_reason"] == "length":
                raise ContextCapacityError("模型输出截断，需拆分当前任务后重试")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    value = json.loads(raw, strict=False)
                    record["repair"] = "unescaped_control_characters_only"
                except json.JSONDecodeError:
                    if repair_json is None:
                        raise
                    value = repair_json(raw, return_objects=True)
                    record["repair"] = "optional_json_repair; business_validation_required"
                record["repaired_output"] = json.dumps(value, ensure_ascii=False)
            if not isinstance(value, dict):
                raise ModelOutputError("模型输出必须为JSON对象，不能是数组、null或其他类型")
            record["json_status"] = "valid"
            return value
        except urllib.error.HTTPError as exc:
            record["error"] = f"HTTP {exc.code}"
            # 只做分类，不输出供应商原始错误体，避免包含原文或凭据。
            detail = exc.read(16384).decode("utf-8", errors="replace").lower()
            markers = ("context_length_exceeded", "maximum context length", "context window",
                       "input too long", "input tokens exceed", "prompt is too long", "上下文长度")
            if exc.code in (400, 413) and (exc.code == 413 or any(m in detail for m in markers)):
                record["error"] = "provider_context_capacity"
                raise ContextCapacityError("供应商上下文容量拒绝，需拆分任务") from None
            if exc.code in (408, 429) or 500 <= exc.code < 600:
                raise TransientModelError(f"模型接口HTTP {exc.code}，未切换其他服务") from None
            raise RuntimeError(f"模型接口HTTP {exc.code}，未切换其他服务") from None
        except json.JSONDecodeError as exc:
            record.update(error="invalid_model_json", json_status="invalid")
            raise ModelOutputError("模型返回无法解析为JSON对象") from exc
        except Exception as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self.last_call_record = record
            self.elapsed += record["elapsed_seconds"]
            record["schema_status"] = "由调用方在trace.jsonl记录"
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            LOGGER.info("模型响应 call=%s role=%s status=%s elapsed=%.2fs tokens=%s images=%s",
                        self.calls, role, "failed" if "error" in record else "received",
                        record["elapsed_seconds"], record.get("usage", {}).get("total_tokens", "unknown"), record["image_count"])
