import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
import requests

# 【修改点】使用新的包路径导入 Config
from new_spe.utils.config_loader import DeepSeekConfig


@dataclass
class KernelCallResult:
    content: str
    raw: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, int]] = None  # 🌟 新增：专门用于储存 Token 消耗数据的字典


class DeepSeekKernel:
    """
    【模型调用内核】
    负责与 LLM API (如 DeepSeek) 进行通信，执行变异算子的重写任务。
    """

    def __init__(self, config: DeepSeekConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.base_url = f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

    def chat(
            self,
            system_msg: str,
            user_msg: str,
            *,
            expect_json: bool,
            stream: bool = True,
            temperature: Optional[float] = None,
            extra: Optional[Dict[str, Any]] = None,
    ) -> KernelCallResult:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if expect_json:
            payload["response_format"] = {"type": "json_object"}
        if stream:
            payload["stream"] = True
            # 🌟 新增：强制要求 OpenAI/DeepSeek 在流式传输的最后返回 Token 统计信息
            payload["stream_options"] = {"include_usage": True}
        if extra:
            payload.update(extra)

        if stream:
            return self._chat_stream(payload)
        return self._chat_once(payload)

    def _chat_once(self, payload: Dict[str, Any]) -> KernelCallResult:
        try:
            resp = requests.post(self.base_url, headers=self._headers(), json=payload, timeout=self.config.timeout_s)
        except Exception as e:
            return KernelCallResult(content="", raw={"error": str(e)})

        try:
            data = resp.json()
        except Exception:
            return KernelCallResult(content="", raw={"status": resp.status_code, "text": resp.text})

        # 🌟 新增：精准拦截 Token 消耗账单
        usage_data = data.get("usage", {})
        if self.verbose and usage_data:
            print(
                f"💰 [API 消耗] 输入: {usage_data.get('prompt_tokens', 0)} | 生成: {usage_data.get('completion_tokens', 0)} | 总计: {usage_data.get('total_tokens', 0)} Tokens")

        if "choices" in data and data["choices"]:
            content = data["choices"][0]["message"].get("content", "")
            return KernelCallResult(content=content or "", raw=data, usage=usage_data)  # 🌟 传入 usage
        return KernelCallResult(content="", raw=data, usage=usage_data)

    def _chat_stream(self, payload: Dict[str, Any]) -> KernelCallResult:
        try:
            resp = requests.post(
                self.base_url,
                headers=self._headers(),
                json=payload,
                timeout=self.config.timeout_s,
                stream=True,
            )
        except Exception as e:
            return KernelCallResult(content="", raw={"error": str(e)})

        if resp.status_code != 200:
            try:
                return KernelCallResult(content="", raw=resp.json())
            except Exception:
                return KernelCallResult(content="", raw={"status": resp.status_code, "text": resp.text})

        chunks: list[str] = []
        usage_data = None  # 🌟 新增：初始化流式账单拦截器

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                decoded = line.decode("utf-8")
            except Exception:
                continue
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk_json = json.loads(data_str)
            except Exception:
                continue

            # 🌟 新增：流式传输的最后一块拼图中，往往藏着 usage 数据，抓出来！
            if "usage" in chunk_json and chunk_json["usage"]:
                usage_data = chunk_json["usage"]

            if "choices" not in chunk_json or not chunk_json["choices"]:
                continue
            delta = chunk_json["choices"][0].get("delta", {})
            if "content" in delta and isinstance(delta["content"], str):
                chunks.append(delta["content"])

        if self.verbose and usage_data:
            print(f"💰 [API 流式消耗] 总计: {usage_data.get('total_tokens', 0)} Tokens")

        return KernelCallResult(content="".join(chunks), raw=None, usage=usage_data)  # 🌟 传入 usage