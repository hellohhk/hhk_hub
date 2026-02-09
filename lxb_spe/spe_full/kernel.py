import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .config import DeepSeekConfig


@dataclass
class KernelCallResult:
    content: str
    raw: Optional[Dict[str, Any]] = None


class DeepSeekKernel:
    def __init__(self, config: DeepSeekConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.base_url = f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}

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

        if "choices" in data and data["choices"]:
            content = data["choices"][0]["message"].get("content", "")
            return KernelCallResult(content=content or "", raw=data)
        return KernelCallResult(content="", raw=data)

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
            if "choices" not in chunk_json or not chunk_json["choices"]:
                continue
            delta = chunk_json["choices"][0].get("delta", {})
            if "content" in delta and isinstance(delta["content"], str):
                chunks.append(delta["content"])

        return KernelCallResult(content="".join(chunks), raw=None)
