import os
import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.6
    timeout_s: int = 120


_RE_KV = re.compile(r"^\s*(?P<key>[A-Za-z_]+)\s*=\s*(?P<val>.+?)\s*$")
_RE_QUOTED = re.compile(r'^[\'"](?P<val>.*)[\'"]$')


def _strip_quotes(s: str) -> str:
    s = s.strip()
    m = _RE_QUOTED.match(s)
    if m:
        return m.group("val")
    if s.startswith("“") and s.endswith("”"):
        return s[1:-1]
    return s


def _parse_apikey_file(apikey_path: str) -> Optional[DeepSeekConfig]:
    if not os.path.exists(apikey_path):
        return None

    in_deepseek_block = False
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    with open(apikey_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.lower().startswith("model_one") and "deepseek" in line.lower():
                in_deepseek_block = True
                continue
            if line.lower().startswith("model_") and "deepseek" not in line.lower():
                if in_deepseek_block:
                    break

            if not in_deepseek_block:
                continue

            m = _RE_KV.match(line)
            if not m:
                continue
            k = m.group("key").strip().lower()
            v = _strip_quotes(m.group("val"))
            if k == "api_key" and v:
                api_key = v
            if k == "base_url" and v:
                base_url = v

    if api_key:
        return DeepSeekConfig(api_key=api_key, base_url=base_url or "https://api.deepseek.com")
    return None


def load_deepseek_config(apikey_path: str) -> DeepSeekConfig:
    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key:
        return DeepSeekConfig(
            api_key=env_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.6")),
            timeout_s=int(os.environ.get("DEEPSEEK_TIMEOUT_S", "120")),
        )

    cfg = _parse_apikey_file(apikey_path)
    if cfg:
        return cfg

    raise RuntimeError(
        "未找到 DeepSeek API Key。请设置环境变量 DEEPSEEK_API_KEY，或在 apikey.txt 的 model_one: deepseek 段落提供 api_key。"
    )

