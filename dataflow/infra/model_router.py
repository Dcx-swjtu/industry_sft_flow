"""Small model router for OpenAI-compatible chat completion endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelProfile:
    name: str
    api_key_env: str
    base_url_env: str
    model_env: str
    base_url: str
    model: str
    stream: bool = False
    allow_nonstream_fallback: bool = True
    timeout: int = 300
    temperature: float = 0.1
    max_tokens: int = 4096
    enable_thinking: bool = False
    reasoning_effort: str = ""
    api_key: str = ""


@dataclass
class ModelRuntime:
    profile: ModelProfile
    stream: bool
    allow_nonstream_fallback: bool
    temperature: float
    max_tokens: int
    timeout: int
    enable_thinking: bool = False
    reasoning_effort: str = ""


class ModelRouter:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def _profile(self, name: str) -> ModelProfile:
        raw = dict((self.config.get("models") or {}).get(name) or {})
        api_key_env = str(raw.get("api_key_env") or "LLM_API_KEY")
        api_key = os.getenv(api_key_env) or ""
        if not api_key and api_key_env == "JUDGE_API_KEY":
            api_key = os.getenv("LLM_API_KEY") or ""
        return ModelProfile(
            name=name,
            api_key_env=api_key_env,
            base_url_env=str(raw.get("base_url_env") or ""),
            model_env=str(raw.get("model_env") or ""),
            base_url=str(raw.get("base_url") or "").rstrip("/"),
            model=str(raw.get("model") or ""),
            stream=bool(raw.get("stream", False)),
            allow_nonstream_fallback=bool(raw.get("allow_nonstream_fallback", True)),
            timeout=int(raw.get("timeout") or 300),
            temperature=float(raw.get("temperature", 0.1)),
            max_tokens=int(raw.get("max_tokens") or 4096),
            enable_thinking=bool(raw.get("enable_thinking", False)),
            reasoning_effort=str(raw.get("reasoning_effort") or ""),
            api_key=api_key,
        )

    def resolve(self, operator_name: str) -> ModelRuntime:
        operator_cfg = dict((self.config.get("operators") or {}).get(operator_name) or {})
        profile_name = str(operator_cfg.get("model_profile") or "strong_vision")
        profile = self._profile(profile_name)
        return ModelRuntime(
            profile=profile,
            stream=bool(operator_cfg.get("stream", profile.stream)),
            allow_nonstream_fallback=bool(
                operator_cfg.get("allow_nonstream_fallback", profile.allow_nonstream_fallback)
            ),
            temperature=float(operator_cfg.get("temperature", profile.temperature)),
            max_tokens=int(operator_cfg.get("max_tokens", profile.max_tokens)),
            timeout=int(operator_cfg.get("timeout", profile.timeout)),
            enable_thinking=bool(operator_cfg.get("enable_thinking", profile.enable_thinking)),
            reasoning_effort=str(operator_cfg.get("reasoning_effort", profile.reasoning_effort) or ""),
        )
