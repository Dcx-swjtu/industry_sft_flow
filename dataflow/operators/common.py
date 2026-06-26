"""Common model invocation helper for ScienceFlow operators."""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_COMPLETION_URL_CACHE: dict[str, str] = {}
_COMPLETION_URL_LOCK = threading.Lock()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _snapshot_mode(value: Any) -> str:
    if isinstance(value, bool):
        return "all" if value else "none"
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on", "all", "full"}:
        return "all"
    if text in {"", "0", "false", "no", "off", "none", "disabled"}:
        return "none"
    if text in {"failure", "failures", "failed", "error", "errors"}:
        return "failure"
    raise ValueError(f"unsupported prompt_snapshot mode: {value!r}")


def _write_prompt_snapshot(
    *,
    prompt_dir: Path,
    prompt_name: str,
    prompt: str,
    payload: dict[str, Any],
    raw_response: dict[str, Any] | None = None,
    response_text: str | None = None,
    parsed: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / f"{prompt_name}_prompt.md").write_text(prompt, encoding="utf-8")
    _write_json(prompt_dir / f"{prompt_name}_payload.json", payload)
    if raw_response is not None:
        _write_json(prompt_dir / f"{prompt_name}_raw_response.json", raw_response)
    if response_text is not None:
        (prompt_dir / f"{prompt_name}_response.txt").write_text(response_text, encoding="utf-8")
    if parsed is not None:
        _write_json(prompt_dir / f"{prompt_name}_parsed.json", parsed)
    if error is not None:
        _write_json(prompt_dir / f"{prompt_name}_error.json", {
            "exception_type": type(error).__name__,
            "message": str(error)[:1000],
        })


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    candidates = _FENCED_JSON_RE.findall(text) + [text]
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("model response did not contain a JSON object")


def _completion_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return ""


def _delta_content(delta: dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: int) -> dict[str, Any]:
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=raw, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("chat completion response must be a JSON object")
    return data


def _post_stream_text(url: str, headers: dict[str, str], body: dict[str, Any], timeout: int) -> str:
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=raw, headers=headers, method="POST")
    chunks: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict):
                chunks.append(_delta_content(delta))
    return "".join(chunks)


def _completion_urls(base_url: str) -> list[str]:
    base = base_url.rstrip("/")
    urls = [base + "/chat/completions"]
    if not base.endswith("/v1"):
        urls.append(base + "/v1/chat/completions")
    with _COMPLETION_URL_LOCK:
        cached = _COMPLETION_URL_CACHE.get(base)
    if cached and cached in urls:
        return [cached] + [url for url in urls if url != cached]
    return urls


def _remember_completion_url(base_url: str, url: str) -> None:
    base = base_url.rstrip("/")
    with _COMPLETION_URL_LOCK:
        _COMPLETION_URL_CACHE[base] = url


def _clear_completion_url_cache() -> None:
    with _COMPLETION_URL_LOCK:
        _COMPLETION_URL_CACHE.clear()


def invoke_prompt(
    *,
    template,
    payload: dict[str, Any],
    image_base64: str | None,
    operator_name: str,
    prompt_name: str,
    prompt_dir: Path,
    router,
    prompt_snapshot_enabled: bool,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> dict[str, Any]:
    prompt = template.render(payload)
    snapshot_mode = _snapshot_mode(prompt_snapshot_enabled)
    if snapshot_mode == "all":
        _write_prompt_snapshot(
            prompt_dir=prompt_dir,
            prompt_name=prompt_name,
            prompt=prompt,
            payload=payload,
        )

    runtime = router.resolve(operator_name)
    profile = runtime.profile
    if not profile.api_key:
        raise RuntimeError(
            f"{operator_name} requires API key env {profile.api_key_env}; "
            "set it before running the ScienceFlow pipeline"
        )
    if not profile.base_url or not profile.model:
        raise RuntimeError(f"{operator_name} model profile is missing base_url or model")

    content: list[dict[str, Any]] = []
    if getattr(template, "requires_image", False) and image_base64:
        content.append({"type": "image_url", "image_url": {"url": image_base64}})
    content.append({"type": "text", "text": prompt})

    body: dict[str, Any] = {
        "model": profile.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": runtime.temperature,
        "max_tokens": runtime.max_tokens,
        "stream": runtime.stream,
    }
    if runtime.enable_thinking:
        body["enable_thinking"] = True
    if runtime.reasoning_effort:
        body["reasoning_effort"] = runtime.reasoning_effort

    headers = {
        "Authorization": f"Bearer {profile.api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    last_raw_response: dict[str, Any] | None = None
    last_response_text: str | None = None
    for attempt in range(max(1, max_retries)):
        last_raw_response = None
        last_response_text = None
        try:
            text = None
            for url in _completion_urls(profile.base_url):
                try:
                    if runtime.stream:
                        text = _post_stream_text(url, headers, body, runtime.timeout)
                    else:
                        response = _post_json(url, headers, body, runtime.timeout)
                        last_raw_response = response
                        if snapshot_mode == "all":
                            _write_json(prompt_dir / f"{prompt_name}_raw_response.json", response)
                        text = _completion_content(response)
                    _remember_completion_url(profile.base_url, url)
                    break
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    if exc.code != 404:
                        raise
            if text is None:
                raise last_error or RuntimeError("no completion endpoint responded")
            last_response_text = text
            if snapshot_mode == "all":
                (prompt_dir / f"{prompt_name}_response.txt").write_text(text, encoding="utf-8")
            parsed = _parse_json_object(text)
            if snapshot_mode == "all":
                _write_json(prompt_dir / f"{prompt_name}_parsed.json", parsed)
            return parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 >= max(1, max_retries):
                break
            time.sleep(base_delay * (2 ** attempt))
    if snapshot_mode == "failure" and last_error is not None:
        _write_prompt_snapshot(
            prompt_dir=prompt_dir,
            prompt_name=prompt_name,
            prompt=prompt,
            payload=payload,
            raw_response=last_raw_response,
            response_text=last_response_text,
            error=last_error,
        )
    raise RuntimeError(f"{operator_name} invocation failed: {last_error}") from last_error
