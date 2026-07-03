from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from omni_bench.config import ModelConfig


@dataclass(slots=True)
class ChatCompletionResult:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def file_url(path: str | Path) -> str:
    return Path(path).expanduser().resolve().as_uri()


class VllmChatClient:
    def __init__(self, model: ModelConfig, default_timeout_s: float = 600.0) -> None:
        self.model = model
        self.timeout_s = model.request_timeout_s or default_timeout_s
        self.client = OpenAI(
            base_url=model.resolved_base_url,
            api_key=model.api_key,
            timeout=self.timeout_s,
        )

    def complete(
        self,
        prompt: str,
        *,
        video_path: str | Path | None = None,
        video_url: str | None = None,
        audio_path: str | Path | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        content: list[dict[str, Any]] = []
        if video_path or video_url:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": video_url or file_url(video_path)},  # type: ignore[arg-type]
                }
            )
        if audio_path:
            content.append({"type": "audio_url", "audio_url": {"url": file_url(audio_path)}})
        content.append({"type": "text", "text": prompt})

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model.api_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
        latency_s = time.perf_counter() - started
        text = response.choices[0].message.content or ""
        usage = response.usage
        return ChatCompletionResult(
            text=text,
            latency_s=latency_s,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )
