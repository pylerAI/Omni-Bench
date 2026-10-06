from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from omni_bench.config import ModelConfig
from omni_bench.video_transport import VIDEO_TRANSPORTS, TranscodeSettings, video_data_url

#: Per-request processor/IO overrides. Servers that reject them outright (HTTP
#: 400) need the keys gone, not just set to False — see ``strip_mm_kwargs``.
MM_KWARG_KEYS = ("mm_processor_kwargs", "media_io_kwargs")


@dataclass(slots=True)
class ChatCompletionResult:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning: str | None = None
    finish_reason: str | None = None

    def meta(self) -> dict[str, Any]:
        """Fields every adapter stores alongside its own record fields."""
        return {
            "latency_s": self.latency_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "reasoning": self.reasoning,
        }


def _reasoning_of(message: Any) -> str | None:
    """vLLM returns reasoning outside ``content``; the field name varies by version."""
    for name in ("reasoning", "reasoning_content"):
        value = getattr(message, name, None)
        if value:
            return value
    extra = getattr(message, "model_extra", None) or {}
    return extra.get("reasoning") or extra.get("reasoning_content")


def file_url(path: str | Path) -> str:
    return Path(path).expanduser().resolve().as_uri()


def merge_extra_body(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    """Merge two extra_body dicts, one level deep. ``override`` wins on leaf
    conflicts; nested dicts under the same key (e.g. ``mm_processor_kwargs``) are
    merged rather than replaced."""
    merged: dict[str, Any] = {}
    for source in (base, override):
        for key, value in (source or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged


class VllmChatClient:
    def __init__(self, model: ModelConfig, default_timeout_s: float = 600.0) -> None:
        self.model = model
        self.timeout_s = model.request_timeout_s or default_timeout_s
        # Model-level defaults (e.g. chat_template_kwargs to toggle reasoning)
        # merged into every request's extra_body.
        self.default_extra_body: dict[str, Any] = dict(model.extra.get("extra_body") or {})
        self.client = OpenAI(
            base_url=model.resolved_base_url,
            api_key=model.api_key,
            timeout=self.timeout_s,
        )
        self.video_transport = str(model.extra.get("video_transport", "file")).lower()
        if self.video_transport not in VIDEO_TRANSPORTS:
            raise ValueError(
                f"Unknown video_transport '{self.video_transport}'. Known: {list(VIDEO_TRANSPORTS)}"
            )
        self.transcode = TranscodeSettings.from_dict(model.extra.get("transcode"))
        self.strip_mm_kwargs = bool(model.extra.get("strip_mm_kwargs", False))

    def video_part_url(self, video_path: str | Path) -> str:
        if self.video_transport == "base64":
            return video_data_url(video_path, self.transcode)
        return file_url(video_path)

    def complete(
        self,
        prompt: str,
        *,
        image_urls: list[str] | None = None,
        video_path: str | Path | None = None,
        video_url: str | None = None,
        audio_path: str | Path | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float | None = None,
        do_sample: bool | None = None,
        system_prompt: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        content: list[dict[str, Any]] = []
        for image_url in image_urls or []:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        if video_path or video_url:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": video_url or self.video_part_url(video_path)},  # type: ignore[arg-type]
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
        body = merge_extra_body(self.default_extra_body, extra_body)
        if top_p is not None:
            body["top_p"] = top_p
        if do_sample is not None:
            body["do_sample"] = do_sample
        if self.strip_mm_kwargs:
            for key in MM_KWARG_KEYS:
                body.pop(key, None)
        response = self.client.chat.completions.create(
            model=self.model.api_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=body or None,
        )
        latency_s = time.perf_counter() - started
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        return ChatCompletionResult(
            text=text,
            latency_s=latency_s,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            reasoning=_reasoning_of(choice.message),
            finish_reason=getattr(choice, "finish_reason", None),
        )
