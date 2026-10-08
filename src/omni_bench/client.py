from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.inference import ClientFrames, InferencePipeline, MediaRequest


@dataclass(slots=True)
class ChatCompletionResult:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class VllmChatClient:
    """Sends an adapter's neutral request; the pipeline decides how it is encoded."""

    def __init__(
        self,
        model: ModelConfig,
        default_timeout_s: float = 600.0,
        *,
        pipeline: InferencePipeline | None = None,
    ) -> None:
        self.model = model
        self.timeout_s = model.request_timeout_s or default_timeout_s
        self.pipeline = pipeline or InferencePipeline.build(model)
        self.client = OpenAI(
            base_url=model.resolved_base_url,
            api_key=model.api_key,
            timeout=self.timeout_s,
        )

    @property
    def frames_mode(self) -> str:
        return self.pipeline.settings.frames

    def complete(
        self,
        prompt: str,
        *,
        video_path: str | Path | None = None,
        audio_path: str | Path | None = None,
        frames: Callable[[], ClientFrames] | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float | None = None,
        do_sample: bool | None = None,
        system_prompt: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        """``frames`` lazily yields client-sampled frames; used only when the
        resolved frames strategy is ``client``."""
        built = self.pipeline.build_request(
            MediaRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                video_path=video_path,
                audio_path=audio_path,
                frames=frames,
                extra_body=extra_body,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        )
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model.api_model_name,
            messages=built.messages,
            max_tokens=built.max_tokens,
            temperature=built.temperature,
            extra_body=built.extra_body,
        )
        latency_s = time.perf_counter() - started
        content = response.choices[0].message.content or ""
        text, _ = self.pipeline.reasoning.split(content, None)
        usage = response.usage
        return ChatCompletionResult(
            text=text,
            latency_s=latency_s,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )


def build_chat_client(
    model: ModelConfig,
    default_timeout_s: float = 600.0,
    *,
    benchmark: BenchmarkConfig | None = None,
    frame_modes: tuple[str, ...] | None = None,
) -> VllmChatClient:
    """Client whose pipeline is resolved for this (model, benchmark) pair.

    ``frame_modes`` is the adapter's supported frames modes, default first.
    """
    pipeline = InferencePipeline.build(model, benchmark, frame_modes=frame_modes)
    return VllmChatClient(model, default_timeout_s=default_timeout_s, pipeline=pipeline)
