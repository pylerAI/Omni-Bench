"""Per-model inference strategies, selected from YAML.

Four independent axes, each a registry of name -> class:

=========  ==========================  ==========================================
axis       registry                    options
=========  ==========================  ==========================================
audio      ``AUDIO_STRATEGIES``        native
frames     ``FRAME_STRATEGIES``        client · server (fixed per adapter)
transport  ``TRANSPORTS``              file
reasoning  ``REASONING_STRATEGIES``    as_is
=========  ==========================  ==========================================

:class:`InferencePipeline` is built once per (model, benchmark) and turns an
adapter's :class:`MediaRequest` into OpenAI chat-completion arguments (and
reads the answer out of the response), so adapters never branch on these
choices. A new option is one class registered on its axis, e.g.
``@TRANSPORTS.register("s3")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omni_bench.inference.audio import AUDIO_STRATEGIES, AudioStrategy
from omni_bench.inference.base import (
    BuildContext,
    BuiltRequest,
    ClientFrames,
    ImageFrames,
    MediaRequest,
    Registry,
    VideoFrames,
    file_url,
    merge_extra_body,
)
from omni_bench.inference.frames import FRAME_STRATEGIES, FrameStrategy
from omni_bench.inference.reasoning import REASONING_STRATEGIES, ReasoningStrategy
from omni_bench.inference.settings import InferenceSettings, InferenceWarning, resolve_inference
from omni_bench.inference.transport import TRANSPORTS, Transport


@dataclass(slots=True)
class InferencePipeline:
    settings: InferenceSettings
    audio: AudioStrategy
    frames: FrameStrategy
    transport: Transport
    reasoning: ReasoningStrategy
    #: Model-level extra_body merged into every request (e.g. chat_template_kwargs).
    default_extra_body: dict[str, Any]

    @classmethod
    def build(
        cls,
        model: Any,
        benchmark: Any | None = None,
        *,
        frame_modes: tuple[str, ...] | None = None,
    ) -> "InferencePipeline":
        settings = resolve_inference(model, benchmark, frame_modes)
        ctx = BuildContext(model=model, benchmark=benchmark)
        return cls(
            settings=settings,
            audio=AUDIO_STRATEGIES.get(settings.audio).create(ctx),
            frames=FRAME_STRATEGIES.get(settings.frames).create(ctx),
            transport=TRANSPORTS.get(settings.transport).create(ctx),
            reasoning=REASONING_STRATEGIES.get(settings.reasoning).create(ctx),
            default_extra_body=dict(model.extra.get("extra_body") or {}),
        )

    def build_request(self, request: MediaRequest) -> BuiltRequest:
        # Frames first: client-side decoding hints join the request body, and
        # whether the original video is sent is settled, before the audio
        # strategy rewrites the request.
        request, visual = self.frames.apply(request, self.transport)
        request, asr_chars = self.audio.apply(request)

        content: list[dict[str, Any]] = list(visual)
        if request.audio_path:
            content.append({"type": "audio_url", "audio_url": {"url": file_url(request.audio_path)}})
        content.append({"type": "text", "text": request.prompt})

        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": content})

        body = merge_extra_body(self.default_extra_body, request.extra_body)
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.do_sample is not None:
            body["do_sample"] = request.do_sample
        return BuiltRequest(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            extra_body=body or None,
            asr_chars=asr_chars,
        )


__all__ = [
    "AUDIO_STRATEGIES",
    "AudioStrategy",
    "BuildContext",
    "BuiltRequest",
    "ClientFrames",
    "FRAME_STRATEGIES",
    "FrameStrategy",
    "ImageFrames",
    "InferencePipeline",
    "InferenceSettings",
    "InferenceWarning",
    "MediaRequest",
    "REASONING_STRATEGIES",
    "ReasoningStrategy",
    "Registry",
    "TRANSPORTS",
    "Transport",
    "VideoFrames",
    "file_url",
    "merge_extra_body",
    "resolve_inference",
]
