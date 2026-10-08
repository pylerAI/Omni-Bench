"""Per-model inference strategies, selected from YAML.

Three independent axes, each a registry of name -> class:

=========  ==========================  ==========================================
axis       registry                    options
=========  ==========================  ==========================================
audio      ``AUDIO_STRATEGIES``        native · none · asr_text
frames     ``FRAME_STRATEGIES``        client · server
transport  ``TRANSPORTS``              file · base64
=========  ==========================  ==========================================

:class:`InferencePipeline` is built once per (model, benchmark) and turns an
adapter's :class:`MediaRequest` into OpenAI chat-completion arguments, so
adapters never branch on these choices. A new option is one class registered on
its axis, e.g. ``@TRANSPORTS.register("s3")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omni_bench.inference.audio import (
    AUDIO_STRATEGIES,
    AsrCommandPool,
    AudioStrategy,
    resolve_asr_settings,
)
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
from omni_bench.inference.settings import InferenceSettings, InferenceWarning, resolve_inference
from omni_bench.inference.transport import TRANSPORTS, Transport

#: Per-request processor/IO overrides. Servers that reject them outright (HTTP
#: 400) need the keys gone, not just set to False — see ``strip_mm_kwargs``.
MM_KWARG_KEYS = ("mm_processor_kwargs", "media_io_kwargs")


@dataclass(slots=True)
class InferencePipeline:
    settings: InferenceSettings
    audio: AudioStrategy
    frames: FrameStrategy
    transport: Transport
    #: Model-level extra_body merged into every request (e.g. chat_template_kwargs).
    default_extra_body: dict[str, Any]

    @classmethod
    def build(
        cls,
        model: Any,
        benchmark: Any | None = None,
        *,
        frame_modes: tuple[str, ...] | None = None,
        asr_pool: AsrCommandPool | None = None,
    ) -> "InferencePipeline":
        settings = resolve_inference(model, benchmark, frame_modes)
        ctx = BuildContext(model=model, benchmark=benchmark, asr_pool=asr_pool)
        return cls(
            settings=settings,
            audio=AUDIO_STRATEGIES.get(settings.audio).create(ctx),
            frames=FRAME_STRATEGIES.get(settings.frames).create(ctx),
            transport=TRANSPORTS.get(settings.transport).create(ctx),
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
        if self.settings.strip_mm_kwargs:
            for key in MM_KWARG_KEYS:
                body.pop(key, None)
        return BuiltRequest(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            extra_body=body or None,
            asr_chars=asr_chars,
        )


__all__ = [
    "AUDIO_STRATEGIES",
    "AsrCommandPool",
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
    "MM_KWARG_KEYS",
    "MediaRequest",
    "Registry",
    "TRANSPORTS",
    "Transport",
    "VideoFrames",
    "file_url",
    "merge_extra_body",
    "resolve_asr_settings",
    "resolve_inference",
]
