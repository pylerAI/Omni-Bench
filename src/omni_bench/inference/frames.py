"""Frames axis: who samples the video.

``client`` — the adapter samples frames (``MediaRequest.frames``) and they are
             sent as images, or as one pre-sampled video with decoding hints.
``server`` — the original video is sent through the transport and the model's
             own processor samples it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

from omni_bench.inference.base import (
    BuildContext,
    ImageFrames,
    MediaRequest,
    Registry,
    VideoFrames,
    merge_extra_body,
)
from omni_bench.inference.transport import Transport

#: (request as seen by the next strategy, visual content parts)
Visual = tuple[MediaRequest, list[dict[str, Any]]]


class FrameStrategy(ABC):
    name: str

    @classmethod
    def create(cls, ctx: BuildContext) -> "FrameStrategy":
        return cls()

    @abstractmethod
    def apply(self, request: MediaRequest, transport: Transport) -> Visual:
        """Visual parts for the request. The returned request keeps ``video_path``
        only if the original video is sent, so ``asr_text`` transcribes just the
        audio the model would otherwise have received."""
        raise NotImplementedError


FRAME_STRATEGIES: Registry[FrameStrategy] = Registry("frames")


def video_part(url: str) -> dict[str, Any]:
    return {"type": "video_url", "video_url": {"url": url}}


@FRAME_STRATEGIES.register("server")
class ServerSampling(FrameStrategy):
    def apply(self, request: MediaRequest, transport: Transport) -> Visual:
        if not request.video_path:
            return request, []
        return request, [video_part(transport.video_url(request.video_path))]


@FRAME_STRATEGIES.register("client")
class ClientSampling(FrameStrategy):
    def apply(self, request: MediaRequest, transport: Transport) -> Visual:
        if request.frames is None:
            if request.video_path:
                raise ValueError("frames=client, but the adapter supplied no client-side frames")
            return request, []
        frames = request.frames()
        request = replace(request, video_path=None)
        if isinstance(frames, ImageFrames):
            return request, [{"type": "image_url", "image_url": {"url": url}} for url in frames.urls]
        if isinstance(frames, VideoFrames):
            if frames.extra_body:
                request = replace(request, extra_body=merge_extra_body(request.extra_body, frames.extra_body))
            return request, [video_part(frames.url)]
        raise TypeError(f"Unsupported client frames: {type(frames).__name__}")
