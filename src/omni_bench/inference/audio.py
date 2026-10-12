"""Audio axis: what happens to the audio modality.

``native`` — send the audio file as an ``audio_url`` part (omni models).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from omni_bench.inference.base import BuildContext, MediaRequest, Registry


class AudioStrategy(ABC):
    name: str

    @classmethod
    def create(cls, ctx: BuildContext) -> "AudioStrategy":
        return cls()

    @abstractmethod
    def apply(self, request: MediaRequest) -> tuple[MediaRequest, int | None]:
        """Rewrite the request; returns it and the length of any text injected into the prompt."""
        raise NotImplementedError


AUDIO_STRATEGIES: Registry[AudioStrategy] = Registry("audio")


@AUDIO_STRATEGIES.register("native")
class NativeAudio(AudioStrategy):
    def apply(self, request: MediaRequest) -> tuple[MediaRequest, int | None]:
        return request, None
