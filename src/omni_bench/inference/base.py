"""Shared contracts for the inference strategies.

An adapter describes *what* to ask in a model-neutral :class:`MediaRequest`;
the three strategy axes (audio · frames · transport) decide *how* it is put on
the wire. Each axis is a :class:`Registry` of name -> class, selected from YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar, Union

T = TypeVar("T")


class Registry(Generic[T]):
    """name -> strategy class for one axis. ``@REGISTRY.register("name")`` adds one."""

    def __init__(self, axis: str) -> None:
        self.axis = axis
        self._classes: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            cls.name = name  # type: ignore[attr-defined]
            self._classes[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        try:
            return self._classes[name]
        except KeyError:
            raise ValueError(f"Unknown {self.axis} '{name}'. Known: {self.names()}") from None

    def names(self) -> list[str]:
        return sorted(self._classes)


@dataclass(frozen=True, slots=True)
class ImageFrames:
    """Client-sampled frames sent as one ``image_url`` part each."""

    urls: list[str]


@dataclass(frozen=True, slots=True)
class VideoFrames:
    """Client-sampled frames packed into one ``video_url`` (e.g. a JPEG-sequence
    data URL), plus the request-body hints the server needs to decode it."""

    url: str
    extra_body: dict[str, Any] | None = None


ClientFrames = Union[ImageFrames, VideoFrames]


@dataclass(slots=True)
class MediaRequest:
    """Model-neutral request an adapter hands to the client.

    ``frames`` is a zero-arg callable so client-side sampling only runs when
    the resolved frames strategy is ``client``.
    """

    prompt: str
    system_prompt: str | None = None
    video_path: str | Path | None = None
    audio_path: str | Path | None = None
    frames: Callable[[], ClientFrames] | None = None
    extra_body: dict[str, Any] | None = None
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float | None = None
    do_sample: bool | None = None


@dataclass(slots=True)
class BuiltRequest:
    """What goes to ``chat.completions.create`` (besides model name)."""

    messages: list[dict[str, Any]]
    max_tokens: int
    temperature: float
    extra_body: dict[str, Any] | None
    asr_chars: int | None = None


@dataclass(slots=True)
class BuildContext:
    """Everything a strategy may need at construction time."""

    model: Any                      # ModelConfig
    benchmark: Any | None = None    # BenchmarkConfig
    asr_pool: Any | None = None     # AsrCommandPool


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
