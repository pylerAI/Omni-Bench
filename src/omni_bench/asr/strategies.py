"""STT strategies.

Every strategy implements the same :class:`SttStrategy` interface, so the
transcription commands never learn which engine is running. New engines are
added by subclassing and calling :func:`register_strategy`; configs select one
by name.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from omni_bench.asr.audio import audio_source, has_audio_stream, media_duration_s
from omni_bench.asr.schema import (
    MediaRef,
    Transcription,
    TranscriptionRequest,
    TranscriptionSegment,
)


@dataclass(slots=True)
class StrategySpec:
    """Serialisable description of a strategy, straight from YAML/CLI."""

    name: str = "faster_whisper"
    model: str = "Systran/faster-whisper-large-v3"
    device: str = "cuda"
    device_index: int = 0
    compute_type: str = "float16"
    language: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "device": self.device,
            "device_index": self.device_index,
            "compute_type": self.compute_type,
            "language": self.language,
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "StrategySpec":
        data = dict(raw or {})
        known = {f for f in cls.__dataclass_fields__ if f != "options"}  # type: ignore[attr-defined]
        options = {**(data.pop("options", None) or {})}
        options.update({k: v for k, v in data.items() if k not in known})
        return cls(**{k: v for k, v in data.items() if k in known}, options=options)


class SttStrategy(ABC):
    """Swappable speech-to-text engine."""

    name: str

    def __init__(self, spec: StrategySpec) -> None:
        self.spec = spec
        self._lock = threading.Lock()
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    def ensure_loaded(self) -> None:
        """Load weights once, lazily, and thread-safely."""
        if self._loaded:
            return
        with self._lock:
            if not self._loaded:
                self.load()
                self._loaded = True

    def load(self) -> None:  # pragma: no cover - default is a no-op
        return None

    # -- contract ----------------------------------------------------------
    @abstractmethod
    def _transcribe(self, audio_path: Path, request: TranscriptionRequest) -> tuple[
        list[TranscriptionSegment], dict[str, Any]
    ]:
        """Return segments plus engine metadata (language, duration, ...)."""

    def transcribe(self, request: TranscriptionRequest) -> Transcription:
        media = request.media_ref()
        started = time.perf_counter()
        if not has_audio_stream(request.media_path):
            empty = Transcription.empty(
                media, backend=self.name, model=self.spec.model, reason="no_audio_stream"
            )
            empty.duration_s = media_duration_s(request.media_path)
            empty.elapsed_s = time.perf_counter() - started
            return empty

        self.ensure_loaded()
        with audio_source(request.media_path) as audio_path:
            segments, meta = self._transcribe(audio_path, request)

        return Transcription(
            media=media,
            backend=self.name,
            model=self.spec.model,
            segments=segments,
            language=meta.get("language"),
            language_probability=meta.get("language_probability"),
            duration_s=meta.get("duration_s"),
            elapsed_s=time.perf_counter() - started,
            params=self.describe_params(request),
        )

    def describe_params(self, request: TranscriptionRequest) -> dict[str, Any]:
        return {
            "device": self.spec.device,
            "device_index": self.spec.device_index,
            "compute_type": self.spec.compute_type,
            "language": request.language or self.spec.language,
            **{**self.spec.options, **request.options},
        }

    # -- identity used by the cache ---------------------------------------
    def cache_namespace(self) -> str:
        model_slug = self.spec.model.strip("/").replace("/", "__")
        return f"{self.name}__{model_slug}"


_REGISTRY: dict[str, Callable[[StrategySpec], SttStrategy]] = {}


def register_strategy(name: str) -> Callable[[type[SttStrategy]], type[SttStrategy]]:
    def decorator(cls: type[SttStrategy]) -> type[SttStrategy]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


def build_strategy(spec: StrategySpec) -> SttStrategy:
    try:
        factory = _REGISTRY[spec.name]
    except KeyError:
        raise ValueError(
            f"Unknown STT strategy '{spec.name}'. Known: {available_strategies()}"
        ) from None
    return factory(spec)


@register_strategy("faster_whisper")
class FasterWhisperStrategy(SttStrategy):
    """CTranslate2 Whisper. Fastest option for bulk offline transcription."""

    DEFAULTS: dict[str, Any] = {
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "word_timestamps": False,
    }

    def __init__(self, spec: StrategySpec) -> None:
        super().__init__(spec)
        self._model: Any = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "faster-whisper is not installed. Install it with "
                "`uv pip install faster-whisper`."
            ) from exc
        self._model = WhisperModel(
            self.spec.model,
            device=self.spec.device,
            device_index=self.spec.device_index,
            compute_type=self.spec.compute_type,
            num_workers=int(self.spec.options.get("num_workers", 1)),
        )

    def _transcribe(self, audio_path: Path, request: TranscriptionRequest):
        options = {**self.DEFAULTS, **self.spec.options, **request.options}
        options.pop("num_workers", None)
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=request.language or self.spec.language,
            **options,
        )
        segments = [
            TranscriptionSegment(
                index=index,
                start_s=float(segment.start),
                end_s=float(segment.end),
                text=str(segment.text).strip(),
            )
            for index, segment in enumerate(segments_iter)
        ]
        meta = {
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration_s": getattr(info, "duration", None),
        }
        return segments, meta


@register_strategy("transformers_whisper")
class TransformersWhisperStrategy(SttStrategy):
    """Reference HF implementation. Slower; useful for cross-checking."""

    DEFAULTS: dict[str, Any] = {
        "chunk_length_s": 30,
        "batch_size": 16,
    }

    def __init__(self, spec: StrategySpec) -> None:
        super().__init__(spec)
        self._pipeline: Any = None

    def load(self) -> None:
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("transformers and torch are required for this strategy.") from exc
        dtype = torch.float16 if self.spec.compute_type in {"float16", "fp16"} else torch.float32
        device = (
            f"cuda:{self.spec.device_index}" if self.spec.device.startswith("cuda") else self.spec.device
        )
        options = {**self.DEFAULTS, **self.spec.options}
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.spec.model,
            torch_dtype=dtype,
            device=device,
            chunk_length_s=int(options["chunk_length_s"]),
            batch_size=int(options["batch_size"]),
        )

    def _transcribe(self, audio_path: Path, request: TranscriptionRequest):
        language = request.language or self.spec.language
        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language
        output = self._pipeline(
            str(audio_path),
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
        segments: list[TranscriptionSegment] = []
        for index, chunk in enumerate(output.get("chunks") or []):
            start, end = (chunk.get("timestamp") or (None, None))[:2]
            segments.append(
                TranscriptionSegment(
                    index=index,
                    start_s=float(start or 0.0),
                    end_s=float(end if end is not None else start or 0.0),
                    text=str(chunk.get("text", "")).strip(),
                )
            )
        if not segments and output.get("text"):
            segments.append(
                TranscriptionSegment(index=0, start_s=0.0, end_s=0.0, text=str(output["text"]).strip())
            )
        return segments, {"language": language, "duration_s": media_duration_s(audio_path)}


__all__ = [
    "FasterWhisperStrategy",
    "MediaRef",
    "SttStrategy",
    "StrategySpec",
    "TransformersWhisperStrategy",
    "available_strategies",
    "build_strategy",
    "register_strategy",
]
