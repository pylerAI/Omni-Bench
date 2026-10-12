"""Audio axis: what happens to the audio modality.

``native``   — send the audio file as an ``audio_url`` part (omni models)
``none``     — drop it, and force ``use_audio_in_video`` off (vision-only baseline)
``asr_text`` — drop it and prepend an ASR transcript to the prompt. The engine
               is chosen by ``asr.strategy.name`` (see :mod:`omni_bench.asr`).

The benchmark's official prompt is kept byte-identical; the transcript block
goes in front of it so the official parsers keep working.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any

from omni_bench.asr import AsrSettings, TranscribeCommand, build_transcribe_command, format_transcript
from omni_bench.asr.schema import TranscriptionRequest
from omni_bench.inference.base import BuildContext, MediaRequest, Registry


class AudioStrategy(ABC):
    name: str

    @classmethod
    def create(cls, ctx: BuildContext) -> "AudioStrategy":
        return cls()

    @abstractmethod
    def apply(self, request: MediaRequest) -> tuple[MediaRequest, int | None]:
        """Rewrite the request; returns it and the transcript length (asr_chars)."""
        raise NotImplementedError


AUDIO_STRATEGIES: Registry[AudioStrategy] = Registry("audio")


@AUDIO_STRATEGIES.register("native")
class NativeAudio(AudioStrategy):
    def apply(self, request: MediaRequest) -> tuple[MediaRequest, int | None]:
        return request, None


def _strip_audio_flags(extra_body: dict[str, Any] | None) -> dict[str, Any] | None:
    """Force ``use_audio_in_video`` off so the server never decodes the track."""
    if not extra_body:
        return extra_body
    body = dict(extra_body)
    processor_kwargs = body.get("mm_processor_kwargs")
    if isinstance(processor_kwargs, dict) and "use_audio_in_video" in processor_kwargs:
        body["mm_processor_kwargs"] = {**processor_kwargs, "use_audio_in_video": False}
    return body


@AUDIO_STRATEGIES.register("none")
class NoAudio(AudioStrategy):
    def apply(self, request: MediaRequest) -> tuple[MediaRequest, int | None]:
        return replace(request, audio_path=None, extra_body=_strip_audio_flags(request.extra_body)), None


@AUDIO_STRATEGIES.register("asr_text")
class AsrTextAudio(NoAudio):
    def __init__(self, settings: AsrSettings, command: TranscribeCommand | None = None) -> None:
        self.settings = settings
        self._command = command
        self._command_lock = threading.Lock()

    @classmethod
    def create(cls, ctx: BuildContext) -> "AudioStrategy":
        settings = resolve_asr_settings(ctx.model, ctx.benchmark)
        return cls(settings, ctx.asr_pool.get(settings) if ctx.asr_pool else None)

    @property
    def command(self) -> TranscribeCommand:
        """Built lazily so constructing the strategy never loads an STT engine."""
        if self._command is None:
            with self._command_lock:
                if self._command is None:
                    self._command = build_transcribe_command(self.settings)
        return self._command

    @staticmethod
    def audio_source(audio_path: str | Path | None, video_path: str | Path | None) -> Path | None:
        """Audio may arrive as its own file or inside the video container."""
        for candidate in (audio_path, video_path):
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if path.exists():
                return path
        return None

    def transcript_block(self, source: Path) -> str:
        request = TranscriptionRequest(media_path=source, language=self.settings.strategy.language)
        if self.settings.strict_cache:
            cached = self.command.lookup(request)
            if cached is None:
                raise FileNotFoundError(
                    f"No cached transcript for {source}. Run scripts/prepare_asr.py "
                    "first, or set asr.strict_cache=false to transcribe inline."
                )
            transcription = cached
        else:
            transcription = self.command.execute(request)
        return format_transcript(
            transcription,
            header=self.settings.header,
            empty_text=self.settings.empty_text,
            with_timestamps=self.settings.with_timestamps,
            max_chars=self.settings.max_chars,
            max_end_s=self.settings.max_end_s,
        )

    def apply(self, request: MediaRequest) -> tuple[MediaRequest, int | None]:
        source = self.audio_source(request.audio_path, request.video_path)
        block = self.transcript_block(source) if source is not None else None
        if block is not None:
            request = replace(request, prompt=f"{block}\n\n{request.prompt}")
        request, _ = super().apply(request)
        return request, len(block) if block is not None else None


def _deep_merge(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ``override`` over ``base``, recursing into nested dicts."""
    merged: dict[str, Any] = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_asr_settings(model: Any, benchmark: Any | None = None) -> AsrSettings:
    """Model-level ``asr:`` block with the benchmark's ``asr:`` block merged in.

    Lets a benchmark tune only what it needs (say ``max_chars`` for long clips)
    without restating the whole engine config.
    """
    raw = dict(model.extra.get("asr") or {})
    if benchmark is not None:
        raw = _deep_merge(raw, benchmark.extra.get("asr") or {})
    return AsrSettings.from_dict(raw)


class AsrCommandPool:
    """Share one STT engine across benchmarks with the same engine config.

    Keyed on the engine and cache location only — the prompt-formatting options
    differ per benchmark but do not change what gets transcribed.
    """

    def __init__(self) -> None:
        self._commands: dict[str, TranscribeCommand] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key_for(settings: AsrSettings) -> str:
        spec = settings.strategy.to_dict()
        return json.dumps({"strategy": spec, "cache_dir": str(settings.cache_dir)}, sort_keys=True)

    def get(self, settings: AsrSettings) -> TranscribeCommand:
        key = self.key_for(settings)
        command = self._commands.get(key)
        if command is not None:
            return command
        with self._lock:
            if key not in self._commands:
                self._commands[key] = build_transcribe_command(settings)
            return self._commands[key]
