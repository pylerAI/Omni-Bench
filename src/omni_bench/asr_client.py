"""Chat clients for models without a native audio encoder.

Adapters all funnel through :meth:`VllmChatClient.complete`, so audio handling
is swapped here instead of in the five benchmark adapters:

* :class:`NoAudioChatClient`  — drop audio entirely (vision-only baseline)
* :class:`AsrTextChatClient`  — transcribe the audio and inject it as text

Video-MME passes neither ``audio_path`` nor ``video_path`` (it sends sampled
frames as images), so it is untouched by both clients and stays byte-identical
to the native-omni runs.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from omni_bench.asr import AsrSettings, TranscribeCommand, build_transcribe_command, format_transcript
from omni_bench.asr.schema import TranscriptionRequest
from omni_bench.client import ChatCompletionResult, VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig


def _strip_audio_flags(extra_body: dict[str, Any] | None) -> dict[str, Any] | None:
    """Force ``use_audio_in_video`` off so the server never decodes the track."""
    if not extra_body:
        return extra_body
    body = dict(extra_body)
    processor_kwargs = body.get("mm_processor_kwargs")
    if isinstance(processor_kwargs, dict) and "use_audio_in_video" in processor_kwargs:
        body["mm_processor_kwargs"] = {**processor_kwargs, "use_audio_in_video": False}
    return body


class NoAudioChatClient(VllmChatClient):
    """Vision-only baseline: audio inputs are dropped before the request."""

    def complete(
        self,
        prompt: str,
        *,
        audio_path: str | Path | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        return super().complete(
            prompt,
            audio_path=None,
            extra_body=_strip_audio_flags(extra_body),
            **kwargs,
        )


class AsrTextChatClient(NoAudioChatClient):
    """Replace the audio modality with an ASR transcript in the prompt.

    The transcript is prepended to the benchmark's official prompt so that the
    prompt string itself stays byte-identical and the official parsers keep
    working.
    """

    def __init__(
        self,
        model: ModelConfig,
        default_timeout_s: float = 600.0,
        *,
        settings: AsrSettings | None = None,
        command: TranscribeCommand | None = None,
    ) -> None:
        super().__init__(model, default_timeout_s=default_timeout_s)
        self.settings = settings or AsrSettings.from_dict(model.extra.get("asr"))
        self._command = command
        self._command_lock = threading.Lock()

    @property
    def command(self) -> TranscribeCommand:
        """Built lazily so constructing the client never loads Whisper."""
        if self._command is None:
            with self._command_lock:
                if self._command is None:
                    self._command = build_transcribe_command(self.settings)
        return self._command

    @staticmethod
    def _audio_source(
        audio_path: str | Path | None, video_path: str | Path | None
    ) -> Path | None:
        """Audio may arrive as its own file or inside the video container."""
        for candidate in (audio_path, video_path):
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if path.exists():
                return path
        return None

    def transcript_block(self, source: Path) -> str:
        request = TranscriptionRequest(
            media_path=source, language=self.settings.strategy.language
        )
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

    def complete(
        self,
        prompt: str,
        *,
        audio_path: str | Path | None = None,
        video_path: str | Path | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        source = self._audio_source(audio_path, video_path)
        if source is not None:
            prompt = f"{self.transcript_block(source)}\n\n{prompt}"
        return super().complete(prompt, audio_path=None, video_path=video_path, **kwargs)


AUDIO_MODES = ("native", "none", "asr_text")
DEFAULT_AUDIO_MODE = "native"


def _deep_merge(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ``override`` over ``base``, recursing into nested dicts."""
    merged: dict[str, Any] = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_audio_mode(
    model: ModelConfig, benchmark: BenchmarkConfig | None = None
) -> str:
    """Benchmark config wins over model config.

    Audio handling is a property of the *protocol*, not just the model: a
    benchmark whose official setting excludes audio (Video-MME) must stay
    audio-free even when the model is running in ``asr_text`` mode.
    """
    mode = model.extra.get("audio_mode", DEFAULT_AUDIO_MODE)
    if benchmark is not None and benchmark.extra.get("audio_mode") is not None:
        mode = benchmark.extra["audio_mode"]
    mode = str(mode).lower()
    if mode not in AUDIO_MODES:
        raise ValueError(f"Unknown audio_mode '{mode}'. Known: {list(AUDIO_MODES)}")
    return mode


def resolve_asr_settings(
    model: ModelConfig, benchmark: BenchmarkConfig | None = None
) -> AsrSettings:
    """Model-level ``asr:`` block with the benchmark's ``asr:`` block merged in.

    Lets a benchmark tune only what it needs (say ``max_chars`` for long clips)
    without restating the whole engine config.
    """
    raw = dict(model.extra.get("asr") or {})
    if benchmark is not None:
        raw = _deep_merge(raw, benchmark.extra.get("asr") or {})
    return AsrSettings.from_dict(raw)


class AsrCommandPool:
    """Share one Whisper instance across benchmarks with the same engine config.

    Keyed on the engine and cache location only — the prompt-formatting options
    differ per benchmark but do not change what gets transcribed.
    """

    def __init__(self) -> None:
        self._commands: dict[str, TranscribeCommand] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key_for(settings: AsrSettings) -> str:
        spec = settings.strategy.to_dict()
        return json.dumps(
            {"strategy": spec, "cache_dir": str(settings.cache_dir)}, sort_keys=True
        )

    def get(self, settings: AsrSettings) -> TranscribeCommand:
        key = self.key_for(settings)
        command = self._commands.get(key)
        if command is not None:
            return command
        with self._lock:
            if key not in self._commands:
                self._commands[key] = build_transcribe_command(settings)
            return self._commands[key]


def build_chat_client(
    model: ModelConfig,
    default_timeout_s: float = 600.0,
    *,
    benchmark: BenchmarkConfig | None = None,
    pool: AsrCommandPool | None = None,
) -> VllmChatClient:
    """Pick a client from the resolved ``audio_mode``."""
    mode = resolve_audio_mode(model, benchmark)
    if mode == "native":
        return VllmChatClient(model, default_timeout_s=default_timeout_s)
    if mode == "none":
        return NoAudioChatClient(model, default_timeout_s=default_timeout_s)
    settings = resolve_asr_settings(model, benchmark)
    return AsrTextChatClient(
        model,
        default_timeout_s=default_timeout_s,
        settings=settings,
        command=pool.get(settings) if pool else None,
    )
