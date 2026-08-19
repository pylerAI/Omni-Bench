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

import threading
from pathlib import Path
from typing import Any

from omni_bench.asr import AsrSettings, TranscribeCommand, build_transcribe_command, format_transcript
from omni_bench.asr.schema import TranscriptionRequest
from omni_bench.client import ChatCompletionResult, VllmChatClient
from omni_bench.config import ModelConfig


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


def build_chat_client(model: ModelConfig, default_timeout_s: float = 600.0) -> VllmChatClient:
    """Pick a client from the model config's ``audio_mode``."""
    mode = str(model.extra.get("audio_mode", "native")).lower()
    if mode == "native":
        return VllmChatClient(model, default_timeout_s=default_timeout_s)
    if mode == "none":
        return NoAudioChatClient(model, default_timeout_s=default_timeout_s)
    if mode == "asr_text":
        return AsrTextChatClient(model, default_timeout_s=default_timeout_s)
    raise ValueError(f"Unknown audio_mode '{mode}'. Known: {list(AUDIO_MODES)}")
