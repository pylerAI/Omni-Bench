"""ASR strategy backed by a vLLM-served ASR model.

Qwen3-ASR is an encoder-decoder chat model whose feature extractor takes a fixed
30-second window (``chunk_length: 30``, ``n_samples: 480000``). Unlike Whisper's
Python implementations it has no sliding-window loop for long audio, so this
strategy slices the input itself and stitches the pieces back together, deriving
each segment's timestamp from its chunk offset.

Serving it separately from the model under test is deliberate: running both on
the same GPUs starves the vLLM workers and kills the engine.
"""

from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from omni_bench.asr.audio import audio_source, ffmpeg_binary, media_duration_s
from omni_bench.asr.schema import TranscriptionRequest, TranscriptionSegment
from omni_bench.asr.strategies import SttStrategy, register_strategy


def slice_wav(source: Path, out_dir: Path, chunk_s: float, limit_s: float | None) -> list[tuple[float, Path]]:
    """Cut ``source`` into ``chunk_s`` pieces, returning (start_s, path) pairs."""
    duration = media_duration_s(source) or 0.0
    if limit_s is not None:
        duration = min(duration, limit_s)
    if duration <= 0:
        return []
    pieces: list[tuple[float, Path]] = []
    start = 0.0
    index = 0
    while start < duration:
        target = out_dir / f"{source.stem}_{index:04d}.wav"
        command = [
            ffmpeg_binary(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-t", f"{chunk_s:.3f}", "-i", str(source),
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(target),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0 and target.exists() and target.stat().st_size > 1024:
            pieces.append((start, target))
        start += chunk_s
        index += 1
    return pieces


@register_strategy("vllm_asr")
class VllmAsrStrategy(SttStrategy):
    """Transcribe through an OpenAI-compatible endpoint serving an ASR model."""

    DEFAULTS: dict[str, Any] = {
        "base_url": "http://127.0.0.1:8002/v1",
        "served_model_name": "Qwen3-ASR-1.7B",
        "chunk_s": 30.0,
        "limit_s": None,          # cap total audio read (fairness knob)
        "max_tokens": 448,
        "temperature": 0.0,
        "system_prompt": "",
        "timeout_s": 300.0,
        "api_key": "EMPTY",
    }

    def __init__(self, spec) -> None:
        super().__init__(spec)
        self._client: Any = None

    def _opt(self, request: TranscriptionRequest | None = None) -> dict[str, Any]:
        merged = {**self.DEFAULTS, **self.spec.options}
        if request is not None:
            merged.update(request.options)
        return merged

    def load(self) -> None:
        from openai import OpenAI

        o = self._opt()
        self._client = OpenAI(base_url=o["base_url"], api_key=o["api_key"],
                              timeout=float(o["timeout_s"]))

    def _transcribe_chunk(self, wav: Path, options: dict[str, Any]) -> str:
        payload = base64.b64encode(wav.read_bytes()).decode("ascii")
        content: list[dict[str, Any]] = [
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{payload}"}}
        ]
        messages: list[dict[str, Any]] = []
        if options.get("system_prompt"):
            messages.append({"role": "system", "content": options["system_prompt"]})
        messages.append({"role": "user", "content": content})
        response = self._client.chat.completions.create(
            model=options["served_model_name"],
            messages=messages,
            max_tokens=int(options["max_tokens"]),
            temperature=float(options["temperature"]),
        )
        return (response.choices[0].message.content or "").strip()

    def _transcribe(self, audio_path: Path, request: TranscriptionRequest):
        options = self._opt(request)
        chunk_s = float(options["chunk_s"])
        limit = options["limit_s"]
        limit_s = float(limit) if limit is not None else None

        segments: list[TranscriptionSegment] = []
        with tempfile.TemporaryDirectory(prefix="vllm-asr-") as tmp:
            for index, (start, piece) in enumerate(
                slice_wav(audio_path, Path(tmp), chunk_s, limit_s)
            ):
                text = self._transcribe_chunk(piece, options)
                if text:
                    segments.append(
                        TranscriptionSegment(
                            index=len(segments),
                            start_s=start,
                            end_s=min(start + chunk_s, (limit_s or 1e9)),
                            text=text,
                        )
                    )
        meta = {
            "language": request.language or self.spec.language,
            "duration_s": media_duration_s(audio_path),
        }
        return segments, meta
