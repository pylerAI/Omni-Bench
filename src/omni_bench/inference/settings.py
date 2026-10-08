"""Resolve the three inference axes from model and benchmark config.

Preferred YAML (model entry; a benchmark entry may set ``audio`` / ``frames``)::

    inference:
      audio: asr_text        # native | none | asr_text
      frames: server         # client | server
      transport: base64      # file | base64
      strip_mm_kwargs: true  # drop mm_processor_kwargs / media_io_kwargs from the body

The pre-``inference:`` flat keys are still read and mean the same thing:
``audio_mode`` -> audio, ``frame_sampling`` -> frames, ``video_transport`` ->
transport, top-level ``strip_mm_kwargs``.

Precedence: benchmark > model > default. ``frames`` is additionally limited to
what the adapter supports (``BenchmarkAdapter.frame_modes``; the first entry is
its protocol default): a model-level choice the adapter cannot honour falls back
to that default, an explicit benchmark-level one is an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: new key -> legacy flat key
LEGACY_KEYS = {
    "audio": "audio_mode",
    "frames": "frame_sampling",
    "transport": "video_transport",
    "strip_mm_kwargs": "strip_mm_kwargs",
}
MODEL_AXES = ("audio", "frames", "transport", "strip_mm_kwargs")
#: transport / strip_mm_kwargs describe the server, so only the model sets them.
BENCHMARK_AXES = ("audio", "frames")

DEFAULT_AUDIO = "native"
DEFAULT_TRANSPORT = "file"
#: Used when no adapter is involved (e.g. a bare client): send the video as is.
DEFAULT_FRAMES = "server"


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    audio: str = DEFAULT_AUDIO
    frames: str = DEFAULT_FRAMES
    transport: str = DEFAULT_TRANSPORT
    strip_mm_kwargs: bool = False

    def describe(self) -> str:
        text = f"audio={self.audio} frames={self.frames} transport={self.transport}"
        return text + (" strip_mm_kwargs" if self.strip_mm_kwargs else "")


def read_inference_block(extra: dict[str, Any], *, allowed: tuple[str, ...], where: str) -> dict[str, Any]:
    """The ``inference:`` block merged with legacy flat keys; only keys that are set.

    ``None`` counts as unset, matching how the legacy keys were read.
    """
    block = extra.get("inference") or {}
    if not isinstance(block, dict):
        raise ValueError(f"{where}: 'inference' must be a mapping, got {type(block).__name__}")
    unknown = sorted(set(block) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown inference key(s) {unknown}. Allowed here: {list(allowed)}")
    values = {k: v for k, v in block.items() if v is not None}
    for key in allowed:
        legacy = LEGACY_KEYS[key]
        if extra.get(legacy) is None:
            continue
        if key in values and _norm(values[key]) != _norm(extra[legacy]):
            raise ValueError(
                f"{where}: inference.{key}={values[key]!r} conflicts with legacy {legacy}={extra[legacy]!r}"
            )
        values.setdefault(key, extra[legacy])
    return values


def _norm(value: Any) -> Any:
    return value.lower() if isinstance(value, str) else value


def resolve_inference(
    model: Any,
    benchmark: Any | None = None,
    frame_modes: tuple[str, ...] | None = None,
) -> InferenceSettings:
    m = read_inference_block(model.extra, allowed=MODEL_AXES, where=f"model '{model.name}'")
    b = (
        read_inference_block(benchmark.extra, allowed=BENCHMARK_AXES, where=f"benchmark '{benchmark.name}'")
        if benchmark is not None
        else {}
    )

    audio = str(b.get("audio", m.get("audio", DEFAULT_AUDIO))).lower()

    if frame_modes:
        if "frames" in b:
            frames = str(b["frames"]).lower()
            if frames not in frame_modes:
                raise ValueError(
                    f"benchmark '{benchmark.name}' sets frames={frames!r}, but its adapter "
                    f"supports only {list(frame_modes)}"
                )
        else:
            frames = str(m.get("frames", frame_modes[0])).lower()
            if frames not in frame_modes:
                frames = frame_modes[0]
    else:
        frames = str(b.get("frames", m.get("frames", DEFAULT_FRAMES))).lower()

    settings = InferenceSettings(
        audio=audio,
        frames=frames,
        transport=str(m.get("transport", DEFAULT_TRANSPORT)).lower(),
        strip_mm_kwargs=bool(m.get("strip_mm_kwargs", False)),
    )
    _validate(settings)
    return settings


def _validate(settings: InferenceSettings) -> None:
    """Fail at resolution time on an unknown option name (registered strategies)."""
    from omni_bench.inference.audio import AUDIO_STRATEGIES
    from omni_bench.inference.frames import FRAME_STRATEGIES
    from omni_bench.inference.transport import TRANSPORTS

    AUDIO_STRATEGIES.get(settings.audio)
    FRAME_STRATEGIES.get(settings.frames)
    TRANSPORTS.get(settings.transport)
