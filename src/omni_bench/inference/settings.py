"""Resolve the inference axes from model and benchmark config.

YAML (model entry; a benchmark entry may set ``audio`` / ``frames``)::

    inference:
      audio: native          # how the audio modality is sent
      frames: client         # who samples video frames (limited per adapter)
      transport: file        # how a local video reaches the server
      reasoning: as_is       # how reasoning is separated from the answer

Every key is optional; omitting the block keeps the defaults below.

Precedence: benchmark > model > default. ``frames`` is additionally limited to
what the adapter supports (``BenchmarkAdapter.frame_modes``; the first entry is
its protocol default): a model-level choice the adapter cannot honour falls back
to that default with a warning, an explicit benchmark-level one is an error.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


class InferenceWarning(UserWarning):
    """An inference setting was adjusted to what the benchmark adapter supports."""


MODEL_AXES = ("audio", "frames", "transport", "reasoning")
#: transport / reasoning describe the server, so only the model sets them.
BENCHMARK_AXES = ("audio", "frames")

DEFAULT_AUDIO = "native"
DEFAULT_TRANSPORT = "file"
DEFAULT_REASONING = "as_is"
#: Used when no adapter is involved (e.g. a bare client): send the video as is.
DEFAULT_FRAMES = "server"


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    audio: str = DEFAULT_AUDIO
    frames: str = DEFAULT_FRAMES
    transport: str = DEFAULT_TRANSPORT
    reasoning: str = DEFAULT_REASONING

    def describe(self) -> str:
        return f"audio={self.audio} frames={self.frames} transport={self.transport} reasoning={self.reasoning}"


def read_inference_block(extra: dict[str, Any], *, allowed: tuple[str, ...], where: str) -> dict[str, Any]:
    """The ``inference:`` block; only keys that are set (``None`` counts as unset)."""
    block = extra.get("inference") or {}
    if not isinstance(block, dict):
        raise ValueError(f"{where}: 'inference' must be a mapping, got {type(block).__name__}")
    unknown = sorted(set(block) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown inference key(s) {unknown}. Allowed here: {list(allowed)}")
    return {k: v for k, v in block.items() if v is not None}


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
    frames = _resolve_frames(model, benchmark, m, b, frame_modes)

    settings = InferenceSettings(
        audio=audio,
        frames=frames,
        transport=str(m.get("transport", DEFAULT_TRANSPORT)).lower(),
        reasoning=str(m.get("reasoning", DEFAULT_REASONING)).lower(),
    )
    _validate(settings)
    return settings


def _known_frames(value: Any, where: str) -> str:
    from omni_bench.inference.frames import FRAME_STRATEGIES

    name = str(value).lower()
    try:
        FRAME_STRATEGIES.get(name)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None
    return name


def _resolve_frames(
    model: Any,
    benchmark: Any | None,
    m: dict[str, Any],
    b: dict[str, Any],
    frame_modes: tuple[str, ...] | None,
) -> str:
    """Benchmark > model > adapter default. Typos always raise; a known mode the
    adapter cannot run falls back to its default with a warning, except when the
    benchmark sets it explicitly (error)."""
    if not frame_modes:
        if "frames" in b:
            return _known_frames(b["frames"], f"benchmark '{benchmark.name}'")
        return _known_frames(m.get("frames", DEFAULT_FRAMES), f"model '{model.name}'")

    default = frame_modes[0]
    if "frames" in b:
        where = f"benchmark '{benchmark.name}'"
        frames = _known_frames(b["frames"], where)
        if frames in frame_modes:
            return frames
        raise ValueError(f"{where} sets frames={frames!r}, but its adapter supports only {list(frame_modes)}")
    if "frames" in m:
        frames = _known_frames(m["frames"], f"model '{model.name}'")
        if frames in frame_modes:
            return frames
        bench = f" for benchmark '{benchmark.name}'" if benchmark is not None else ""
        _warn_fallback(f"model '{model.name}'{bench}", f"frames={frames!r}", default, frame_modes)
    return default


def _warn_fallback(where: str, requested: str, used: str, frame_modes: tuple[str, ...]) -> None:
    warnings.warn(
        f"{where}: {requested} is not supported by the adapter ({list(frame_modes)}); using frames={used!r}",
        InferenceWarning,
        stacklevel=4,
    )


def _validate(settings: InferenceSettings) -> None:
    """Fail at resolution time on an unknown option name (registered strategies)."""
    from omni_bench.inference.audio import AUDIO_STRATEGIES
    from omni_bench.inference.frames import FRAME_STRATEGIES
    from omni_bench.inference.reasoning import REASONING_STRATEGIES
    from omni_bench.inference.transport import TRANSPORTS

    REASONING_STRATEGIES.get(settings.reasoning)
    AUDIO_STRATEGIES.get(settings.audio)
    FRAME_STRATEGIES.get(settings.frames)
    TRANSPORTS.get(settings.transport)
