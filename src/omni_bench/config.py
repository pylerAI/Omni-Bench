from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "benchmarks" / "default.yaml"


@dataclass(slots=True)
class VllmConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    tensor_parallel_size: int | None = 1
    data_parallel_size: int | str | None = "auto"
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModelConfig:
    name: str
    weight_path: str
    served_model_name: str | None = None
    base_url: str | None = None
    api_key: str = "EMPTY"
    vllm: VllmConfig = field(default_factory=VllmConfig)
    request_timeout_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def api_model_name(self) -> str:
        return self.served_model_name or self.name

    @property
    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        return f"http://{self.vllm.host}:{self.vllm.port}/v1"


@dataclass(slots=True)
class BenchmarkConfig:
    name: str
    enabled: bool = True
    data_path: str | None = None
    annotation_file: str | None = None
    video_dir: str | None = None
    result_subdir: str | None = None
    split: str = "test"
    limit: int | None = None
    mode: str = "av"
    max_tokens: int = 8192
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunConfig:
    result_dir: Path
    models: list[ModelConfig]
    benchmarks: list[BenchmarkConfig]
    request_timeout_s: float = 600.0


class ConfigWarning(UserWarning):
    """A config key that no code reads (likely a typo)."""


TOP_LEVEL_KEYS = frozenset({"global", "models", "benchmarks"})
GLOBAL_KEYS = frozenset({"result_dir", "request_timeout_s"})

#: Model keys outside the ModelConfig fields that code actually reads.
MODEL_EXTRA_KEYS = frozenset({
    "inference", "extra_body", "asr", "transcode",
    # legacy spellings of inference.* (still honoured)
    "audio_mode", "frame_sampling", "video_transport", "strip_mm_kwargs",
})

#: Benchmark keys outside the BenchmarkConfig fields, per adapter. "*" applies to all.
BENCHMARK_EXTRA_KEYS: dict[str, frozenset[str]] = {
    # max_workers: cli.py reads it (with concurrency) for every benchmark's perf summary.
    "*": frozenset({"inference", "audio_mode", "frame_sampling", "asr", "concurrency", "limit_mode",
                    "max_workers"}),
    "av_speakerbench": frozenset({"dataset_name", "category", "sub_category", "task_id"}),
    "omnidcbench": frozenset({"run_metrics", "metric_gt_file", "metric_evaluator", "max_workers",
                              "enable_sodam", "metric_credentials"}),
    "omnivideobench": frozenset({"max_frames", "num_frames", "fps", "max_workers", "preprocess_workers",
                                 "preprocess_cache_dir", "top_p", "do_sample", "system_prompt"}),
    "videomme": frozenset({"max_frames", "max_pixels", "use_subtitles", "subtitle_dir",
                           "subtitle_max_chars", "use_audio", "audio_cache_dir"}),
    "worldsense": frozenset({"num_frames", "preprocess_cache_dir"}),
}


def _warn_unknown(kind: str, name: str, extra: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(extra) - allowed)
    if unknown:
        warnings.warn(
            f"{kind} '{name}': unknown config key(s) {unknown} are ignored "
            f"(misspelled? known extra keys: {sorted(allowed)})",
            ConfigWarning,
            stacklevel=3,
        )


def _known_keys(cls: type) -> set[str]:
    return set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]


def _split_known(raw: dict[str, Any], cls: type) -> tuple[dict[str, Any], dict[str, Any]]:
    known = _known_keys(cls) - {"extra"}
    return {k: v for k, v in raw.items() if k in known}, {k: v for k, v in raw.items() if k not in known}


def _load_model(raw: dict[str, Any]) -> ModelConfig:
    data, extra = _split_known(raw, ModelConfig)
    vllm_raw = data.pop("vllm", {}) or {}
    data["vllm"] = VllmConfig(**vllm_raw)
    data["extra"] = extra
    _warn_unknown("model", str(data.get("name")), extra, MODEL_EXTRA_KEYS)
    return ModelConfig(**data)


def _load_benchmark(raw: dict[str, Any]) -> BenchmarkConfig:
    data, extra = _split_known(raw, BenchmarkConfig)
    data["extra"] = extra
    name = str(data.get("name"))
    _warn_unknown("benchmark", name, extra,
                  BENCHMARK_EXTRA_KEYS["*"] | BENCHMARK_EXTRA_KEYS.get(name, frozenset()))
    return BenchmarkConfig(**data)


def _read_yaml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return config_path, yaml.safe_load(f) or {}


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (base_dir / resolved).resolve()
    return resolved


def load_config(path: str | Path, benchmark_path: str | Path | None = None) -> RunConfig:
    config_path, raw = _read_yaml(path)
    benchmark_config_path = Path(benchmark_path) if benchmark_path else DEFAULT_BENCHMARK_CONFIG
    benchmark_config_path, benchmark_raw = _read_yaml(benchmark_config_path)

    for label, doc in ((str(config_path), raw), (str(benchmark_config_path), benchmark_raw)):
        _warn_unknown("config file", label, doc, TOP_LEVEL_KEYS)
    global_cfg = {**(benchmark_raw.get("global", {}) or {}), **(raw.get("global", {}) or {})}
    _warn_unknown("global", "global", global_cfg, GLOBAL_KEYS)
    result_dir = Path(global_cfg.get("result_dir", "results"))
    result_dir = _resolve_path(result_dir, base_dir=config_path.parent)

    models = [_load_model(item) for item in raw.get("models", [])]
    benchmark_items = benchmark_raw.get("benchmarks", raw.get("benchmarks", []))
    benchmarks = [_load_benchmark(item) for item in benchmark_items]
    if not models:
        raise ValueError("Config must contain at least one model.")
    if not benchmarks:
        raise ValueError(f"Benchmark config must contain at least one benchmark: {benchmark_config_path}")

    return RunConfig(
        result_dir=result_dir,
        models=models,
        benchmarks=benchmarks,
        request_timeout_s=float(global_cfg.get("request_timeout_s", 600.0)),
    )
