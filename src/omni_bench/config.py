from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from omni_bench.io import warn


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
    #: Label shown in report.html; the result dir name (``name``) when unset.
    display_name: str | None = None
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
    #: Benchmark config actually loaded (after CLI > model key > default).
    benchmark_config_path: Path | None = None
    #: Model config files merged through ``extends:``, base first, the given file last.
    config_chain: list[Path] = field(default_factory=list)


#: ``benchmark_config`` / ``extends`` are read from the model config file only.
TOP_LEVEL_KEYS = frozenset({"global", "models", "benchmarks", "benchmark_config", "extends"})
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
        warn(f"{kind} '{name}': unknown config key(s) {unknown} are ignored "
             f"(misspelled? known keys: {sorted(allowed)})")


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


def deep_merge(base: Any, override: Any) -> Any:
    """``override`` over ``base``: dicts merge key by key (override wins), any
    other value (lists included) is replaced wholesale."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = deep_merge(base[key], value) if key in base else value
        return merged
    return override


def _absolutize_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Path-valued keys resolve against the file that declares them."""
    out = dict(raw)
    if out.get("benchmark_config"):
        out["benchmark_config"] = str(_resolve_path(Path(str(out["benchmark_config"])).expanduser(), base_dir=base_dir))
    glob = out.get("global")
    if isinstance(glob, dict) and glob.get("result_dir") is not None:
        out["global"] = {**glob, "result_dir": str(_resolve_path(Path(str(glob["result_dir"])).expanduser(), base_dir=base_dir))}
    return out


def _merge_config(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """One ``extends`` step. ``models`` is special: when base and child each
    define exactly one model, the child's entry is deep-merged onto the base's;
    otherwise the child's list replaces the base's."""
    merged = deep_merge(base, {k: v for k, v in child.items() if k != "models"})
    if "models" in child:
        b_models, c_models = base.get("models"), child.get("models")
        if isinstance(b_models, list) and isinstance(c_models, list) and len(b_models) == 1 and len(c_models) == 1:
            merged["models"] = [deep_merge(b_models[0], c_models[0])]
        else:
            merged["models"] = c_models
    return merged


def load_raw_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> tuple[dict[str, Any], list[Path]]:
    """A model config with its ``extends:`` merged (path-valued keys absolutized).

    ``extends:`` names one base config, or a list of them, relative to the
    file. A list is merged left to right, then the file itself on top; bases
    may extend further. Returns the merged mapping (without ``extends``) and
    every file that contributed, in merge order (the given file last).
    """
    config_path = Path(path).expanduser().resolve()
    if config_path in _seen:
        cycle = " -> ".join(str(p) for p in (*_seen, config_path))
        raise ValueError(f"config extends cycle: {cycle}")
    _, raw = _read_yaml(config_path)
    raw = _absolutize_paths(raw, config_path.parent)
    parents = raw.pop("extends", None) or []
    if isinstance(parents, (str, Path)):
        parents = [parents]
    merged: dict[str, Any] = {}
    chain: list[Path] = []
    for parent in parents:
        base, base_chain = load_raw_config(
            _resolve_path(Path(str(parent)).expanduser(), base_dir=config_path.parent), (*_seen, config_path)
        )
        merged = _merge_config(merged, base) if merged else base
        chain.extend(p for p in base_chain if p not in chain)
    merged = _merge_config(merged, raw) if merged else raw
    return merged, [*chain, config_path]


def resolve_benchmark_config(
    config_path: str | Path | None, benchmark_path: str | Path | None = None
) -> Path:
    """CLI ``--benchmark-config`` > the model config's ``benchmark_config:`` key
    (relative to that file) > ``configs/benchmarks/default.yaml``."""
    if benchmark_path:
        return Path(benchmark_path).expanduser().resolve()
    if config_path:
        raw, _ = load_raw_config(config_path)
        if raw.get("benchmark_config"):
            return Path(raw["benchmark_config"])
    return DEFAULT_BENCHMARK_CONFIG


def load_config(path: str | Path, benchmark_path: str | Path | None = None) -> RunConfig:
    config_path = Path(path)
    raw, chain = load_raw_config(config_path)
    benchmark_config_path = resolve_benchmark_config(config_path, benchmark_path)
    benchmark_config_path, benchmark_raw = _read_yaml(benchmark_config_path)

    _warn_unknown("config file", str(config_path), raw, TOP_LEVEL_KEYS)
    _warn_unknown("config file", str(benchmark_config_path), benchmark_raw, TOP_LEVEL_KEYS - {"benchmark_config"})
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
        benchmark_config_path=benchmark_config_path,
        config_chain=chain,
    )
