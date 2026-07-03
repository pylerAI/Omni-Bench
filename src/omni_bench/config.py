from __future__ import annotations

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
    return ModelConfig(**data)


def _load_benchmark(raw: dict[str, Any]) -> BenchmarkConfig:
    data, extra = _split_known(raw, BenchmarkConfig)
    data["extra"] = extra
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

    global_cfg = {**(benchmark_raw.get("global", {}) or {}), **(raw.get("global", {}) or {})}
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
