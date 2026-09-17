# Omni-Bench

**Omni-Bench is an all-in-one evaluation pipeline focused on omni-modal models.**

Like [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), Omni-Bench aims to run many benchmarks from a single runner and store their results in one unified location. It is not a general-purpose VLM evaluation toolkit, though: it **focuses on omni models and omni benchmarks that process audio, video, and text together**. It keeps general-purpose machinery to a minimum and instead optimizes for fast experiment iteration on top of vLLM serving and for faithfully tracking each benchmark's official protocol.

[Evaluation plan](docs/plan.md) · [Setup](#setup) · [Serving](#serving) · [Running evaluations](#running-evaluations) · [Results](#results)

## Goals

The project aims to provide:

1. A common execution interface for evaluating omni models
2. Reproducible benchmark runs against a vLLM-compatible endpoint
3. Faithful preservation of each benchmark's official prompt, parser, and result format
4. A clean split between model configs and benchmark configs
5. Results organized as `records.jsonl`, `summary.json`, and `overall_report.md`

## Supported Models and Benchmarks

Initial evaluation targets:

| Model | Config |
| --- | --- |
| Qwen3-Omni-30B-A3B-Instruct | `configs/models/qwen3_omni.yaml` |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 | `configs/models/nemotron_3_nano_omni.yaml` |

Supported benchmarks:

| Benchmark | Docs | Adapter |
| --- | --- | --- |
| AV-SpeakerBench | [docs/av_speakerbench.md](docs/av_speakerbench.md) | `av_speakerbench` |
| WorldSense | [docs/worldsense.md](docs/worldsense.md) | `worldsense` |
| Video-MME | [docs/videomme.md](docs/videomme.md) | `videomme` |
| OmniVideoBench | [docs/omnivideobench.md](docs/omnivideobench.md) | `omnivideobench` |
| OmniDCBench | [docs/omnidcbench.md](docs/omnidcbench.md) | `omnidcbench` |

## Project Layout

```text
configs/
  benchmarks/default.yaml
  models/qwen3_omni.yaml
  models/nemotron_3_nano_omni.yaml
docs/
scripts/
  serve_qwen3_omni.sh
  serve_nemotron_3_nano_omni.sh
  extract_worldsense_videos.sh
  extract_omnidcbench_videos.sh
src/omni_bench/
  adapters/
  cli.py
  client.py
  config.py
  io.py
submodules/
```

## Setup

```bash
uv sync
git submodule update --init --recursive
```

`vllm[audio]` is already included in the project's default dependencies.

`flash-attn` must be installed per environment with a wheel matching your CUDA, PyTorch, Python, and platform combination. On the current environment (Linux x86_64, Python 3.12, torch 2.11.0+cu130) the following wheel works:

```bash
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl"
```

## Data Preparation

Some datasets ship as archives and must be extracted before evaluation.

```bash
bash scripts/extract_worldsense_videos.sh
bash scripts/extract_omnidcbench_videos.sh
```

Default local dataset paths:

| Benchmark | Local path |
| --- | --- |
| AV-SpeakerBench | `/gpfs/public/datasets/AV-SpeakerBench/` |
| WorldSense | `/gpfs/public/datasets/WorldSense/` |
| Video-MME | `/gpfs/public/datasets/Video-MME/` |
| OmniVideoBench | `/gpfs/public/datasets/OmniVideoBench/` |
| OmniDCBench | `/gpfs/public/datasets/OmniDCBench/` |

## QuickStart

1. Start the vLLM server.

```bash
bash scripts/serve_qwen3_omni.sh
```

2. Run the benchmark you want.

```bash
uv run omni-bench run \
  --config configs/models/qwen3_omni.yaml \
  --benchmark av_speakerbench
```

3. Check the results.

```text
results/qwen3-omni/av_speakerbench/
```

## Serving

vLLM is started first, in a separate terminal.

```bash
bash scripts/serve_qwen3_omni.sh
```

Or:

```bash
bash scripts/serve_nemotron_3_nano_omni.sh
```

Default serving settings:

- Tensor parallel size: `1`
- Data parallel size: number of visible GPUs
- GPU memory utilization: `0.9`
- Allowed local media path: `/gpfs/public/datasets`
- Max audio decode duration: `3600` seconds

## Running Evaluations

Run a single benchmark:

```bash
uv run omni-bench run \
  --config configs/models/qwen3_omni.yaml \
  --benchmark av_speakerbench
```

Run every benchmark enabled in the default benchmark config:

```bash
uv run omni-bench run \
  --config configs/models/qwen3_omni.yaml
```

Use a different benchmark config:

```bash
uv run omni-bench run \
  --config configs/models/qwen3_omni.yaml \
  --benchmark-config configs/benchmarks/default.yaml
```

List the available adapters:

```bash
uv run omni-bench list-benchmarks
```

## Results

[`report.html`](report.html) at the repository root is a report that organizes all benchmark results into charts and tables. It reflects **results measured as of 2026-07-08** (Qwen3-Omni-30B-A3B-Instruct, Nemotron-3-Nano-Omni-30B-A3B-Reasoning FP8/BF16/NVFP4). Every `omni-bench run` regenerates `results/report.html`; the top-level `report.html` is a snapshot of it at that point in time.

Results are stored under:

```text
results/<model-name>/<benchmark-name>/
```

Default output files:

- `records.json`
- `records.jsonl`
- `summary.json`

Some adapters also produce a separate file that can be fed into the official evaluator. For example, Video-MME writes `official_results.json` and OmniDCBench writes `predictions.jsonl`.

Results across all benchmarks are aggregated into:

- `results/run_summary.json`
- `results/overall_report.json`
- `results/overall_report.md`

## Development Guide

To add a new benchmark, follow these steps:

1. Implement the adapter under `src/omni_bench/adapters/`
2. Register the adapter in `src/omni_bench/adapters/__init__.py`
3. Add an entry to `configs/benchmarks/default.yaml` or to a separate benchmark config
4. Document the protocol, metrics, and output table in `docs/<benchmark_name>.md`

To add a new model, add a model config under `configs/models/` and, if needed, write a `scripts/serve_<model>.sh`.
