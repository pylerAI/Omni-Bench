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
| Qwen3.8-27B + Whisper | `configs/models/qwen3_8_27b_*.yaml` |
| Nemotron 3.5 Super VL + Whisper | `configs/models/nemotron_3_5_super_vl_*.yaml` |

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
  asr/                  ASR engine settings (whisper_large_v3.yaml, qwen3_asr.yaml)
  benchmarks/           default.yaml + per-measurement benchmark configs
  models/               one model config per measurement (inference:, display_name:, benchmark_config:)
docs/                   benchmark docs, inference_strategies.md, asr.md
scripts/
  serve_*.sh            vLLM serve scripts (qwen3_omni, nemotron_3_nano_omni, qwen3_8_27b, qwen3_asr)
  prepare_asr.py        pre-fill the ASR transcript cache
  pretranscode_videos.py  pre-build the base64 transcode cache
  strip_truncated.py    drop records cut off at max_tokens so a re-run retries them
  rescore_mcq.py        stricter MCQ letter extraction next to the official parser
  build_report.py
  extract_worldsense_videos.sh
  extract_omnidcbench_videos.sh
src/omni_bench/
  adapters/             one adapter per benchmark (run / parse_record / finalize)
  asr/                  ASR engines, transcript cache, prompt formatting
  inference/            audio / frames / transport / reasoning strategy registries
  cli.py                run · rescore · serve · list-benchmarks
  client.py
  config.py
  io.py
  perf.py               throughput / latency block in summary.json
  report.py
  rescore.py            re-parse stored records without inference
  serving.py
  subtitles.py          Video-MME SRT cues matched to sampled frames
  video_transport.py    base64 transport and transcode cache
submodules/
tests/                  pytest suite (no GPU, server, or dataset needed)
.github/workflows/tests.yml
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

Each `summary.json` also carries a `perf` block, computed from the records after the run (`src/omni_bench/perf.py`):

| Key | Meaning |
| --- | --- |
| `samples` · `errors` | Records in the run and rows that failed |
| `wall_s` · `wall_min` | Wall-clock time of the benchmark (frame decoding and transcript lookup included) |
| `samples_per_s` · `s_per_sample` | Throughput over that wall time |
| `concurrency` | In-flight requests (`concurrency`, else `max_workers`) |
| `latency_s` · `prompt_tokens` · `completion_tokens` | Per-request distributions: `n`, `mean`, `p50`, `p90`, `p99`, `max`, `sum`. Latency includes server queueing, so compare it only within one run |
| `prompt_tokens_per_s` · `output_tokens_per_s` · `total_tokens_per_s` | Token throughput over the wall time |

Every run also writes `config_used.json` to the benchmark directory and to the model directory (the latter is overwritten by each benchmark, so the benchmark-level copy is authoritative). It records `written_at`, `argv`, `config_path`, `benchmark_config_path` (resolved), `result_dir`, `request_timeout_s`, `git` (commit, branch, dirty), `inference` (the resolved audio / frames / transport / reasoning), and the full `model` and `benchmarks` settings. `omni-bench rescore` reads it to re-apply the same settings, and the HTML report takes model labels from its `model.display_name`.

## Development Guide

To add a new benchmark, follow these steps:

1. Implement the adapter under `src/omni_bench/adapters/` by subclassing `BenchmarkAdapter`:
   - `run()` does inference. It hands each request to `client.complete()` (video path, audio path, lazily sampled client frames) instead of building OpenAI content parts itself, and ends by calling `finalize()`.
   - `parse_record(record)` returns the fields the official parser derives from the stored response.
   - `finalize(records, ...)` writes the summary and output files from finished records.
   - `frame_modes` lists the frame-sampling modes the protocol allows; the first is the default.

   `parse_record` and `finalize` are abstract rather than helpers inside `run` so that stored records can be re-parsed and re-summarized through exactly the code a run uses, without inference. `omni-bench rescore` relies on this.
2. Register the adapter in `src/omni_bench/adapters/__init__.py`
3. Add an entry to `configs/benchmarks/default.yaml` or to a separate benchmark config
4. Document the protocol, metrics, and output table in `docs/<benchmark_name>.md`
5. Run the tests: `uv sync && uv run pytest`

To add a new model, add a model config under `configs/models/` and, if needed, write a `scripts/serve_<model>.sh`. Set `display_name:` for the label in the HTML report. Per-model input handling (ASR transcript, frame sampling, video transport, reasoning) is set in the config's `inference:` block; see [docs/inference_strategies.md](docs/inference_strategies.md) and, for ASR transcripts, [docs/asr.md](docs/asr.md).
