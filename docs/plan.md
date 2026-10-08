# Omni-Bench Evaluation Plan

## Purpose

Omni-Bench provides an all-in-one evaluation pipeline for omni-modal models. The initial comparison covers two models:

- Qwen3-Omni-30B-A3B-Instruct
- Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8

Evaluation focuses on video + audio multimodal understanding:

- Audio-visual understanding
- Audio-visual reasoning
- Temporal audio-visual alignment
- Speaker and speech grounding
- General long-video understanding
- Multi-task video understanding
- Throughput and latency

## Models

| Model | Local Weight Path |
| --- | --- |
| Qwen3-Omni-30B-A3B-Instruct | `/gpfs/public/artifacts/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/` |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 | `/gpfs/public/artifacts/models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8/` |

## Benchmarks

- [AV-SpeakerBench](av_speakerbench.md)
- [WorldSense](worldsense.md)
- [Video-MME](videomme.md)
- [OmniVideoBench](omnivideobench.md)
- [OmniDCBench](omnidcbench.md)

## Evaluation Protocol

The runner serves each model behind an OpenAI-compatible vLLM endpoint, and the benchmark adapters send their requests to that endpoint. Each adapter preserves the benchmark's published prompt and result format as closely as possible and writes per-sample records under `results/`.

When a standalone official evaluator exists, the runner also produces a file that can be fed directly into it. When there is no official code, or the official repo only documents reproduction through an external framework, the runner stores transparent records plus locally computed summary metrics.

## Result Locations

Per-benchmark detailed results are stored under:

```text
results/<model-name>/<benchmark-name>/
```

Default files:

- `records.json`: all per-sample results
- `records.jsonl`: per-sample results as JSONL
- `summary.json`: the summary computed by the benchmark adapter

Additional per-benchmark files:

| Benchmark | Extra output | Description |
| --- | --- | --- |
| Video-MME | `official_results.json` | Input format for the official evaluator |
| WorldSense | `vlmeval_rating.json` | VLMEvalKit-style duration/domain/task/audio-class rating |
| OmniDCBench | `predictions.jsonl` | Input format for the official TimeChat-Captioner `Eval` script |

The runner aggregates results across all benchmarks into:

- `results/run_summary.json`: raw summaries per model and per benchmark
- `results/overall_report.json`: representative metrics per benchmark flattened into one row
- `results/overall_report.md`: the full result summary as a markdown table

## Overall Report Format

`overall_report` is built around the representative figure for each benchmark. Metrics that have not been computed yet are shown as `-`.

Representative benchmark metrics:

| Model | AV-SpeakerBench Acc | WorldSense Acc | Video-MME Acc | OmniVideoBench Acc | OmniDCBench F1 | OmniDCBench mIoU | OmniDCBench SODA_M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Omni | - | - | - | - | - | - | - |
| Nemotron-3-Nano-Omni | - | - | - | - | - | - | - |

Throughput is measured separately with `vllm bench throughput` (offline, fixed input/output lengths, text generation) rather than by a benchmark adapter. Results are written to `results/throughput.json` and to the Model throughput section of the HTML report.

| Model | Requests/sec | Output Tokens/sec | Total Tokens/sec |
| --- | --- | --- | --- |
| Qwen3-Omni | - | - | - |
| Nemotron-3-Nano-Omni | - | - | - |

Representative metric definitions:

- AV-SpeakerBench: accuracy over all 3,212 multiple-choice questions
- WorldSense: overall accuracy computed by merging all duration buckets and aggregating the per-domain scores. Detailed analysis also looks at the mean of the per-domain accuracies
- Video-MME: multiple-choice QA accuracy over the full set, as computed by the official evaluator
- OmniVideoBench: accuracy over all QA pairs. Detailed analysis also looks at per-audio-type and per-question-type accuracy
- OmniDCBench:
  - F1: F1 of the matching between predicted and ground-truth timestamp/caption segments
  - mIoU: mean temporal overlap between predicted and ground-truth timestamp spans
  - SODA_M: dense caption score combining temporal alignment and caption content matching
- Throughput (`vllm bench throughput`, offline, fixed input/output lengths):
  - Requests/sec: requests processed per second
  - Output Tokens/sec: generation (decode) token throughput
  - Total Tokens/sec: combined prompt + output token throughput
