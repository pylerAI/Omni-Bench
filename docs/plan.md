# Omni-Bench 평가 계획

## 목적

Omni-Bench는 omni-modal model을 평가하기 위한 all-in-one evaluation pipeline을 제공합니다. 초기 비교 대상은 다음 두 모델입니다.

- Qwen3-Omni-30B-A3B-Instruct
- Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8

평가는 video + audio multimodal understanding 능력에 초점을 둡니다.

- Audio-visual understanding
- Audio-visual reasoning
- Temporal audio-visual alignment
- Speaker and speech grounding
- General Long-video understanding
- Multi-task video understanding
- Throughput and latency

## 모델

| Model | Local Weight Path |
| --- | --- |
| Qwen3-Omni-30B-A3B-Instruct | `/gpfs/public/artifacts/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/` |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 | `/gpfs/public/artifacts/models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8/` |

## Benchmark

- [AV-SpeakerBench](av_speakerbench.md)
- [WorldSense](worldsense.md)
- [Video-MME](videomme.md)
- [OmniVideoBench](omnivideobench.md)
- [OmniDCBench](omnidcbench.md)

## 평가 프로토콜

runner는 각 모델을 OpenAI-compatible vLLM endpoint로 serving한 뒤, benchmark adapter가 해당 endpoint에 요청을 보내는 구조로 동작합니다. 각 adapter는 benchmark가 공개한 prompt/result format을 최대한 유지하고, per-sample record를 `results/` 아래 저장합니다.

standalone official evaluator가 있는 경우 runner는 official evaluator에 넣을 수 있는 파일을 함께 생성합니다. official code가 없거나 외부 framework 재현만 안내하는 경우에는 transparent record와 local summary metric을 저장합니다.

## 결과 저장 위치

각 benchmark의 상세 결과는 아래 경로에 저장됩니다.

```text
results/<model-name>/<benchmark-name>/
```

기본 파일은 다음과 같습니다.

- `records.json`: per-sample 결과 전체
- `records.jsonl`: per-sample 결과 JSONL
- `summary.json`: benchmark adapter가 계산한 summary

benchmark별 추가 파일:

| Benchmark | 추가 출력 | 설명 |
| --- | --- | --- |
| Video-MME | `official_results.json` | official evaluator 입력 포맷 |
| Video-MME | `throughput_summary.json` | latency / throughput summary |
| WorldSense | `vlmeval_rating.json` | VLMEvalKit 방식 duration/domain/task/audio-class rating |
| OmniDCBench | `predictions.jsonl` | TimeChat-Captioner official `Eval` script 입력 포맷 |

전체 benchmark 결과는 runner가 아래 파일로 취합합니다.

- `results/run_summary.json`: 모델별, benchmark별 raw summary
- `results/overall_report.json`: benchmark별 대표 metric을 한 row로 펼친 JSON
- `results/overall_report.md`: markdown table 형태의 전체 결과 요약

## 전체 결과 요약 형식

`overall_report`는 benchmark별 대표 수치를 중심으로 구성합니다. 값이 아직 계산되지 않은 metric은 `-`로 표시합니다.

| Model | AV-SpeakerBench Acc | WorldSense Acc | Video-MME Acc | Video-MME Avg Latency | Video-MME P95 Latency | Video-MME Samples/sec | Video-MME Tokens/sec | OmniVideoBench Acc | OmniDCBench F1 | OmniDCBench mIoU | OmniDCBench SODA_M |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Omni | - | - | - | - | - | - | - | - | - | - | - |
| Nemotron-3-Nano-Omni | - | - | - | - | - | - | - | - | - | - | - |

대표 metric 정의:

- AV-SpeakerBench: overall accuracy
- WorldSense: VLMEvalKit-style overall accuracy
- Video-MME: official evaluator accuracy와 throughput summary
- OmniVideoBench: overall accuracy
- OmniDCBench: F1, mIoU, SODA_M
