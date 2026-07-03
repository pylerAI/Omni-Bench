
# Qwen3-Omni / Nemotron-nano-Omni Benchmark 실험 계획

## 1. 목적

Qwen3-Omni와 Nemotron-nano-Omni의 **Video + Audio multimodal understanding 성능**을 비교 평가

주요 평가 대상은 아래와 같음

- Audio-Visual understanding
- Audio-Visual reasoning
- Temporal audio-visual alignment
- Speaker / speech grounding
- Long-video understanding
- Multi-task video understanding
- Throughput / latency / cost efficiency

---

## 2. 평가 대상 모델

- **Qwen3-Omni-30B-A3B-Instruct**
    - https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
    - Local path: `/gpfs/public/artifacts/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/`
- **Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8**
    - https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
    - Local path: `/gpfs/public/artifacts/models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8/`

---

## 3. Benchmark 후보 우선순위

### 3.1 WorldSense (ICLR 2026)

- https://arxiv.org/abs/2502.04326v1
- Local path (gpfs): `/gpfs/public/datasets/WorldSense/`

**선정 이유**

Real-world video에서 **visual, audio, text를 동시에 요구하는 omnimodal understanding benchmark**임. Audio-Visual synchronized video 기반이라 Qwen3-Omni / Nemotron-nano-Omni 비교에 가장 직접적임.

**특징 / 분포**

| 항목 | 내용 |
| --- | --- |
| 데이터 규모 | 1,662개 audio-visual synchronized video |
| QA 수 | 3,172개 multi-choice QA |
| 도메인 | 8개 primary domain |
| 세부 카테고리 | 67개 subcategory |
| Task | 26개 task |
| 평가 포인트 | audio-visual fusion, real-world event understanding, cross-modal reasoning |

출처: WorldSense는 1,662개 audio-visual synchronized video, 3,172개 MCQ, 8개 domain, 67개 subcategory, 26개 task로 구성됨.

### 평가 Metric

| Metric | 설명 |
| --- | --- |
| Overall Accuracy | 전체 MCQ 정답률 |
| Domain-wise Accuracy | 도메인별 정답률 |
| Task-wise Accuracy | task 유형별 정답률 |

---

### 3.2 AV-SpeakerBench (CVPR 2026 Findings)

https://arxiv.org/abs/2512.02231

- Local path (gpfs): `/gpfs/public/datasets/AV-SpeakerBench/`

**선정 이유**

Speaker-centric audio-visual reasoning 평가에 적합함. 단순 ASR 성능이 아니라 **누가 말했는지, 무엇을 말했는지, 언제 말했는지**를 audio와 visual cue를 결합해 판단해야 함.

**특징 / 분포**

| 항목 | 내용 |
| --- | --- |
| QA 수 | 3,212개 multiple-choice questions |
| 데이터 유형 | real-world speaker video |
| 평가 포인트 | who spoke, what was said, when it happened |
| 강점 | speech와 visual speaker grounding을 분리 평가 가능 |

출처: AV-SpeakerBench는 3,212개 MCQ로 구성되며 speaker-centric audiovisual reasoning, 특히 who/what/when alignment를 평가함.

### 평가 Metric

| Metric | 설명 |
| --- | --- |
| Overall Accuracy | 전체 MCQ 정답률 |
| Sub-task Accuracy | speaker / speech / temporal 유형별 정답률 |

---

### 3.3 Video-MME

**선정 이유**

Audio-Visual fusion 전용은 아니지만, long-video MLLM의 기본 체력 측정용으로 적합함. Short / medium / long video 구간별 성능 비교가 가능하고, subtitle/audio 정보가 video understanding에 미치는 영향도 확인 가능함.

- Local path (gpfs): `/gpfs/public/datasets/Video-MME/`

**특징 / 분포**

| 항목 | 내용 |
| --- | --- |
| 데이터 규모 | 900개 video |
| 총 길이 | 254시간 |
| QA 수 | 2,700개 human-annotated QA |
| Duration | short / medium / long video 포함 |
| 평가 포인트 | long-video understanding, temporal reasoning, subtitle/audio 활용 |

출처: Video-MME는 900개 video, 총 254시간, 2,700개 QA로 구성된 video MLLM benchmark임.

### 평가 Metric

| Metric | 설명 |
| --- | --- |
| Overall Accuracy | 전체 MCQ 정답률 |
| Short-video Accuracy | short video subset 정답률 |
| Medium-video Accuracy | medium video subset 정답률 |
| Long-video Accuracy | long video subset 정답률 |
| Domain-wise Accuracy | 도메인별 정답률 |

### Throughput Metric

Video-MME에서만 별도 측정

| Metric | 설명 |
| --- | --- |
| Avg Latency / Sample | sample 1개당 평균 처리 시간 |
| P50 Latency | latency median |
| P95 Latency | latency 95 percentile |
| Samples / sec | 초당 처리 sample 수 |
| Videos / hour | 시간당 처리 video 수 |
| Tokens / sec | prompt / completion / total token throughput |

Video-MME 실행 시 `throughput_summary.json`에 위 throughput metric을 저장한다.

---

### 3.4 OmniVideoBench (ICLR 2026)

**선정 이유**

Audio와 Visual cue의 **synergistic reasoning**을 평가하는 benchmark임. 단순 정답률 외에도 step-by-step reasoning annotation이 있어, 어떤 modality evidence를 활용해야 했는지 error analysis에 유리함.

- Local path (gpfs): `/gpfs/public/datasets/OmniVideoBench/`

**특징 / 분포**

| 항목 | 내용 |
| --- | --- |
| 데이터 규모 | 628개 video |
| QA 수 | 1,000개 QA pair |
| Video 길이 | 수 초 ~ 30분 |
| 카테고리 | 8개 major category, 68개 subcategory |
| Question type | 13개 reasoning type |
| 추가 annotation | step-by-step reasoning trace |
| 평가 포인트 | audio-visual complementarity, logical consistency, long-term temporal reasoning |

출처: OmniVideoBench는 628개 video, 1,000개 QA, 8개 category, 68개 subcategory, 13개 reasoning type으로 구성되며, 각 QA에 reasoning trace가 포함됨.

### 평가 Metric

| Metric | 설명 |
| --- | --- |
| Overall Accuracy | 전체 정답률 |
| Category-wise Accuracy | category별 정답률 |
| Question-type Accuracy | reasoning type별 정답률 |

---

### 3.5 MLVU

**선정이유**

Long video understanding 평가에 적합한 benchmark임

Video-MME와 함께 긴 영상 기반의 comprehension, reasoning, summarization 성능을 확인하는 용도로 사용함

WorldSense / AV-SpeakerBench / OmniVideoBench가 Audio-Visual 특화 성능을 보는 역할이라면, MLVU는 **long-form video understanding baseline** 역할로 사용함

- Local path (gpfs): `/gpfs/public/datasets/MLVU/`

**특징 / 분포**

| 항목 | 내용 |
| --- | --- |
| 데이터 규모 | 2,593개 video |
| 총 영상 길이 | 약 3,000시간 |
| 문제 수 | 약 9,000개 QA / task instance |
| Task 유형 | multi-choice QA, summarization 등 |
| 주요 평가 대상 | long video comprehension, temporal reasoning, event understanding, summarization |
| 평가 포인트 | long-form video understanding, temporal dependency, multi-task reasoning |

출처: MLVU는 long video understanding 평가를 위해 구성된 benchmark로, 2,593개 video, 약 3,000시간 분량의 영상, 약 9,000개 task instance를 포함함

**평가 Metric**

| Metric | 설명 |
| --- | --- |
| Overall Accuracy | 객관식 QA 전체 정답률 |
| Task-wise Accuracy | task 유형별 정답률 |
| Generation Metric | summarization 등 generation task에서 benchmark가 정의한 공식 metric 사용 |

---

## 4. Unified Evaluation Runner

이 저장소의 루트 패키지는 `uv` 환경에서 실행되는 통합 evaluation runner를 제공한다. 모델은 먼저 `vllm serve`의 OpenAI-compatible API로 띄우고, runner는 각 benchmark의 official prompt/result protocol에 맞춘 adapter를 통해 결과를 `results/` 아래에 저장한다.

### 4.1 설치

```bash
uv sync
```

기본 의존성에 `vllm`이 포함되어 있어 `--serve` 옵션으로 runner가 직접 `vllm serve`를 띄울 수 있다.

`flash-attn`은 CUDA, PyTorch, Python, platform 조합에 맞는 wheel을 사용자 환경별로 설치한다. 예를 들어 현재 실험 환경이 `torch==2.11.0+cu130`, Python `3.12`, Linux `x86_64`라면 아래 wheel 조합을 사용할 수 있다.

```bash
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl"
```

현재 `submodules/` 아래 official repo들은 git submodule로 등록되어 있으므로 새 환경에서는 아래처럼 초기화한다.

```bash
git submodule update --init --recursive
```

### 4.2 설정 파일

모델 config는 모델별로 독립된 YAML 파일에 둔다.

- `models[].name`: 결과 디렉터리와 run summary에 사용할 모델 이름
- `models[].weight_path`: vLLM이 로드할 local weight 경로
- `models[].served_model_name`: vLLM `--served-model-name` 및 API `model` 값
- `global.result_dir`: 결과 저장 루트

기본 모델 config:

- `configs/models/qwen3_omni.yaml`
- `configs/models/nemotron_3_nano_omni.yaml`

benchmark config는 `configs/benchmarks/default.yaml`에 분리되어 있으며, 실행 시 별도 지정하지 않으면 이 default config를 사용한다.

- `benchmarks[].name`: 실행할 benchmark adapter 이름
- `benchmarks[].data_path`, `annotation_file`, `video_dir`: benchmark별 데이터 경로

WorldSense 데이터는 Hugging Face download 후 video zip 형태로 저장되므로 최초 실행 전에 한 번 압축을 해제한다.

```bash
bash scripts/extract_worldsense_videos.sh
```

### 4.3 Default Experiment Setting

- Serving backend: `vllm serve`
- API protocol: OpenAI-compatible chat completions
- Tensor parallel size: `1`
- Data parallel size: `auto` (available GPU count)
- Max generation tokens: `8192`
- Sampling temperature: `0`
- Benchmark config: `configs/benchmarks/default.yaml`

### 4.4 실행

vLLM serving은 별도 터미널에서 모델별 script로 실행한다.

```bash
bash scripts/serve_qwen3_omni.sh
```

Nemotron은 별도 script로 실행한다.

```bash
bash scripts/serve_nemotron_3_nano_omni.sh
```

vLLM 서버가 떠 있는 상태에서 evaluation runner를 실행한다.

```bash
uv run omni-bench run \
  --config configs/models/qwen3_omni.yaml \
  --benchmark av_speakerbench
```

`omni-bench run --serve`는 편의 옵션으로 남겨두지만, serving flag 튜닝과 로그 확인이 필요한 실험에서는 위처럼 script로 vLLM을 직접 띄우는 방식을 기본으로 사용한다.

기본 benchmark config가 아닌 다른 benchmark config를 쓰려면:

```bash
uv run omni-bench run \
  --config configs/models/qwen3_omni.yaml \
  --benchmark-config configs/benchmarks/default.yaml \
  --benchmark av_speakerbench
```

현재 config에서 생성되는 vLLM serve 명령만 확인하려면:

```bash
uv run omni-bench serve \
  --config configs/models/qwen3_omni.yaml \
  --model qwen3-omni \
  --print-only
```

### 4.5 결과 구조

각 benchmark 결과는 아래처럼 모델별로 저장된다.

```text
results/
  qwen3-omni/
    av_speakerbench/
      records.json
      records.jsonl
      summary.json
  nemotron-3-nano-omni/
    av_speakerbench/
      records.json
      records.jsonl
      summary.json
  run_summary.json
```

`Video-MME`는 official template과 같은 nested JSON을 `official_results.json`으로 저장한다. `WorldSense`는 official naming scheme인 `<prompting>___<modelname>___results.jsonl` 파일과 `official_analysis.txt`를 함께 저장한다.

### 4.6 지원 adapter

```bash
uv run omni-bench list-benchmarks
```

현재 지원 목록은 `av_speakerbench`, `omnivideobench`, `videomme`, `mlvu`, `worldsense`이다. `WorldSense` official repo는 standalone evaluator 대신 VLMEvalKit 재현을 안내하므로, 이 runner는 released multiple-choice annotation을 직접 평가한다. `MLVU` adapter는 official multiple-choice evaluation protocol을 우선 지원하며, summarization 등 generation task는 별도 official evaluator 연동이 필요하다.