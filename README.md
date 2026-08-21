# Omni-Bench

**Omni-Bench는 omni-modal model 평가에 집중한 all-in-one evaluation pipeline입니다.**

Omni-Bench는 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)과 유사하게 여러 benchmark를 하나의 runner에서 실행하고 결과를 통일된 위치에 저장하는 것을 목표로 합니다. 다만 범용 VLM evaluation toolkit이 아니라, **audio-video-text를 함께 처리하는 omni model과 omni benchmark 평가에 집중**합니다. 불필요한 범용 구현은 줄이고, vLLM serving 기반의 실험 반복과 benchmark별 official protocol 추적을 쉽게 하는 데 초점을 둡니다.

[평가 계획](docs/plan.md) · [Qwen3.8-27B 계획](docs/qwen3.8_plan.md) · [환경 설정](#환경-설정) · [Serving](#serving) · [평가 실행](#평가-실행) · [결과](#결과)

## 목표

이 프로젝트가 지향하는 것은 다음과 같습니다.

1. Omni model 평가를 위한 공통 실행 인터페이스 제공
2. vLLM-compatible endpoint 기반의 반복 가능한 benchmark 실행
3. benchmark별 official prompt, parser, result format을 최대한 유지
4. 모델별 config와 benchmark별 config 분리
5. 결과를 `records.jsonl`, `summary.json`, `overall_report.md` 형태로 정리

## 지원 모델과 Benchmark

초기 평가 대상 모델:

| Model | Config |
| --- | --- |
| Qwen3-Omni-30B-A3B-Instruct | `configs/models/qwen3_omni.yaml` |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 | `configs/models/nemotron_3_nano_omni.yaml` |
| Qwen3.8-27B + Whisper ([평가](docs/qwen3.8_plan.md) · [구현](docs/qwen3_8_whisper.md)) | `configs/recommend/*.yaml` |

지원 benchmark:

| Benchmark | 문서 | Adapter |
| --- | --- | --- |
| AV-SpeakerBench | [docs/av_speakerbench.md](docs/av_speakerbench.md) | `av_speakerbench` |
| WorldSense | [docs/worldsense.md](docs/worldsense.md) | `worldsense` |
| Video-MME | [docs/videomme.md](docs/videomme.md) | `videomme` |
| OmniVideoBench | [docs/omnivideobench.md](docs/omnivideobench.md) | `omnivideobench` |
| OmniDCBench | [docs/omnidcbench.md](docs/omnidcbench.md) | `omnidcbench` |

## Audio modality

Audio encoder가 없는 모델은 model config의 `audio_mode`로 audio 처리를 바꿉니다. adapter는 수정하지 않습니다.

| `audio_mode` | 동작 |
| --- | --- |
| `native` (기본) | 모델에 audio를 그대로 전달 — omni 모델 |
| `none` | audio를 제거 — vision-only baseline |
| `asr_text` | Whisper로 전사해 프롬프트에 주입 — cascade |

`asr_text`를 쓰기 전에 선행 전사를 돌려두면 평가가 Whisper에 블로킹되지 않습니다.

```bash
uv run python scripts/prepare_asr.py --asr-config configs/asr/whisper_large_v3.yaml
```

자세한 내용은 [docs/qwen3_8_whisper.md](docs/qwen3_8_whisper.md)를 참고하세요.

## Frame sampling

benchmark config의 `frame_sampling`으로 프레임 추출 주체를 고릅니다.

| 값 | 동작 |
| --- | --- |
| `client` (기본) | harness가 프레임을 뽑아 `image_url`로 전송 |
| `server` | 원본 영상을 넘겨 모델 프로세서가 결정 |

## 프로젝트 구조

```text
configs/
  benchmarks/default.yaml
  models/qwen3_omni.yaml
  models/nemotron_3_nano_omni.yaml
  recommend/                      Qwen3.8-27B recommend config (thinking / non-think)
  asr/whisper_large_v3.yaml
docs/
scripts/
  serve_qwen3_omni.sh
  serve_qwen3_8_27b.sh
  serve_nemotron_3_nano_omni.sh
  prepare_asr.py                  선행 ASR 전사 (GPU 분산 · 재개 가능)
  strip_truncated.py              잘린 thinking 레코드 제거
  rescore_mcq.py                  저장된 응답으로 MCQ 재채점
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

## 환경 설정

```bash
uv sync
git submodule update --init --recursive
```

`vllm[audio]`는 프로젝트 기본 dependency에 포함되어 있습니다.

`flash-attn`은 CUDA, PyTorch, Python, platform 조합에 맞는 wheel을 환경별로 설치해야 합니다. 현재 Linux x86_64, Python 3.12, torch 2.11.0+cu130 환경에서는 아래 wheel을 사용할 수 있습니다.

```bash
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl"
```

## 데이터 준비

일부 dataset은 archive 형태로 배포되므로 평가 전에 압축 해제가 필요합니다.

```bash
bash scripts/extract_worldsense_videos.sh
bash scripts/extract_omnidcbench_videos.sh
```

기본 local dataset 경로는 다음과 같습니다.

| Benchmark | Local path |
| --- | --- |
| AV-SpeakerBench | `/gpfs/public/datasets/AV-SpeakerBench/` |
| WorldSense | `/gpfs/public/datasets/WorldSense/` |
| Video-MME | `/gpfs/public/datasets/Video-MME/` |
| OmniVideoBench | `/gpfs/public/datasets/OmniVideoBench/` |
| OmniDCBench | `/gpfs/public/datasets/OmniDCBench/` |

## QuickStart

1. vLLM 서버를 실행합니다.

```bash
bash scripts/serve_qwen3_omni.sh
```

2. 원하는 benchmark를 실행합니다.

```bash
uv run omni-bench run \
  --config configs/models/qwen3_omni.yaml \
  --benchmark av_speakerbench
```

3. 결과를 확인합니다.

```text
results/qwen3-omni/av_speakerbench/
```

## Serving

vLLM은 별도 터미널에서 먼저 실행합니다.

```bash
bash scripts/serve_qwen3_omni.sh
```

또는:

```bash
bash scripts/serve_nemotron_3_nano_omni.sh
```

기본 serving 설정:

- Tensor parallel size: `1`
- Data parallel size: 사용 가능한 GPU 수
- GPU memory utilization: `0.9`
- Allowed local media path: `/gpfs/public/datasets`
- Max audio decode duration: `3600`초

## 평가 실행

단일 benchmark 실행:

```bash
uv run omni-bench run \
  --config configs/models/qwen3_omni.yaml \
  --benchmark av_speakerbench
```

default benchmark config에 활성화된 전체 benchmark 실행:

```bash
uv run omni-bench run \
  --config configs/models/qwen3_omni.yaml
```

별도 benchmark config 사용:

```bash
uv run omni-bench run \
  --config configs/models/qwen3_omni.yaml \
  --benchmark-config configs/benchmarks/default.yaml
```

Qwen3.8-27B + Whisper — recommend config (`frame_sampling: server` + Qwen 권장 샘플링):

```bash
# thinking
uv run omni-bench run \
  --config configs/recommend/qwen3_8_27b_whisper_thinking.yaml \
  --benchmark-config configs/recommend/bench_thinking.yaml \
  --benchmark worldsense

# non-thinking
uv run omni-bench run \
  --config configs/recommend/qwen3_8_27b_whisper_nothink.yaml \
  --benchmark-config configs/recommend/bench_nothink.yaml \
  --benchmark worldsense

# Video-MME 는 ASR 을 주입하지 않아 별도 config
uv run omni-bench run \
  --config configs/recommend/qwen3_8_27b_videomme_thinking.yaml \
  --benchmark-config configs/recommend/bench_videomme_thinking.yaml \
  --benchmark videomme
```

thinking 런은 출력이 `max_tokens`에 걸리면 최종 답이 유실되고, official parser(첫 `[ABCD]`)로는 채점할 수 없습니다.

```bash
python scripts/strip_truncated.py all   # 잘린 레코드 제거 → 위 명령 재실행하면 그 건만 돈다
python scripts/rescore_mcq.py <records> # 저장된 응답으로 재채점
```

사용 가능한 adapter 목록 확인:

```bash
uv run omni-bench list-benchmarks
```

## 결과

프로젝트 최상위의 [`report.html`](report.html)은 전체 benchmark 결과를 chart/table로 정리한 리포트입니다. **2026-07-08 기준으로 측정된 결과**입니다 (Qwen3-Omni-30B-A3B-Instruct, Nemotron-3-Nano-Omni-30B-A3B-Reasoning FP8/BF16/NVFP4). `omni-bench run`이 매 실행마다 `results/report.html`을 자동 생성하며, 최상위 `report.html`은 그 시점의 스냅샷입니다.

결과는 아래 경로에 저장됩니다.

```text
results/<model-name>/<benchmark-name>/
```

기본 출력 파일:

- `records.json`
- `records.jsonl`
- `summary.json`

일부 adapter는 official evaluator에 넣을 수 있는 별도 파일도 생성합니다. 예를 들어 Video-MME는 `official_results.json`, OmniDCBench는 `predictions.jsonl`을 저장합니다.

전체 benchmark 결과는 아래 파일로 취합됩니다.

- `results/run_summary.json`
- `results/overall_report.json`
- `results/overall_report.md`

## 개발 가이드

새 benchmark를 추가할 때는 다음 순서를 따릅니다.

1. `src/omni_bench/adapters/`에 adapter 구현
2. `src/omni_bench/adapters/__init__.py`에 adapter 등록
3. `configs/benchmarks/default.yaml` 또는 별도 benchmark config에 항목 추가
4. `docs/<benchmark_name>.md`에 protocol, metric, output table 정리

새 모델을 추가할 때는 `configs/models/` 아래 model config를 추가하고, 필요하면 `scripts/serve_<model>.sh`를 작성합니다.
