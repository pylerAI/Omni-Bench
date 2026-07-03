# Omni-Bench

Omni-Bench는 omni-modal model을 평가하기 위한 all-in-one evaluation pipeline입니다. vLLM compatible endpoint로 모델을 serving하고, audio-visual 및 long-video benchmark를 동일한 runner에서 실행할 수 있도록 구성합니다.

초기 평가 대상은 Qwen3-Omni와 Nemotron-3-Nano-Omni이며, 이후 다른 omni model과 benchmark도 YAML config와 adapter module을 통해 확장할 수 있습니다.

## Update

- Initial commit

## 문서

- [평가 계획](docs/plan.md)
- [AV-SpeakerBench](docs/av_speakerbench.md)
- [WorldSense](docs/worldsense.md)
- [Video-MME](docs/videomme.md)
- [OmniVideoBench](docs/omnivideobench.md)
- [OmniDCBench](docs/omnidcbench.md)

## 프로젝트 구성

```text
configs/
  benchmarks/default.yaml
  models/qwen3_omni.yaml
  models/nemotron_3_nano_omni.yaml
Docs/
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

사용 가능한 adapter 목록 확인:

```bash
uv run omni-bench list-benchmarks
```

## 결과

결과는 아래 경로에 저장됩니다.

```text
results/<model-name>/<benchmark-name>/
```

기본 출력 파일:

- `records.json`
- `records.jsonl`
- `summary.json`

일부 adapter는 official evaluator에 넣을 수 있는 별도 파일도 생성합니다. 예를 들어 Video-MME는 `official_results.json`, OmniDCBench는 `predictions.jsonl`을 저장합니다.
