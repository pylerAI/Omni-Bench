# Qwen3.8-27B + Whisper 평가

## 목적

AAII bench에서 Qwen3.8-27B가 52점으로 GPT-5.6-Luna와 동급의 언어 성능을 보였습니다. 같은 모델의 멀티모달 능력이 omni 전용 모델을 넘어서는지 확인하고, 넘어선다면 다운스트림(AiD Video) 백본 교체를 검토합니다.

Qwen3.8-27B는 vision + text 모델로 **audio encoder가 없습니다**. audio를 Whisper transcript로 대체하므로, 이 실험은 **native omni vs cascaded VLM + ASR** 의 비교입니다.

> **결론 — thinking을 켜면 교체 가치가 있다.** thinking에서 Qwen3-Omni-30B를 4종 모두 앞서고 Qwen3.5-Omni-Flash급이 됩니다. 단 **이득은 전부 thinking에서 나옵니다** — non-thinking은 대등하거나 뒤집니다. 대가는 실행 시간 4~25배입니다.

| 항목 | 값 |
| --- | --- |
| 아키텍처 | `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) |
| text 백본 | 64층 하이브리드 (linear attention 3 : full attention 1) · hidden 5120 · GQA 24/4 |
| 컨텍스트 | 262,144 (mRoPE interleaved) |
| vision tower | 27층 · hidden 1152 · patch 16 · out 5120 |
| 파라미터 형태 | **dense 27B** (비교 대상 Qwen3-Omni는 MoE 30B-A3B, active 3B) |
| audio | 없음 — Whisper large-v3 cascade로 대체 |
| 웨이트 | `/gpfs/public/artifacts/models/Qwen/Qwen3.8-27B/` |

구현 상세는 [Qwen3.8-27B + Whisper (cascaded ASR)](qwen3_8_whisper.md)를 참고하세요.

---

## 1. 세팅 — recommend config

Qwen 권장 설정만 사용합니다. 이전 evalkit 기반(프레임 고정 · `temperature 0`)이 아닙니다.

### 샘플링

| 항목 | non-think | thinking |
| --- | --- | --- |
| `temperature` | 0.7 | 1.0 |
| `top_p` / `top_k` | 0.80 / 20 | 0.95 / 20 |
| `min_p` | 0.0 | 0.0 |
| `presence_penalty` | 1.5 | 0.0 |
| `repetition_penalty` | 1.0 | 1.0 |
| `enable_thinking` | false | **true** |
| `max_tokens` | 4,096 | **32,768** |

`max_tokens`는 thinking에서 8,192로 시작했다가 **4.4~13.9%가 잘려**(추론 중 끊겨 최종 답 유실) 32,768로 올려 전량 재측정했습니다 → 잘림 0.

### 입력 처리 — `frame_sampling: server`

프레임을 클라이언트에서 뽑지 않고 **원본 영상을 그대로 보내** 모델 프로세서가 결정하게 합니다. 값은 모델 웨이트의 `video_preprocessor_config.json` 그대로입니다.

| 항목 | 값 | 의미 |
| --- | --- | --- |
| `fps` | 2 | 초당 2장 추출 |
| `max_frames` | 768 | 프레임 수 상한 — 384초(6.4분) 넘으면 걸린다 |
| `min_frames` | 4 | 하한 |
| `size.longest_edge` | 25,165,824 px | **비디오 전체** 픽셀 예산 (프레임당이 아니다) |
| `patch_size` / `merge_size` | 16 / 2 | 토큰 1개 = 32×32 px |
| `temporal_patch_size` | 2 | 인접 2프레임을 3D 패치로 묶음 → 토큰 절반 |

픽셀 예산이 비디오 전체에 걸립니다. 해상도로 환산하면:

| 해상도 | 프레임당 px | 예산에 들어가는 프레임 | 프레임당 토큰 |
| --- | --- | --- | --- |
| 1280×720 (720p 원본) | 921,600 | **27장** | 450 |
| 854×480 (480p) | 409,920 | 61장 | 200 |
| 640×360 (360p) | 230,400 | 109장 | 112 |
| 480×256 | 122,880 | 204장 | 60 |
| 224×128 | 28,672 | 877장 | 14 |

**720p를 지키면 27장밖에 못 보고, 768장을 채우려면 224×128까지 줄여야 합니다.** 프로세서는 후자를 택합니다 — 공간 해상도를 팔아 시간 해상도를 삽니다. 실제 측정값(720p 원본):

| 영상 길이 | 요청 프레임 (fps 2) | 실제 프레임 | 프레임 해상도 | 비디오 토큰 |
| --- | --- | --- | --- | --- |
| 97초 | 194 | 194 | 256×480 | 11,640 |
| 500초 | 1,000 | 768 (상한) | 128×224 | 10,752 |
| 2,820초 | 5,639 | 768 (상한) | 128×224 | 10,752 |

- **6.4분을 넘으면 프레임 수가 멈추고 해상도만 계속 깎입니다** — 500초와 2,820초의 해상도가 같습니다
- 이전 evalkit 방식은 프레임 64장 고정 · 프레임당 `max_pixels 602,112`(약 768×720) · `image_url` 64개 전송(각 이미지가 독립이라 `temporal_patch` 병합 이득 0) → prompt 34,141토큰
- server 방식은 prompt **13,734토큰**(40%) · wall **2.2배 빠름** · **장문 응답 30.9% → 0.2%로 파싱 결함 소멸**(공식/개선 파서 차이 +12.30 → +0.07)

### 그 외

| 항목 | 값 |
| --- | --- |
| 서빙 | `MAX_MODEL_LEN` 131,072 · B200×4 (DP=4) · TP=1 · `gpu_memory_utilization` 0.9 |
| 동시성 | 12 |
| ASR | faster-whisper `large-v3` · fp16 · `beam_size 5` · `vad_filter true` · language auto · **task = transcribe** |
| Video-MME | **ASR 미주입** — official 프로토콜이 audio를 입력으로 쓰지 않고, 비교 모델도 같은 조건 |

`translate`는 omni 모델에 없는 번역 단계를 cascade에만 주게 되어 사용하지 않습니다.

---

## 2. 결과

| Benchmark | 우리 모델 | non-think | thinking | Qwen3-Omni 30B | Qwen3.5-Omni Flash | Qwen3.5-Omni Plus | Gemini |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Video-MME (w/o sub) | Qwen3.8-27B | 66.89 | **77.33** | 70.5 | 77.0 | 81.9 | 79.6 (2.5-Flash-Thinking) |
| WorldSense | Qwen3.8-27B + Whisper | 47.48 | **61.32** | 54.0 | 57.9 | 62.8 | 65.5 (3.1 Pro) |
| OmniVideoBench | Qwen3.8-27B + Whisper | 42.70 | **52.50** | 38.40 | — | — | 57.83 (2.5-Pro) |
| AV-SpeakerBench | Qwen3.8-27B + Whisper | 50.34 | **65.35** | — | 65.2 | 71.3 | 75.1 (3.1 Pro) |

Qwen3.5-Omni와 Gemini는 **웨이트 비공개 · 유료 API**라 백본 후보가 아닙니다. 교체 가능한 상대는 **Qwen3-Omni-30B** 하나이고, thinking에서 4종 모두 이깁니다.

### 이득은 전부 thinking에서 나온다

| Benchmark | non-think | thinking | Δ |
| --- | --- | --- | --- |
| Video-MME | 66.89 | **77.33** | **+10.44** |
| WorldSense | 47.48 | **61.32** | **+13.84** |
| OmniVideoBench | 42.70 | **52.50** | **+9.80** |
| AV-SpeakerBench | 50.34 | **65.35** | **+15.01** |

변수를 하나씩 바꿔 순효과를 분리했습니다.

| 변수 | 효과 |
| --- | --- |
| 샘플링 파라미터 (temp 0 → 권장값) | −0.13 ~ +2.90 → **무효** |
| frame sampling (고정 → 서버 기본값) | −1.00 ~ +0.15 → **무효** |
| **thinking** | **+9.80 ~ +15.01** |

**Qwen3-Omni는 thinking이 오히려 손해입니다** — Video-MME 70.5 → 69.7(−0.8) · LVBench 50.2 → 49.0 · MLVU 75.2 → 72.9. active 3B MoE와 dense 27B의 차이로 보입니다. **추론 시간을 성능으로 바꿀 수 있는지가 두 모델의 갈림길입니다.**

### 비용

| Benchmark | n | non-think | thinking | 배수 | 생성 토큰 |
| --- | --- | --- | --- | --- | --- |
| Video-MME | 2,700 | 20.5분 | 138.9분 | 6.8× | 2.5 → 1,891 |
| WorldSense | 3,172 | 21.8분 | 127.4분 | 5.8× | 5.2 → 1,869 |
| OmniVideoBench | 1,000 | 25.6분 | 86.7분 | 3.4× | 2.0 → 3,914 |
| AV-SpeakerBench | 3,212 | 8.1분 | 206.1분 | **25.4×** | 2.0 → 3,667 |

에러 0건. GPU 가동률은 약 8% — 병목이 vLLM APIServer의 **CPU 비디오 디코딩**입니다(NVDEC 미사용). 동시성을 12→48로 올리면 처리량이 절반으로 떨어집니다(스래싱).

### 비교 수치 출처

- Qwen3-Omni 30B — [Qwen3-Omni Technical Report](https://arxiv.org/html/2509.17765) Table 9(Video-MME) · Table 11(WorldSense) · Table 10(Instruct vs Thinking 대조)
- Qwen3.5-Omni Flash · Plus — [Qwen3.5-Omni Technical Report](https://arxiv.org/html/2604.15804v1) Table 6(Video-MME) · Table 7(WorldSense · AV-SpeakerBench)
- Gemini — 위 두 report의 같은 표, OmniVideoBench만 [OmniVideoBench](https://arxiv.org/html/2510.10689) Table 3
- AV-SpeakerBench의 Qwen3-Omni 수치는 없습니다 — report(2025-09)보다 벤치 공개(2025-12)가 늦습니다

추론 모드는 **Qwen3-Omni만 non-thinking 확정**(Instruct 변형)이고, Qwen3.5-Omni·Gemini는 보고서에 표기가 없습니다. Gemini 열은 행마다 세대가 달라 세로 비교가 불가하며 각 벤치 최고 버전 하나만 기입했습니다.

### 미측정 — OmniDCBench

thinking 출력의 **60.8%가 잘려**(긴 캡션 JSON + 추론이 `max_tokens`를 함께 소비) 보완이 미완입니다. 체크포인트 712/1,122가 보존되어 있습니다. non-thinking 기준으로는 F1 0.483이며, transcript가 이 벤치에서만 부호가 음(−)입니다 — audio 89%가 중국어인데 영어 caption과 timestamp JSON을 요구하기 때문입니다.

---

## 3. 실행

```bash
export UV_PROJECT_ENVIRONMENT=/gpfs/private/ail/venvs/omni-bench
export ALLOWED_LOCAL_MEDIA_PATH=/gpfs/public
export OMNI_BENCH_RESULT_DIR=/gpfs/public/artifacts/ail/omni-bench/runs/results

# 1. ASR 선행 전사 — vLLM과 시간축 분리 필수 (GPU 공유 시 EngineCore 사망)
uv run python scripts/prepare_asr.py \
    --asr-config configs/asr/whisper_large_v3.yaml \
    --gpus 0,1,2,3 --threads-per-gpu 8

# 2. 서빙
MAX_MODEL_LEN=131072 VLLM_BIN=$UV_PROJECT_ENVIRONMENT/bin/vllm \
    bash scripts/serve_qwen3_8_27b.sh

# 3. 평가 (bench = videomme | worldsense | omnivideobench | av_speakerbench)
uv run omni-bench run --config configs/recommend/qwen3_8_27b_whisper_thinking.yaml \
    --benchmark-config configs/recommend/bench_thinking.yaml   --benchmark <bench>   # thinking
uv run omni-bench run --config configs/recommend/qwen3_8_27b_whisper_nothink.yaml \
    --benchmark-config configs/recommend/bench_nothink.yaml --benchmark <bench>   # non-think

# 4. 잘린 건 보완 — 제거 후 3번 재실행하면 그 건만 다시 돈다
python scripts/strip_truncated.py all
```

### 산출물

```text
/gpfs/public/artifacts/ail/omni-bench/runs/
  results/qwen3.8-27b-srv-think/            Video-MME thinking
  results/qwen3.8-27b-srv-nothink/          Video-MME non-think
  results/qwen3.8-27b-whisper-srvthink/     4종 thinking
  results/qwen3.8-27b-whisper-srvnothink/   4종 non-think
  results/qwen3-omni-repro/                 Qwen3-Omni 재현 (비교 기준)
  archive/                                  탐색·중간 실험 산출물
```

각 `<benchmark>/records.jsonl` + `summary.json`. `summary.json`의 `perf` 블록에 `wall_s` · `samples_per_s` · `latency_s`(mean/p50/p90/p99/max) · `prompt_tokens`·`completion_tokens` 분포가 기록됩니다. `wall_s`는 frame 디코딩과 transcript 조회를 포함해 재실행 시간을 예측하는 값이고, `latency_s`는 서버 큐 대기가 섞여 같은 런 안에서만 비교합니다.

ASR 캐시 `/gpfs/public/artifacts/ail/omni-bench/cache/asr/` 12,660건은 재사용 가능합니다.
