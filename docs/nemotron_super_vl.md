# Nemotron 3.5 Super VL 평가

> **NDA** — NVIDIA Early Access 체크포인트입니다(GA 2026-10-15 예정). 모델 이름·스펙·결과를 사외로 공유하지 마십시오.

## 목적

NVIDIA Early Access로 받은 Nemotron 3.5 Super VL을 기존 후보(Qwen3.8-27B, Nemotron-3-Nano-Omni, Qwen3-Omni)와 같은 벤치마크로 비교합니다. 함께 확인한 항목은 다음과 같습니다.

- think / no-think 효과
- 양자화(BF16 vs NVFP4) 손실
- ASR 교체 효과(Whisper vs Nemotron ASR)

Super VL은 **audio encoder가 없는** vision + text 모델입니다. Qwen3.8과 같은 방식으로 audio를 Whisper transcript로 대체합니다(cascaded VLM + ASR).

> **결론 — 기존 open-model과 비슷한 수준이고, think에서도 크게 앞서지 않습니다.**
> - no-think는 Qwen3.8-27B·Nemotron-Omni와 대등합니다.
> - think 효과는 +1.6 ~ +6.6으로, Qwen3.8(+9.8 ~ +15.0)보다 작습니다.
> - NVFP4 양자화 손실은 ±1.2 이내로 무시할 수준입니다.

| 항목 | 값 |
| --- | --- |
| 모델 | Nemotron 3.5 Super VL — 120B MoE · active 12B · thinking / tool calling 지원 |
| 체크포인트 | GA 후보 빌드 2026-10-01 (`nemotron_3_5_super_ga_candidate_mtp_boosted[_nvfp4]_20261001_vv0.1`) |
| 입력 | text · image · video (**audio 없음**) |
| 서빙 | Platform 팀 원격 vLLM 0.31.0 (sel2 `vllm` 네임스페이스) · `max_model_len` 65,536 |
| 엔드포인트 | BF16 `http://vllm-nemotron-3-5-super-vl-bf16.vllm:8000/v1` (B200×2) · NVFP4 `http://vllm-nemotron-3-5-super-vl-nvfp4.vllm:8000/v1` (B200×1) |

`*.vllm.idc.k8s` 주소는 클러스터 Pod 안에서 DNS가 풀리지 않으므로 위의 클러스터 내부 이름을 씁니다.

---

## 1. 세팅

Qwen3.8 recommend config([qwen3.8_plan.md](qwen3.8_plan.md))와 같은 입력 조건을 유지했습니다. 바꾼 것은 모델 샘플링과, 원격 서버 제약 때문에 필요한 전송 방식뿐입니다.

### 샘플링

Super VL은 공개 모델 카드가 없으므로, 같은 계열인 Nemotron 3 Super의 공식 권장값을 씁니다(think·no-think 공통).

| 항목 | no-think | think |
| --- | --- | --- |
| `temperature` / `top_p` | 1.0 / 0.95 | 1.0 / 0.95 |
| 그 외 (`top_k`, `presence_penalty` 등) | 보내지 않음 (서버 기본값) | 동일 |
| `enable_thinking` | false | **true** |
| `max_tokens` | 4,096 | **32,768** (처음부터) |

think 응답은 서버의 reasoning parser가 `message.reasoning`과 `content`로 나눠 줍니다. `content`에는 답만 남으므로, Qwen3.8 think와 달리 **official parser로 바로 채점**합니다(`rescore_mcq.py` 불필요).

### 원격 서버 제약과 대응

| 제약 | 대응 |
| --- | --- |
| `file://` 경로를 읽지 못함 (`--allowed-local-media-path` 없음) | `inference.transport: base64` — 영상을 base64 data URL로 전송 |
| 요청 본문이 너무 큼 (Video-MME 최대 922MB) | 200MB 초과 파일만 ≤720p로 재인코딩해 캐시 (fps 유지, 오디오 제거). 대상: Video-MME 156개 · OmniVideoBench 87개 |
| `mm_processor_kwargs` / `media_io_kwargs`를 넣으면 400 | `inference.strip_mm_kwargs: true` — 해당 키를 요청에서 제거 |
| audio 입력 거부 | `inference.audio: asr_text` (Video-MME는 벤치 설정에서 `none`) |
| 요청별 프레임 설정 불가 | 서버 기본값 사용. **영상 1개당 약 4.6k 토큰으로 고정**됨 (vLLM 기본 32프레임으로 추정, 미확인) |

영상 입력량은 Qwen3.8(`inference.frames: server`, 약 10~13k 토큰)의 절반 이하입니다. 서버 설정(`--media-io-kwargs`)은 Platform 팀만 바꿀 수 있습니다.

### 그 외

| 항목 | 값 |
| --- | --- |
| `inference.frames` | `server` (벤치 설정 · Qwen3.8과 동일) |
| ASR | faster-whisper `large-v3` 기존 캐시 재사용 (`strict_cache: true`, 12,660건) |
| Video-MME | ASR 미주입 · 자막 미사용 (official 프로토콜) |
| OmniVideoBench system prompt | 어댑터 기본값 그대로 (Qwen3.8 · Nemotron-Omni 런과 동일 조건) |
| 동시성 | no-think 24 · think 48 (결과에는 영향 없음) |

---

## 2. 결과

모든 런에서 에러 0건입니다. 비교 모델의 출처는 다음과 같습니다.

- Nemotron-Omni · Qwen3-Omni: [report.html](../report.html)
- Qwen3.8: [qwen3.8_plan.md](qwen3.8_plan.md)

### 2.1 모델 비교 (BF16 · no-think)

| Benchmark | Super VL | Nemotron-Omni (BF16) | Qwen3.8-27B | Qwen3-Omni 30B |
| --- | --- | --- | --- | --- |
| Video-MME (w/o sub) | 67.30 | 67.81 | 66.89 | **70.19** |
| WorldSense | **52.21** | 50.63 | 47.48 | 51.36 |
| OmniVideoBench | 42.40 | 40.10 | **42.70** | 41.20 |
| AV-SpeakerBench | 47.42 | 50.28 | 50.34 | **55.98** |

AV-SpeakerBench가 비교군 중 가장 낮습니다. 화자를 판별하는 문항이 많아, transcript에 화자 정보가 없는 cascade 구성에 불리합니다.

### 2.2 think / no-think (BF16)

| Benchmark | Super VL no-think | Super VL think | Δ | Qwen3.8 no-think | Qwen3.8 think | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Video-MME | 67.30 | 69.19 | +1.89 | 66.89 | 77.33 | +10.44 |
| WorldSense | 52.21 | 55.42 | +3.22 | 47.48 | 61.32 | +13.84 |
| OmniVideoBench | 42.40 | 44.00 | +1.60 | 42.70 | 52.50 | +9.80 |
| AV-SpeakerBench | 47.42 | 54.02 | +6.60 | 50.34 | 65.35 | +15.01 |

think 효과가 Qwen3.8보다 훨씬 작습니다. 길게 추론할수록 정답률이 떨어졌습니다. WorldSense 기준으로 생성 300토큰 미만은 60.2%, 4k 토큰 이상은 37.0%입니다.

시각 입력이 약 4.6k 토큰으로 제한된 것이 원인 중 하나로 추정됩니다. 입력에 없는 장면은 추론을 길게 해도 보완되지 않습니다.

### 2.3 양자화 (Δ = NVFP4 − BF16)

| Benchmark | BF16 think | NVFP4 think | Δ | BF16 no-think | NVFP4 no-think | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Video-MME | 69.19 | 68.30 | −0.89 | 67.30 | 67.11 | −0.19 |
| WorldSense | 55.42 | 54.57 | −0.85 | 52.21 | 51.61 | −0.60 |
| OmniVideoBench | 44.00 | 45.20 | +1.20 | 42.40 | 42.50 | +0.10 |
| AV-SpeakerBench | 54.02 | 54.33 | +0.31 | 47.42 | 47.20 | −0.22 |

8개 항목 모두 ±1.2 이내입니다. NVFP4(B200×1)로 서빙해도 성능 차이가 거의 없습니다.

### 2.4 ASR 비교 (BF16 · no-think, Δ = Nemotron − Whisper)

| Benchmark | Whisper large-v3 | nemotron-3.5-asr-streaming-0.6b | Δ |
| --- | --- | --- | --- |
| WorldSense | 52.21 | 50.25 | −1.95 |
| OmniVideoBench | 42.40 | 41.10 | −1.30 |
| AV-SpeakerBench | 47.42 | 46.86 | −0.56 |

전사 품질을 정성 비교한 결과는 다음과 같습니다.

| 모델 | 강점 | 약점 |
| --- | --- | --- |
| Nemotron ASR | 환각, 엉뚱한 언어로의 번역, 긴 구간 누락이 없음 | 숫자를 영단어로 풀어 씀, 겹치는 발화·비영어(중국어 등)에 약함 |
| Whisper | 고유명사, 숫자, 다화자 대화에 강함 | 무음·음악 구간 환각("Thank you for watching" 등) |

Nemotron 캐시 중 4개 영상(OmniVideoBench `video_93/532/534/539`)은 타임스탬프 없이 생성됐습니다. 해당 문항은 1,000건 중 5건입니다.

### 잘림 (think, `finish_reason == length`, max_tokens 32,768)

| Benchmark | BF16 | NVFP4 |
| --- | --- | --- |
| Video-MME | 35 | 30 |
| WorldSense | 14 | 8 |
| OmniVideoBench | 13 | 18 |
| AV-SpeakerBench | 6 | 8 |

전체의 0.2~1.8%입니다. 잘린 건은 오답으로 집계했습니다(재측정 안 함).

### 비용

| 항목 | no-think | think |
| --- | --- | --- |
| 생성 토큰 (평균) | 2~15 | 1,500~2,900 |
| 총 소요 | 구성당 약 5~6시간 | 구성당 약 12~14시간 |

병목은 서버 API 프로세스의 **CPU 영상 디코딩**입니다. 특히 Video-MME long 영상(30~60분)은 문항당 약 4분이 걸렸습니다.

- 동시성을 올려도 GPU까지 가는 요청 수가 늘지 않습니다. 처리 중 요청은 20~40건이었고 KV 캐시 사용률은 10% 미만이었습니다.
- 해결하려면 서버 쪽 `--api-server-count`를 늘려야 합니다(Platform 팀 설정).

---

## 3. 구현 (`qwen-3.8` 대비 변경)

채점 로직(답 추출 · 정확도 집계 · 리포트)은 `main`과 같습니다. 변경은 요청 전송과 실행 편의 기능에 한정됩니다.

| 기능 | 파일 | 설명 |
| --- | --- | --- |
| base64 영상 전송 | `src/omni_bench/inference/transport.py`, `src/omni_bench/video_transport.py` | model config `inference.transport: file \| base64`. base64일 때 `transcode:` 기준을 넘는 파일은 재인코딩 캐시를 씀. 캐시 키는 경로·크기·mtime·재인코딩 파라미터이고, 동시 접근에 안전함 |
| 재인코딩 사전 생성 | `scripts/pretranscode_videos.py` | 평가 전에 캐시를 병렬로 미리 생성. 캐시가 없으면 클라이언트가 그 자리에서 생성 |
| mm kwargs 제거 | `src/omni_bench/inference/__init__.py` | `inference.strip_mm_kwargs: true`면 `mm_processor_kwargs`·`media_io_kwargs`를 요청에서 삭제 |
| reasoning 기록 | `src/omni_bench/client.py`, `adapters/*` | `reasoning` · `finish_reason` · 토큰 수를 레코드에 저장 |
| 에러 행 재시도 | `src/omni_bench/io.py`, `adapters/*` | 요청 하나가 실패해도 벤치 전체가 멈추지 않고 에러 행으로 기록. 재실행하면 에러 행만 다시 돌림 |
| 실행 옵션 | `src/omni_bench/cli.py` | `--limit N`, `--limit-mode head\|spread`(전체에서 고르게 N개), `--result-dir`, 실행 설정 스냅샷 `config_used.json`(git 커밋 포함) |
| 잘림 정리 | `scripts/strip_truncated.py` | `--run-dir` 지원, `finish_reason == length` 기준 판정 |
| Nemotron ASR | `src/omni_bench/asr/nemo_streaming.py` | 캐시만 읽는 ASR 전략 `nemo_streaming`. 캐시가 없으면 에러 |

---

## 4. 실행

```bash
export UV_PROJECT_ENVIRONMENT=/gpfs/private/ail/venvs/omni-bench

# 0. (선택) 대용량 영상 재인코딩 캐시 사전 생성
uv run --no-sync python scripts/pretranscode_videos.py \
    --config configs/nemotron_super_vl/bf16_nothink.yaml \
    --video-dir /gpfs/public/datasets/Video-MME/data \
    --video-dir /gpfs/public/datasets/OmniVideoBench/videos

# 1. 스모크 — 벤치당 20건을 전체에서 고르게
uv run --no-sync omni-bench run --config configs/nemotron_super_vl/bf16_nothink.yaml \
    --benchmark-config configs/nemotron_super_vl/bench_nothink.yaml --benchmark worldsense \
    --limit 20 --limit-mode spread --result-dir /tmp/smoke

# 2. 본 실행 — 측정 config 파일 하나 = 런 하나. 벤치는 --benchmark로 하나씩 (생략 시 벤치 설정 전체)
#    model config: {bf16,nvfp4}_{nothink,think}.yaml / bench config: bench_{nothink,think}.yaml
uv run --no-sync omni-bench run --config configs/nemotron_super_vl/bf16_nothink.yaml \
    --benchmark-config configs/nemotron_super_vl/bench_nothink.yaml --benchmark worldsense

# 3. ASR 비교 (Nemotron ASR 캐시, BF16 · no-think, Video-MME 제외)
uv run --no-sync omni-bench run --config configs/nemotron_super_vl/bf16_nothink_nemotron_asr.yaml \
    --benchmark-config configs/nemotron_super_vl/bench_nothink.yaml --benchmark worldsense
```

원격 서버 전용이므로 `--serve`는 쓰지 않습니다. 측정 당시에는 사외 통신을 막기 위해 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`을 두고, 벤치 4종(videomme · worldsense · omnivideobench · av_speakerbench)을 위 명령으로 순차 실행했습니다. 긴 런은 `nohup … > <log> 2>&1 &`로 백그라운드에서 돌립니다.

| config | 용도 |
| --- | --- |
| `configs/models/nemotron_3_5_super_vl_{bf16,nvfp4}.yaml` | 모델 기본 설정 (엔드포인트 · 전송 · 샘플링 · Whisper ASR 캐시). 기본은 no-think |
| `configs/nemotron_super_vl/{bf16,nvfp4}_{nothink,think}.yaml` | 측정에 쓴 설정 (Qwen3.8의 `configs/recommend/`에 해당). 위 모델 설정과 요청 내용이 같고 결과 저장 위치와 think 여부만 다름 |
| `configs/nemotron_super_vl/bench_{nothink,think}.yaml` | 벤치 설정 (max_tokens · temperature · 동시성) |
| `configs/nemotron_super_vl/bf16_nothink_nemotron_asr.yaml` | ASR만 `nemo_streaming`으로 바꾼 설정 |

`configs/models/`의 모델 설정과 측정에 쓴 설정이 같은 요청을 만드는지 확인했습니다. OpenAI SDK를 스텁으로 바꿔 실제 요청 본문을 비교했고, BF16·NVFP4 모두 동일했습니다.

같은 명령을 다시 실행하면 끝난 문항은 건너뛰고 에러 행만 다시 돕니다.

### 산출물

```text
/gpfs/public/artifacts/ail/nemotron-compare/
  README.md · summary.json                          전체 결과 요약 (기계 판독용 JSON 포함)
  omni-bench/
    nemotron-3.5-super-vl-bf16-nothink/              2.1 · 2.2 · 2.3 · 2.4(Whisper)
    nemotron-3.5-super-vl-bf16-think/                2.2 · 2.3
    nemotron-3.5-super-vl-nvfp4-nothink/             2.3
    nemotron-3.5-super-vl-nvfp4-think/               2.3
    nemotron-3.5-super-vl-bf16-nothink-nemotron-asr/ 2.4(Nemotron ASR)
      <benchmark>/records.jsonl · summary.json · config_used.json
    logs/ · _smoke/
  cache/transcode/                                   200MB 초과 영상 재인코딩본
```

ASR 캐시(공용):

- Whisper: `/gpfs/public/artifacts/ail/omni-bench/cache/asr/faster_whisper__Systran__faster-whisper-large-v3/`
- Nemotron: `/gpfs/public/artifacts/ail/omni-bench/cache/asr/nemo_streaming__nvidia__nemotron-3.5-asr-streaming-0.6b/`
