# Qwen3.8-27B + Whisper 평가 계획

## 목적

AAII bench에서 Qwen3.8-27B가 52점으로 GPT-5.6-Luna와 동급의 언어 성능을 보였습니다. 이 계획은 같은 모델의 멀티모달 능력이 omni 전용 모델을 넘어서는지 확인하고, 넘어선다면 다운스트림 태스크의 백본 교체를 검토하는 것을 목표로 합니다.

비교 대상은 [평가 계획](plan.md)의 기존 런과 동일한 4개 결과입니다.

- Qwen3-Omni-30B-A3B-Instruct
- Nemotron-3-Nano-Omni-30B-A3B-Reasoning (FP8 / BF16 / NVFP4)

판단 기준은 audio 채널이 실제로 사용되는 4개 benchmark(AV-SpeakerBench, WorldSense, OmniVideoBench, OmniDCBench)에 둡니다. Video-MME는 현재 구현에서 양쪽 모두 audio를 사용하지 않으므로 frame 기반 VLM 기본기 비교 항목으로 읽습니다.

## 모델

| Model | Local Weight Path |
| --- | --- |
| Qwen3.8-27B | `/gpfs/public/artifacts/models/Qwen/Qwen3.8-27B/` |
| Whisper large-v3 (ASR) | `Systran/faster-whisper-large-v3` (HF cache) |

Qwen3.8-27B는 vision + text 모델입니다. `image_token_id`와 `video_token_id`는 있으나 **audio encoder가 없습니다**.

| 항목 | 값 |
| --- | --- |
| 아키텍처 | `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) |
| text 백본 | 64층 하이브리드 (linear attention 3 : full attention 1) · hidden 5120 · GQA 24/4 |
| 컨텍스트 | 262,144 (mRoPE interleaved, section 11/11/10) |
| vision tower | 27층 · hidden 1152 · patch 16 · out 5120 |
| 파라미터 형태 | **dense** (비교 대상 두 omni 모델은 MoE 30B-A3B, active 3B) |
| audio | 없음 — Whisper cascade로 대체 |

따라서 이 실험은 **native omni model** 과 **cascaded VLM + ASR** 의 비교입니다.

## Benchmark

기존 런과 동일한 5개 benchmark, 동일한 dataset, 동일한 입력 설정을 사용합니다.

| Benchmark | 문서 | 샘플 | 대표 metric |
| --- | --- | --- | --- |
| AV-SpeakerBench | [av_speakerbench.md](av_speakerbench.md) | 3,212 | Accuracy |
| WorldSense | [worldsense.md](worldsense.md) | 3,172 | Accuracy |
| Video-MME | [videomme.md](videomme.md) | 2,700 | Accuracy |
| OmniVideoBench | [omnivideobench.md](omnivideobench.md) | 1,000 | Accuracy |
| OmniDCBench | [omnidcbench.md](omnidcbench.md) | 1,122 | F1 · mIoU · SODA_M |

## 평가 프로토콜

runner 구조와 benchmark별 official prompt/parser는 기존과 동일합니다. 모든 adapter가 `VllmChatClient.complete()` 하나만 호출하므로, audio 처리는 client 계층에서만 교체하고 **adapter 코드는 수정하지 않습니다**. benchmark별 frame sampling 설정도 `configs/benchmarks/default.yaml`을 공유해 그대로 적용됩니다.

구현 상세는 [Qwen3.8-27B + Whisper (cascaded ASR)](qwen3_8_whisper.md)를 참고하세요.

### audio_mode

model config와 benchmark config가 `audio_mode`로 audio 처리를 지정합니다. benchmark 값이 model 값을 덮어씁니다.

| `audio_mode` | 동작 |
| --- | --- |
| `native` | audio를 모델에 그대로 전달 (기존 omni 런) |
| `none` | audio를 제거 |
| `asr_text` | Whisper transcript를 prompt에 주입 |

### benchmark별 audio 경로

| Benchmark | 비주얼 입력 | audio 소스 | 처리 |
| --- | --- | --- | --- |
| AV-SpeakerBench | `video_path` (원본 영상) | 영상 내 audio track (`use_audio_in_video: true`) | ffmpeg demux 후 전사 |
| OmniDCBench | `video_path` (원본 영상) | 영상 내 audio track (`use_audio_in_video: true`) | ffmpeg demux 후 전사 |
| WorldSense | `image_urls` (frame 8장) | `audio_path` (별도 WAV) | 그대로 전사 |
| OmniVideoBench | `video_url` (2.0 fps, 최대 120 frame) | `audio_path` (별도 WAV) | 그대로 전사 |
| Video-MME | `image_urls` (frame 64장) | **없음** | **주입하지 않음** (`audio_mode: none`) |

### frame sampling

benchmark마다 official protocol이 달라 설정이 통일되어 있지 않습니다. 기존 런과 동일하게 유지합니다.

| Benchmark | 샘플링 주체 | 설정 | 근거 |
| --- | --- | --- | --- |
| AV-SpeakerBench | 서버 (모델 프로세서) | 모델 기본값 | official repo에 frame protocol 규정 없음. 클립이 5~22초로 짧아 context window 안에 들어옴 |
| OmniDCBench | 서버 (모델 프로세서) | 모델 기본값 | 클립이 70초 이하. per-request sampling kwargs는 서버가 무시함 |
| WorldSense | 클라이언트 | 균등 8 frame (영상 길이 무관) | VLMEvalKit `WorldSense_8frame_audio` variant |
| OmniVideoBench | 클라이언트 | 2.0 fps · 최대 120 frame | official Qwen3-Omni eval 설정 |
| Video-MME | 클라이언트 | 균등 64 frame · frame당 602,112 px | VLMEvalKit `Video-MME_64frame` 등가 |

서버에 위임하는 두 benchmark는 모델의 video processor 기본값에 따라 frame 수가 결정되므로, 모델 간 비주얼 입력량이 다를 수 있습니다. 각 모델을 자신의 native 설정으로 평가하는 프로토콜을 기존 런과 동일하게 따르되, 런 종료 후 `records.jsonl`의 `prompt_tokens`를 모델 간 비교해 실제 입력량 차이를 결과와 함께 기록합니다.

### Video-MME 주의사항

official Video-MME는 frame 외에 subtitle과 audio도 입력 modality로 규정합니다. 그러나 현재 adapter는 다음 상태입니다.

- audio: 미사용. 초기 구현은 원본 영상을 서버에 전달했으나, long 클립에서 context window를 초과해(276k token) client-side frame sampling으로 전환되었고 audio가 함께 제외되었습니다.
- subtitle: 미사용. `use_subtitles: false`이며, dataset parquet에 subtitle 컬럼 자체가 없습니다. official subtitle은 별도 SRT로 배포되고 adapter에 로딩 경로가 없습니다.

따라서 Video-MME 수치는 4개 모델 모두 frame + 질문 텍스트만 사용한 결과입니다. 상호 비교는 유효하지만 official leaderboard 수치와의 직접 대조는 부적절합니다. Qwen3.8-27B에도 ASR을 주입하지 않아 기존 런과 입력을 동일하게 유지합니다.

### ASR 프로토콜

| 항목 | 값 |
| --- | --- |
| 엔진 | faster-whisper (CTranslate2) · `Systran/faster-whisper-large-v3` |
| compute type | float16 |
| task | `transcribe` (원어 유지). `translate`는 omni 모델에 없는 번역 단계를 cascade에만 주게 되어 사용하지 않음 |
| language | 자동 감지 |
| 옵션 | `beam_size: 5` · `vad_filter: true` · `condition_on_previous_text: false` |
| 주입 위치 | benchmark official prompt **앞**. official prompt 문자열은 바이트 단위로 보존 |
| 형식 | `[mm:ss] 발화` 줄 단위. 발화가 없으면 `Audio transcript: (no speech detected)` |
| 예시 | benchmark별 실제 조립 결과는 [qwen3_8_whisper.md](qwen3_8_whisper.md#실제-조립되는-프롬프트) |
| 캐시 | `/gpfs/public/artifacts/ail/omni-bench/cache/asr/<strategy>__<model>/` · media path + size + mtime 기준 |

전사는 평가 전 별도 배치로 수행합니다(GPU 4장 분산, 재개 가능). 평가 프로세스는 캐시만 읽습니다.

## 실험

| # | 구성 | `audio_mode` | 대상 | Config |
| --- | --- | --- | --- | --- |
| E1 | Qwen3.8-27B + Whisper | `asr_text` (Video-MME는 `none`) | 5개 전체 | `configs/models/qwen3_8_27b_whisper.yaml` |

ASR 배치 전사 설정은 `configs/asr/whisper_large_v3.yaml`입니다.

audio를 제거한 baseline(`audio_mode: none`)이나 audio caption 보조 채널은 측정하지 않습니다.

- audio를 빼면 점수가 내려가는 것은 자명하고, 그 크기가 교체 판단을 바꾸지 않습니다. Video-MME는 애초에 두 조건의 입력이 동일해 중복입니다.
- 비언어 audio(음악 · 효과음 · 화자 특성 · 발화 강도 · 피치)를 보강할 수 있는 보조 모델이 확보되어 있지 않습니다. 로컬에 있는 audio captioning / tagging 모델은 AudioCaps 도메인 캡션이나 AudioSet 라벨을 출력하므로 AV-SpeakerBench가 묻는 화자 속성에 답할 수 없습니다. 따라서 이 한계는 해결 대상이 아니라 **cascade 구조의 한계로 결론에 기록**합니다.

`vllm bench throughput`은 별도로 돌리지 않습니다. 교체 판단에 필요한 비용 신호는 E1 런이 benchmark별로 기록하는 `perf` 블록으로 확보합니다.

## Serving

| 항목 | 값 |
| --- | --- |
| 스크립트 | `scripts/serve_qwen3_8_27b.sh` |
| Tensor parallel size | 1 |
| Data parallel size | 사용 가능한 GPU 수 (B200 4장) |
| GPU memory utilization | 0.9 |
| Allowed local media path | `/gpfs/public/datasets` |

Qwen3.8-27B는 audio를 사용하지 않으므로 omni 서빙 스크립트의 `VLLM_MAX_AUDIO_DECODE_DURATION_S`와 `--moe-backend`는 제거했습니다.

dense 27B는 활성 파라미터가 MoE(active 3B) 대비 크므로, data parallel로 4장을 사용해도 처리량은 기존 omni 모델보다 낮을 것으로 예상합니다. 요청 분배는 vLLM의 data-parallel load balancer가 처리하므로 애플리케이션 측 로직은 없습니다.

## 결과 저장 위치

```text
results/qwen3.8-27b-whisper/<benchmark-name>/
```

기존과 동일하게 `records.json`, `records.jsonl`, `summary.json`이 저장됩니다. `summary.json`에는 benchmark별 처리량·레이턴시가 `perf` 블록으로 함께 기록됩니다 — `wall_s`, `samples_per_s`, `latency_s`(mean/p50/p90/p99/max), `prompt_tokens`·`completion_tokens` 분포, `*_tokens_per_s`. `wall_s`는 frame 디코딩과 transcript 조회까지 포함해 재실행 시간을 예측하는 값이고, `latency_s`는 서버 큐 대기가 섞여 같은 런 안에서만 비교합니다.

 benchmark별 추가 출력(Video-MME `official_results.json`, WorldSense `vlmeval_rating.json`, OmniDCBench `predictions.jsonl`)도 그대로 생성됩니다.

ASR 산출물:

```text
/gpfs/public/artifacts/ail/omni-bench/cache/asr/
  faster_whisper__Systran__faster-whisper-large-v3/
    <key[:2]>/<key>.json            transcript (segment + timestamp + engine 메타)
  prepare_asr_report.json           배치 전사 요약 (config 스냅샷 포함)
```

## 결과 (2026-08-19 측정)

benchmark별 최선값 기준. E1 = Whisper transcript 주입, E0 = audio 제거. Video-MME는 benchmark 단위로 audio를 쓰지 않으므로 E0/E1이 동일합니다.

| Benchmark | E0 | E1 | E1−E0 | best | Qwen3-Omni | 격차 |
| --- | --- | --- | --- | --- | --- | --- |
| AV-SpeakerBench | 46.58 | **50.47** | +3.89 | 50.47 | 55.98 | −5.51 |
| WorldSense | 38.75 | **46.97** | +8.22 | 46.97 | 51.36 | −4.39 |
| Video-MME | 56.78 | 56.78 | 0 | 56.78 | 70.19 | −13.41 |
| OmniVideoBench | 37.60 | **41.70** | +4.10 | 41.70 | 41.20 | **+0.50** |
| OmniDCBench F1 | **55.93** | 48.25 | **−7.68** | 55.93 | 71.99 | −16.06 |

기존 4개 모델 수치는 2026-07-08 측정 결과(`report.html`)입니다. 총 19,712 샘플 · 94.5분 · 에러 0.

Video-MME는 파싱 결함의 영향을 받습니다. 저장된 응답으로 재채점하면 **67.26**(격차 −2.93)입니다. 자세한 내용은 [qwen3_8_whisper.md](qwen3_8_whisper.md)를 참고하세요.

### transcript의 기여는 benchmark에 따라 부호가 갈립니다

- MCQ 3종은 모두 양(+): WorldSense +8.22, OmniVideoBench +4.10, AV-SpeakerBench +3.89
- **captioning 1종은 음(−): OmniDCBench −7.68.** 이 데이터셋은 audio의 89%가 중국어인데 영어 caption과 timestamp JSON을 요구합니다. Precision·Recall이 함께 7.7점씩 떨어져 매칭되는 구간 자체가 줄었습니다
- 하위 항목 수준에서는 MCQ에서도 음수 구간이 있습니다. AV-SpeakerBench의 Speech Intensity −5.34, Activity Recognition −3.40 — audio의 비언어적 속성을 묻거나 순수 비주얼인 항목입니다. 반대로 Speech Recognition +16.42, Speaker Detection +13.58

### 결론

Qwen3.8-27B + Whisper cascade는 **Qwen3-Omni를 대체할 수준이 아닙니다.** 5종 중 1종(OmniVideoBench +0.50)만 앞서고, 나머지는 4.39~16.06점 뒤집니다.

격차의 원인이 cascade 구조가 아니라는 점이 중요합니다. audio를 쓰지 않는 항목에서도 일관되게 밀립니다 — AV-SpeakerBench Visual-centric −4.72, WorldSense Recognition −5.23, Video-MME 전체(audio 미사용, 재채점 후에도) −2.93. 즉 **비주얼 이해력 자체의 격차**이며 ASR 품질을 올려서 메울 수 없습니다.

또한 Video-MME에서 Qwen3.8-27B는 프레임당 588 토큰(Qwen3-Omni는 192)을 받아 **비주얼 입력이 3배 많았음에도** 뒤졌습니다.

## 세부 분석 항목

cascade 구조의 한계가 드러날 지점을 benchmark 세부 축으로 확인합니다.

| Benchmark | 세부 축 | 확인 내용 |
| --- | --- | --- |
| AV-SpeakerBench | Visual / Audio / Speaker-centric | Audio-centric에서 Qwen3-Omni 56.09를 넘는지. Speech Pitch · Intensity · Speaker Recognition은 ASR로 표현되지 않는 항목 |
| WorldSense | Reasoning / Understanding / Recognition · duration | 비주얼이 8 frame으로 제한된 조건에서의 상대 성능 |
| Video-MME | short / medium / long | frame 64장 고정이므로 long은 frame 간격 약 37초. 장영상 이해력이 아니라 희소 snapshot 성능으로 해석 |
| OmniVideoBench | audio type (Speech / Sound / Music) | Sound · Music은 ASR로 커버 불가 — 하락 예상 구간 |
| OmniDCBench | F1 · mIoU · Precision · Recall | timestamp 정렬 품질. audio 포함 요청 실패 시 `use_audio_in_video: false` 폴백이 있어 `fallback_reason` 발생 건수도 확인 |

## 예상 리스크

| 리스크 | 대응 |
| --- | --- |
| 비언어 audio 손실 — 음악 · 효과음 · 화자 특성 · 발화 강도/피치는 ASR에 담기지 않음 | 보강 수단이 없으므로 cascade의 구조적 한계로 결론에 기록. AV-SpeakerBench 세부 항목에서 손실 폭을 확인 |
| 시간 정렬 손실 — transcript는 텍스트이므로 frame과의 동기가 약함 | `[mm:ss]` timestamp 유지. Temporal Localization · Event Sorting 항목으로 검증 |
| ~~vLLM이 `qwen3_5` 아키텍처를 미지원할 가능성~~ | **해소** — vLLM 0.24.0이 `Qwen3_5ForConditionalGeneration`을 등록 |
| Whisper 환각 — 발화 없는 클립에서 자막 크레딧 패턴 출력 | `vad_filter`만으로는 부족함을 확인. `no_speech_threshold` · `hallucination_silence_threshold` 조정과 패턴 후처리 검토 |
| dense 27B의 낮은 throughput으로 총 소요 시간 초과 | benchmark 우선순위(AV-SpeakerBench → OmniVideoBench → 나머지)로 순차 진행 |
| prompt 길이 증가 — transcript 삽입으로 입력 token 증가 | 262k context로 여유. 처리량 비교 시 입력 길이 차이를 명시 |
| 서버 위임 benchmark의 모델 간 frame 수 차이 | `prompt_tokens` 비교로 정량화해 결과와 함께 기록 |

## 진행 순서

1. `uv sync` · flash-attn wheel 설치
2. Qwen3.8-27B vLLM 서빙 smoke test
3. ASR 배치 전사 (`scripts/prepare_asr.py`, GPU 4장)
4. E1 실행
5. 결과 표 작성 및 모델 교체 여부 판단
