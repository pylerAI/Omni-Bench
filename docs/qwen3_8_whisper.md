# Qwen3.8-27B + Whisper (cascaded ASR)

실험 계획과 결과 표는 [Qwen3.8-27B + Whisper 평가 계획](qwen3.8_plan.md)에 있습니다. 이 문서는 구현과 실행 방법을 다룹니다.

## 배경

AAII bench에서 Qwen3.8-27B가 52점으로 GPT-5.6-Luna와 동급의 언어 성능을 보였습니다. 같은 모델의 멀티모달 능력이 omni 전용 모델(Qwen3-Omni, Nemotron-3-Nano-Omni)을 넘어서는지 확인하고, 넘어선다면 다운스트림 태스크의 백본 교체를 검토하는 것이 이 실험의 목적입니다.

## 왜 cascade인가

`Qwen3.8-27B`(`Qwen3_5ForConditionalGeneration`)는 vision + text 모델입니다. `image_token_id` · `video_token_id`는 있지만 **audio encoder가 없습니다**. omni 모델과 같은 벤치마크로 비교하려면 audio를 다른 경로로 넣어야 하므로, Whisper로 전사한 텍스트를 프롬프트에 주입합니다.

따라서 이 실험은 **native omni vs cascaded VLM + ASR** 의 비교입니다.

| 항목 | 값 |
| --- | --- |
| 아키텍처 | `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) |
| text 백본 | 64층 하이브리드 (linear attention 3 : full attention 1) · hidden 5120 · GQA 24/4 |
| 컨텍스트 | 262,144 (mRoPE interleaved) |
| vision tower | 27층 · hidden 1152 · patch 16 |
| audio | **없음** — Whisper cascade로 대체 |

## 실험 구성 — recommend config

Qwen 권장 설정으로 **non-think / thinking 두 조건**을 측정합니다. 이전 evalkit 기반(프레임 고정 · `temperature 0`)은 archive로 옮겼습니다.

| 조건 | `enable_thinking` | temp / top_p / presence | `max_tokens` | Config |
| --- | --- | --- | --- | --- |
| non-think | false | 0.7 / 0.80 / 1.5 | 4,096 | `qwen3_8_27b_whisper_nothink.yaml` + `bench_nothink.yaml` |
| thinking | **true** | 1.0 / 0.95 / 0.0 | **32,768** | `qwen3_8_27b_whisper_thinking.yaml` + `bench_thinking.yaml` |

두 조건 모두 `top_k 20` · `min_p 0.0` · `repetition_penalty 1.0` · `frame_sampling: server`입니다. config는 `configs/recommend/` 에 있습니다.

`audio_mode`는 4종에서 `asr_text`, Video-MME만 `none`입니다 — official 프로토콜이 audio를 입력으로 쓰지 않고 비교 모델도 같은 조건이기 때문입니다.

**결과는 thinking이 유일한 유효 변수입니다.** 샘플링 파라미터(−0.13 ~ +2.90)와 frame sampling(−1.00 ~ +0.15)은 분산 범위이고, thinking만 +9.80 ~ +15.01을 만듭니다. 수치는 [평가 문서](qwen3.8_plan.md#2-결과)를 참고하세요.

## 벤치마크별 audio 경로

adapter는 전부 `VllmChatClient.complete()` 하나만 호출하므로, audio 처리는 client 계층에서만 교체됩니다. **adapter 코드는 수정하지 않았습니다.**

recommend config는 5종 전부 `frame_sampling: server` — 원본 영상을 그대로 넘깁니다.

| Benchmark | 비주얼 | audio 소스 | cascade 처리 |
| --- | --- | --- | --- |
| AV-SpeakerBench | `video_path` | 영상 컨테이너 내 오디오 트랙 | ffmpeg로 demux 후 전사 |
| OmniDCBench | `video_path` | 영상 컨테이너 내 오디오 트랙 | ffmpeg로 demux 후 전사 · `use_audio_in_video`는 false로 강제 |
| WorldSense | `video_path` | `audio_path` (.wav) | 그대로 전사 |
| OmniVideoBench | `video_path` | `audio_path` (.wav) | 그대로 전사 |
| Video-MME | `video_path` | **없음** | **주입 없음** |

AV-SpeakerBench와 OmniDCBench는 어댑터가 원래부터 `video_path`를 보내므로 `frame_sampling` 키가 필요 없습니다. 나머지 셋은 이 키로 클라이언트 프레임 추출을 끕니다.

Video-MME는 `audio_path`를 넘기지 않고, 그 위에 `configs/benchmarks/default.yaml`에서 `audio_mode: none`으로 못박아 두었습니다 — official 프로토콜이 audio를 입력으로 쓰지 않고 비교 모델도 같은 조건입니다.

## audio_mode 우선순위

`audio_mode`는 model config와 benchmark config 양쪽에서 지정할 수 있고, **benchmark 값이 model 값을 덮어씁니다**. audio 사용 여부는 모델의 성질이기도 하지만 benchmark protocol의 성질이기도 하기 때문입니다 — official 설정에서 audio가 빠진 benchmark는 모델이 `asr_text` 모드로 돌아도 audio-free로 유지되어야 합니다.

```yaml
# configs/models/qwen3_8_27b_whisper.yaml
models:
  - name: qwen3.8-27b-whisper
    audio_mode: asr_text        # 모델 기본값

# configs/benchmarks/default.yaml
benchmarks:
  - name: videomme
    audio_mode: none            # 이 benchmark만 예외
```

benchmark config의 `asr:` 블록도 model의 `asr:` 블록 위에 병합됩니다. 필요한 항목만 적으면 되고 엔진 설정을 다시 쓸 필요가 없습니다.

```yaml
  - name: omnidcbench
    asr:
      max_chars: 4000           # 나머지는 model 값 상속
```

STT 엔진은 `AsrCommandPool`이 관리해 동일한 엔진 설정을 쓰는 benchmark 사이에서 **한 번만 로드**됩니다. prompt 포맷 옵션(`max_chars` 등)은 benchmark마다 달라도 엔진은 공유됩니다.

## 실제 조립되는 프롬프트

`AsrTextChatClient`는 transcript 블록을 official prompt **앞**에 붙이고 둘 사이에 빈 줄 하나를 넣습니다. official prompt 문자열은 바이트 단위로 그대로 남아 official parser가 영향을 받지 않습니다.

아래는 15초 클립에 발화 3개가 잡힌 경우의 예시입니다. transcript 블록은 모든 벤치마크에서 동일하고, 그 아래 official prompt만 벤치마크마다 다릅니다.

### AV-SpeakerBench
미디어 파트: `video_url` (원본 영상 · 서버 기본 frame 샘플링)

```text
Audio transcript of the media (speech recognised automatically):
[00:01] So the first thing you want to do is preheat the oven.
[00:05] Meanwhile, mix the flour and the sugar together.
[00:11] You'll hear it start to sizzle right about now.

Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
How many people speak in the video?
A. one
B. two
C. three
D. four
The best answer is:
```

### WorldSense
미디어 파트: `image_url` × 8 (JPEG data URI)

```text
Audio transcript of the media (speech recognised automatically):
[00:01] So the first thing you want to do is preheat the oven.
[00:05] Meanwhile, mix the flour and the sugar together.
[00:11] You'll hear it start to sizzle right about now.

These are the frames of a video and the corresponding audio. Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
Question: What is the speaker preparing?
A. bread
B. soup
C. salad
D. cake
Answer: 
```

### OmniVideoBench
미디어 파트: `video_path` (원본 영상 · 서버 프레임 샘플링) · system prompt 별도

```text
Audio transcript of the media (speech recognised automatically):
[00:01] So the first thing you want to do is preheat the oven.
[00:05] Meanwhile, mix the flour and the sugar together.
[00:11] You'll hear it start to sizzle right about now.

You are given a video. Based on the content of the video, answer the following question:

Question:
What sound occurs right after the mixing step?

Options:
A. sizzling
B. silence
C. music
D. applause

Answer with the option's letter directly(e.g., A, B, C, or D).If your access to the video content is limited, at least one option that is more likely than the others must be chosen.Mustn't give any other reason for can not choose!
```

### OmniDCBench
미디어 파트: `video_url` (원본 영상 · 서버 기본 frame 샘플링)

```text
Audio transcript of the media (speech recognised automatically):
[00:01] So the first thing you want to do is preheat the oven.
[00:05] Meanwhile, mix the flour and the sugar together.
[00:11] You'll hear it start to sizzle right about now.

Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.

Return only a valid JSON array. Do not include markdown fences or any extra text. Each array item must describe one temporal segment and include:
- "timestamp": a string in "MM:SS-MM:SS" format, relative to the start of this clip.
- "caption": a detailed audio-visual caption for that segment.
Use enough segments to cover the full video from beginning to end.
This clip is exactly 15 seconds long. Every timestamp must lie within 00:00-00:15, and no timestamp may exceed 00:15. Do not describe or invent any segment beyond the end of the clip; stop once the clip ends.
```

### Video-MME — 주입하지 않음
미디어 파트: `image_url` × 64 (JPEG data URI · frame당 602,112 px 상한)

`audio_mode: none`이므로 transcript가 붙지 않습니다. 기존 4모델 런과 프롬프트가 완전히 동일합니다.

```text
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
What is shown at the end of the video?
A. a cake
B. a car
C. a dog
D. a book
The best answer is:
```

### 발화가 없을 때
오디오 트랙이 없거나 VAD가 발화를 못 찾으면 블록이 한 줄로 대체됩니다.

```text
Audio transcript: (no speech detected)

<official prompt>
```

- `max_chars`를 설정하면 앞부분을 잘라내고 `...(truncated)...`를 표시합니다 — 뒤쪽을 남기는 이유는 긴 클립에서 답이 후반에 있는 경우가 많기 때문입니다
- transcript는 항상 미디어 파트 **뒤**, 텍스트 파트의 맨 앞에 들어갑니다

## 아키텍처

```text
src/omni_bench/asr/
  schema.py      TranscriptionRequest / Transcription — 모든 커맨드·전략의 공통 입출력 스키마
  strategies.py  SttStrategy ABC + 레지스트리 (스트래티지 패턴)
  audio.py       ffmpeg demux · 오디오 트랙 유무 판정
  cache.py       media 식별자 기반 transcript 디스크 캐시
  commands.py    Command ABC · TranscribeCommand · BatchTranscribeCommand (커맨드 패턴)
  format.py      Transcription -> 프롬프트 블록
src/omni_bench/asr_client.py
  NoAudioChatClient   audio_mode: none
  AsrTextChatClient   audio_mode: asr_text
  build_chat_client   config -> client 팩토리
```

**스트래티지 패턴** — STT 엔진은 `SttStrategy` 하나의 인터페이스로 교체됩니다. 커맨드는 어떤 엔진이 도는지 알지 못합니다.

| 전략 이름 | 엔진 | 용도 |
| --- | --- | --- |
| `faster_whisper` | CTranslate2 | **기본** — 대량 오프라인 전사에 가장 빠름 |
| `transformers_whisper` | HF pipeline | 레퍼런스 대조용 |

새 엔진은 `SttStrategy`를 상속하고 `@register_strategy("이름")`을 붙이면 등록됩니다.

**커맨드 패턴** — 모든 전사는 `TranscriptionRequest`를 받아 `Transcription`을 돌려줍니다. 캐시를 아는 곳은 커맨드뿐이며, 커맨드가 만드는 JSON이 디스크 캐시 포맷이자 후속 분석 스크립트의 계약입니다.

## 캐시

```text
/gpfs/public/artifacts/ail/omni-bench/cache/asr/<strategy>__<model-slug>/<key[:2]>/<key>.json
```

저장소는 NFS(`/home/ail`) 위에 있어 이런 핫한 저장소를 두기에 맞지 않고, transcript는 특정 checkout보다 오래 남아야 하므로 gpfs에 둡니다.

`key`는 `resolved path + size + mtime`의 SHA-1입니다. 상위 산출물(예: 재추출된 `.wav`)이 바뀌면 자동으로 무효화되고, 전략이나 Whisper 모델을 바꾸면 namespace가 달라져 서로 섞이지 않습니다.

## 실행

### 1. 선행 전사 (권장)

```bash
uv run python scripts/prepare_asr.py \
  --asr-config configs/asr/whisper_large_v3.yaml \
  --gpus 0,1,2,3 --threads-per-gpu 2
```

GPU당 워커 프로세스 1개, 프로세스당 스레드 풀로 병렬 처리합니다. 캐시된 파일은 건너뛰므로 **중단 후 재개 가능**합니다. 결과 요약은 `cache/asr/prepare_asr_report.json`에 저장됩니다.

주요 인자:

| 인자 | 설명 |
| --- | --- |
| `--benchmark` | 특정 벤치마크만. 반복 지정 가능 |
| `--media-root` | 추가 디렉터리/파일 |
| `--strategy` · `--model` · `--language` | config 값 덮어쓰기 |
| `--dry-run` | 대상 파일만 세어보고 종료 |

### 2. 서빙

```bash
MAX_MODEL_LEN=131072 VLLM_BIN=$UV_PROJECT_ENVIRONMENT/bin/vllm \
  bash scripts/serve_qwen3_8_27b.sh
```

`MAX_MODEL_LEN`을 올리는 이유는 thinking 출력(최대 32,768)과 server frame sampling의 입력(최대 25,760)을 함께 담기 위함입니다. 기본값으로는 thinking 런이 컨텍스트를 넘깁니다.

### 3. 평가

```bash
export ALLOWED_LOCAL_MEDIA_PATH=/gpfs/public
# bench = videomme | worldsense | omnivideobench | av_speakerbench

uv run omni-bench run --config configs/recommend/qwen3_8_27b_whisper_thinking.yaml \
    --benchmark-config configs/recommend/bench_thinking.yaml   --benchmark <bench>   # thinking
uv run omni-bench run --config configs/recommend/qwen3_8_27b_whisper_nothink.yaml \
    --benchmark-config configs/recommend/bench_nothink.yaml --benchmark <bench>   # non-think
```

Video-MME는 ASR을 주입하지 않으므로 별도 model config(`configs/recommend/qwen3_8_27b_videomme_thinking.yaml` · `..._nothink.yaml`)를 씁니다.

### 4. 잘린 건 보완

```bash
python scripts/strip_truncated.py all   # --dry-run 으로 먼저 세어볼 수 있다
```

`max_tokens`에 도달해 최종 답이 유실된 레코드를 제거합니다. 그 뒤 3번을 다시 실행하면 어댑터 재개 로직이 **제거된 건만** 다시 돌립니다.

판정 기준이 두 개입니다.

| 기준 | 이유 |
| --- | --- |
| `completion_tokens >= max_tokens` | 명시적 잘림 |
| `<think>`는 열렸는데 `</think>`가 없다 | 어댑터가 다른 상한을 쓰는 경우를 잡는다 — OmniVideoBench가 1,024에 걸렸을 때 첫 기준으로는 놓쳤다 |

OmniDCBench는 `prediction_json`이 `null`인 건도 함께 제거합니다. 캡셔닝은 JSON이 안 뽑히면 토큰이 남아도 그 샘플이 0점입니다.

**보완 효과는 작습니다.** 잘린 문항은 모델이 결론을 못 내려 헤매던 어려운 문항이므로, 토큰을 더 줘도 정답률이 평균보다 낮습니다 — Video-MME 77.22 → 77.33, WorldSense 61.13 → 61.32. AV-SpeakerBench는 정상 종료분만 본 68.14가 실제로는 65.35였습니다.

## ASR task — `transcribe` 고정

Whisper는 `transcribe`(원어 유지)와 `translate`(영어로 번역) 두 task를 지원합니다. faster-whisper 기본값이 `transcribe`이고, 이 프로젝트는 그 기본값을 그대로 씁니다.

`translate`를 쓰지 않는 이유는 공정성입니다. omni 모델은 오디오를 원어로 듣고 번역 단계가 없으므로, cascade에만 번역을 붙이면 비교 대상에 없는 처리 단계를 추가하는 셈이 됩니다. 벤치마크 질문이 영어라 `translate`가 cascade에 유리할 수 있다는 점이 오히려 이 선택의 근거입니다.

바꿀 필요가 생기면 config의 `options`에 `task: translate`를 넣으면 됩니다. `**options`로 그대로 전달되므로 코드 수정은 필요하지 않습니다.

전사된 클립의 `language`와 `language_probability`는 캐시 JSON에 파일별로 남으므로, 배치 종료 후 비영어 비율을 집계해 이 선택의 영향 범위를 확인할 수 있습니다.

## 검증 상태

| 항목 | 결과 |
| --- | --- |
| vLLM 아키텍처 지원 | vLLM 0.24.0이 `Qwen3_5ForConditionalGeneration`을 등록함 |
| 서빙 | GPU 1장 6분, DP=4 11분에 기동. FlashInfer JIT용 `ninja`가 PATH에 없으면 가중치 로드 후 첫 샘플링 커널 빌드 시점에 실패 |
| thinking | chat template이 `<think>`를 기본으로 염. non-think / thinking 두 조건을 모두 측정 (아래 참고) |
| 프롬프트 조립 | WorldSense 실제 영상 1편으로 확인 — official prompt 바이트 보존 |
| 캐시 재조회 | 0.2ms (엔진 미호출) |

### ASR 배치 실측 (전량 전사 완료)

GPU 4장 · 스레드 8 · faster-whisper large-v3 fp16. 오디오 **308.8시간**을 GPU 시간 **31.03시간**(RTF 10x)에 전사, 벽시계 약 40분. 실패 0건.

| Dataset | 파일 | 발화 | 무발화 | 오디오 없음 | 세그먼트 | 오디오 | RTF | 비영어 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AV-SpeakerBench | 6,159 | 4,098 | 2,061 | 2,053 | 21,729 | 23.8h | 8x | 0% |
| OmniVideoBench | 1,884 | 1,707 | 177 | 0 | 143,564 | 201.7h | 11x | 15% |
| OmniDCBench | 1,122 | 980 | 142 | 2 | 22,825 | 18.1h | 7x | **89%** |
| WorldSense | 1,662 | 1,524 | 138 | 0 | 43,316 | 65.2h | 9x | 3% |
| 합계 | 10,827 | 8,309 | 2,518 | 2,055 | 231,434 | 308.8h | 10x | — |

- AV-SpeakerBench의 "오디오 없음"은 데이터셋이 `visual_only/` 변종을 포함하기 때문입니다
- **OmniDCBench는 89%가 중국어**입니다(zh 862 / en 99). 프롬프트는 영어이고 영어 캡션을 요구하므로, `transcribe`를 유지하면 모델이 중국어 transcript로 영어 캡션을 만들어야 합니다. Qwen3-Omni도 같은 조건이므로 공정성은 유지되지만 해석 시 알고 봐야 합니다
- 크레딧 패턴 환각은 아래 별도 절에서 전수 집계했습니다 (214건 / 1.69%)

### thinking — 성능을 결정하는 변수

Qwen3.8-27B의 chat template은 `<think>`를 기본으로 엽니다. 초기에는 껐지만, recommend config 측정에서 **thinking이 유일하게 유효한 변수**로 확인됐습니다 (+9.80 ~ +15.01). `extra_body.chat_template_kwargs.enable_thinking`으로 지정하므로 코드 변경은 없습니다.

| 조건 | 생성 토큰 | 응답 형태 |
| --- | --- | --- |
| `enable_thinking: false` | 2 ~ 5 | `B` |
| `enable_thinking: true` | 1,869 ~ 3,914 | `...추론... </think>\n\nB` |

**official parser는 thinking 출력에 쓸 수 없습니다.** 첫 `[ABCD]` 문자를 집으므로 추론 텍스트에서 오답이 잡힙니다 — Video-MME thinking 런에서 official 27.70 대 개선 파서 77.33입니다. `scripts/rescore_mcq.py`의 `robust_extract`는 명시적 마커를 우선하고 없으면 마지막 단독 문자를 취하므로 `</think>` 뒤의 최종 답을 정확히 잡습니다 (추출 실패 2/2,700, 명시적 마커 포함 98.5%).

thinking은 두 가지 대가가 있습니다.

| 항목 | 내용 |
| --- | --- |
| 실행 시간 | 4 ~ 25배 (AV-SpeakerBench 8.1분 → 206.1분) |
| 잘림 | `max_tokens` 8,192에서 4.4 ~ 13.9%가 최종 답 유실 → 32,768로 상향 필요. OmniDCBench는 캡션 JSON까지 겹쳐 60.8% |

### 알려진 문제 — Whisper 환각 (1.69%)

발화가 없는 클립에서 Whisper가 자막 크레딧 패턴을 출력합니다. 캐시 12,660건 전수 조사:

| 구분 | 건수 | 비율 |
| --- | --- | --- |
| 무발화로 정상 처리 (세그먼트 0) | 2,677 | 21.1% |
| **환각 확정** (발화 총 길이 > 영상 길이) | **214** | **1.69%** |
| 동일 문장 반복 루프 | 6 | 0.05% |

```text
영상  8.9초 / 발화 30.0초 (3.4배) · "Thank you for watching!"
영상 13.2초 / 발화 30.0초 (2.3배) · "© transcript Emily Beynon"
```

발화 길이가 **정확히 30.0초**로 찍히는 것이 특징입니다 — Whisper의 30초 윈도우를 통째로 채운 것이고, 9초 영상에 30초 발화는 불가능합니다. 학습 데이터에 자동 자막이 대량 포함돼 무음에서 "영상 끝에 흔히 나오는 말"을 생성합니다.

**미대응입니다.** 환각 전사가 한 문장(약 10 토큰)이고 프롬프트가 5,290 ~ 25,760이라 기여가 무시 가능하며, 내용에 정답 단서가 없습니다. 다만 무발화 영상에 "말이 있다"는 잘못된 신호를 주므로 화자 수를 묻는 항목에는 이론상 해롭습니다.

거르려면 `발화 총 길이 > 영상 길이 × 1.05` 규칙 하나로 214건 전부 잡힙니다. 무발화 클립 비중이 높은 다운스트림에 적용할 때는 `format_transcript` 앞단에 넣는 것이 좋습니다.

## 환경 구축 — README 명령을 그대로 쓰면 안 된다

README의 flash-attn 설치 명령에는 두 가지 함정이 있습니다.

```bash
# README 그대로 — 위험
uv pip install "https://.../flash_attn-2.8.3+cu130torch2.11-...whl"

# 실제로 써야 하는 형태
uv pip install --python <venv>/bin/python --no-deps "https://.../flash_attn-...whl"
```

`--no-deps`가 없으면 wheel이 torch를 고정하지 않아 uv가 최신 torch를 끌어옵니다. 실제로 실행했을 때 `torch==2.13.0`, `triton==3.7.1` 설치가 계획됐습니다 — vLLM 0.24.0은 torch 2.11.0에 맞춰 빌드되어 있으므로 그대로 진행되면 서빙이 깨집니다. wheel 파일명의 `+cu130torch2.11`이 이미 torch 2.11 전용임을 말하고 있습니다.

`--python`도 필요합니다. `uv pip`은 `UV_PROJECT_ENVIRONMENT`를 참조하지 않으므로(그 변수는 `uv sync`/`uv run` 전용), 홈이 NFS라 venv를 다른 곳에 둔 환경에서는 설치 대상이 어긋납니다.

### flash-attn은 attention backend를 바꾸지 않는다

설치 전후로 서빙 로그가 동일합니다.

| 계층 | 설치 전 | 설치 후 |
| --- | --- | --- |
| ViT attention | `FLASH_ATTN` | `FLASH_ATTN` |
| MMEncoderAttention | `FLASH_ATTN` | `FLASH_ATTN` |
| 메인 attention | `FLASHINFER` | `FLASHINFER` |

vLLM이 번들 커널(`vllm-flash-attn`)을 쓰기 때문에 외부 패키지 유무와 무관합니다. AV-SpeakerBench 정확도도 50.78 → 50.90으로 사실상 불변이고 `prompt_tokens`는 4,728 그대로입니다. 즉 이 설치는 재현성 관점에서 변수가 아닙니다.

## frame sampling — 서버 위임을 채택했다

`frame_sampling: server`는 프레임을 클라이언트에서 뽑지 않고 원본 영상을 그대로 보내 모델 프로세서가 결정하게 합니다. 값은 모델 웨이트의 `video_preprocessor_config.json` 그대로입니다 — `fps 2` · `max_frames 768` · `min_frames 4` · `size.longest_edge 25,165,824` (**비디오 전체** 픽셀 예산).

핵심은 픽셀 예산이 프레임당이 아니라 비디오 전체에 걸린다는 점입니다. 해상도로 환산하면:

| 해상도 | 프레임당 px | 예산에 들어가는 프레임 | 프레임당 토큰 |
| --- | --- | --- | --- |
| 1280×720 (720p 원본) | 921,600 | **27장** | 450 |
| 854×480 (480p) | 409,920 | 61장 | 200 |
| 640×360 (360p) | 230,400 | 109장 | 112 |
| 480×256 | 122,880 | 204장 | 60 |
| 224×128 | 28,672 | 877장 | 14 |

720p를 지키면 27장밖에 못 보고, 768장을 채우려면 224×128까지 줄여야 합니다. 프로세서는 후자를 택합니다 — **공간 해상도를 팔아 시간 해상도를 삽니다.** 720p 원본 실측:

| 영상 길이 | 요청 프레임 (fps 2) | 실제 프레임 | 프레임 해상도 | 비디오 토큰 |
| --- | --- | --- | --- | --- |
| 97초 | 194 | 194 | 256×480 | 11,640 |
| 500초 | 1,000 | 768 (상한) | 128×224 | 10,752 |
| 2,820초 | 5,639 | 768 (상한) | 128×224 | 10,752 |

**6.4분(384초)을 넘으면 프레임 수가 768에서 멈추고 해상도만 계속 깎입니다** — 500초와 2,820초의 해상도가 같습니다.

### 정확도 영향 — 순효과는 없고 부수 효과가 크다

Video-MME 기준으로 evalkit 방식(프레임 64장 고정)과 비교하면 전체 정확도는 −1.00으로 분산 범위입니다. 다만 duration별로 갈립니다.

| 구간 | evalkit 64장 | server | 차이 |
| --- | --- | --- | --- |
| short | 78.67 | 80.22 | +1.55 |
| medium | 65.33 | 66.67 | +1.34 |
| long | 59.67 | 53.78 | **−5.89** |

long의 하락은 **공간 해상도 붕괴**에서 옵니다(128×224). task_type별로 보면 지각 과제에 집중됩니다.

| 유형 | 대표 과제 | 차이 |
| --- | --- | --- |
| 지각 (무엇이 보이나) | Spatial Reasoning −18.2 · Object Recognition −16.7 · OCR −14.3 · Counting −12.5 | **−12 ~ −18** |
| 추론 (무슨 일이 일어나나) | Action Reasoning +3.9 · Information Synopsis −2.5 · Temporal Reasoning −4.4 | −4 ~ +4 |

128×224에서도 "사람이 걷다가 앉는다"는 추론은 되지만 "저 표지판에 뭐라고 쓰였나"는 불가능합니다. **thinking이 이 손실을 보상합니다** — long에서 non-think 53.78 → thinking 67.78 (+14.00).

### 채택 이유 — 파싱 결함이 사라진다

정확도가 아니라 측정 품질 때문입니다.

| 항목 | evalkit 64장 | server |
| --- | --- | --- |
| 장문 응답 | 30.9% | **0.2%** |
| official 대 robust 파서 차이 | **+12.30** | **+0.07** |
| prompt 토큰 | 34,141 | **13,734** |
| wall (Video-MME 2,700건) | 45.4분 | **20.5분** |

evalkit 방식에서는 어느 파서를 쓰느냐로 12점이 갈렸고(55.59 대 67.89), Qwen3-Omni는 장문 응답이 0건이라 그 손실이 우리 모델에만 적용됐습니다. server 방식에서는 두 파서가 일치하므로 **모델 간 비교가 파서 선택에 좌우되지 않습니다.**

## Whisper와 vLLM은 같은 GPU에 올리지 않는다

두 단계를 시간축으로 분리해야 합니다. 한 번 겹쳐 돌렸다가 vLLM EngineCore가 죽었습니다.

```
EngineCore encountered a fatal error
  shm_broadcast.py ... self._spin_condition.wait(timeout_ms=...)
  TimeoutError
Dumping scheduler stats: num_running_reqs=6
```

vLLM이 `gpu_memory_utilization 0.9`로 GPU 0을 165GB 점유한 상태에서 Whisper를 같은 GPU에 올렸고, 경합이 그 GPU의 vLLM 워커를 굶겨 shm dequeue 타임아웃이 났습니다. OOM은 아니었지만 결과는 엔진 사망이고, 이후 요청은 전부 `APIConnectionError`가 됩니다. Video-MME 2,700건 중 1,495건(long 전량, medium 594건)이 이렇게 실패했습니다.

메모리 여유(183GB 중 18GB 남음)만 보면 들어갈 것처럼 보이지만, 문제는 메모리가 아니라 **SM 경합으로 인한 응답 지연**입니다.

### 실패한 런에서 재개할 때

adapter의 재개 로직은 레코드의 식별자만 봅니다(`question_id` 등). 에러 레코드도 "완료"로 취급하므로, 서버 사망으로 실패한 건은 **`records.jsonl`에서 에러 행을 지운 뒤** 재실행해야 다시 시도됩니다. 지우지 않으면 실패분이 그대로 남은 채 집계됩니다.

## Video-MME 파싱 결함

Video-MME 2,700건 중 **540건(20%)에서 Qwen3.8-27B가 "letter only" 지시를 무시하고 산문으로 답합니다.**

```
Based on the video, the stages of human evolution are presented in
chronological order. The first stage shown is **Dryopithecus**...
```

adapter의 `extract_answer`는 접두어를 제거한 뒤 `re.search(r"[ABCD]")`로 위치 무관하게 첫 A-D 문자를 뽑습니다. `Based`의 **B**가 걸립니다. **540건 중 539건이 `B`로 파싱**됐고, 정답 분포가 B 204 / C 146 / A 95 / D 95이므로 이 그룹의 정답률 37.96%는 "무조건 B"의 기대값(37.8%)과 사실상 같습니다 — 모델 성능이 아닙니다.

`scripts/rescore_mcq.py`는 저장된 응답만으로 재채점합니다(추론 불필요). 명시적 답 표기(`answer is X`, `**X**`, `(X)`, 행 선두 `X.`)를 먼저 찾고, 없으면 단어 경계 기준 **마지막** 단독 A-D를 씁니다.

| Benchmark | 장문 응답 | official | robust | 차이 |
| --- | --- | --- | --- | --- |
| Video-MME | 540 (20.0%) | 56.78 | **67.26** | **+10.48** |
| WorldSense | 103 (3.2%) | 46.97 | 47.67 | +0.69 |
| OmniVideoBench | 3 (0.3%) | 41.70 | 41.70 | 0.00 |
| AV-SpeakerBench | 0 (0%) | 50.47 | 50.47 | 0.00 |

AV-SpeakerBench가 정확히 동일하다는 점이 이것이 파서 일반의 문제가 아니라 특정 응답 형태에서만 발생함을 보여줍니다. Qwen3-Omni는 장문 응답이 0건이라 **결함이 우리 모델만 깎는 방향**으로 작동했습니다.

### server frame sampling으로 해소됐다

위 수치는 evalkit 방식(프레임 64장을 `image_url`로 전송)에서 측정한 것입니다. recommend config로 바꾸면 이 결함이 사라집니다.

| 조건 | 장문 응답 | official | robust | 차이 |
| --- | --- | --- | --- | --- |
| evalkit 64장 | 30.9% | 55.59 | 67.89 | **+12.30** |
| **server (non-think)** | **0.2%** | **66.81** | **66.89** | **+0.07** |

비디오로 전송하면 모델이 지시를 따라 단답을 냅니다. 원인은 확정하지 못했지만 이미지 64장 나열이 지시 준수를 방해한 것으로 보입니다.

**thinking 런에서는 다시 필요합니다.** thinking은 출력이 100% 장문이므로 official parser로는 27.70, `robust_extract`로는 77.33입니다. `</think>` 뒤에 단답이 오는 구조라 명시적 마커 포함률이 98.5%이고 추출 실패는 2,700건 중 2건입니다.

## frame당 토큰 — 두 모델이 같다

두 모델의 vision patch 설정이 동일합니다. `Qwen3.8-27B`의 `video_preprocessor_config.json`과 Qwen3-Omni의 `Qwen2VLVideoProcessor` 모두 `patch_size 16` · `merge_size 2` · `temporal_patch_size 2` 입니다. 따라서 같은 면적의 frame은 같은 토큰 수가 됩니다.

| Model | patch · merge · temporal | Video-MME prompt 평균 |
| --- | --- | --- |
| Qwen3.8-27B | 16 · 2 · 2 | 34,141 |
| Qwen3-Omni | 16 · 2 · 2 | 34,131 |

실측이 10 토큰 차이로 일치합니다. **비주얼 입력량은 동일합니다.**

`max_pixels: 602112`는 frame당 면적 상한이고, 토큰은 patch/merge로 결정됩니다 — 두 모델이 같으므로 이 값은 양쪽에 같게 작용합니다. 모델별로 다른 것은 전체 픽셀 예산(`size.longest_edge`, 25,165,824 대 12,845,056)뿐이고, Video-MME frame은 그 한참 아래여서 차이가 나타나지 않습니다.

### frame을 이미지로 보내면 temporal 병합이 사라진다

evalkit 방식에서 Video-MME adapter는 frame 64장을 `image_url` 파트 64개로 보냈습니다(`video_url` 0개). recommend config는 `video_path`를 보내므로 아래 문제가 없습니다. `image_processing_qwen2_vl.py`는 이미지 1장을 `temporal_patch_size`만큼 복제해 채우고 temporal grid를 1로 고정하므로, frame 간 병합이 일어나지 않습니다.

| 전송 방식 | temporal grid | 토큰 |
| --- | --- | --- |
| `image_url` × 64 (evalkit) | 64 | 약 34,100 |
| `video_path` (server, 실측) | 97 | **13,734** |

즉 evalkit 방식은 같은 정보를 2배 토큰으로 넣고, 인코더는 frame 간 시간 관계를 받지 못했습니다. mRoPE의 temporal 축도 video 입력에서만 제대로 쓰입니다. recommend config에서는 프레임이 194장으로 3배 늘었는데도 토큰이 40%로 줄었습니다 — temporal 병합과 비디오 전체 픽셀 예산이 함께 작용한 결과입니다.

## 스모크 테스트

의존성(faster-whisper, vLLM 서버) 없이 로직만 검증합니다. 가짜 STT 전략과 스텁 OpenAI SDK를 써서 프롬프트 조립, 캐시, audio_mode 분기를 확인합니다.

```bash
python3 tests/smoke_asr.py       # 전략 레지스트리 · 캐시 · demux · 배치 · 포맷
python3 tests/smoke_client.py    # audio_mode 3종 동작 · 프롬프트 주입 위치
python3 tests/smoke_override.py  # benchmark 오버라이드 · asr 병합 · 엔진 풀 공유
```

실제 Whisper 가중치와 vLLM 서버를 쓰는 경로는 별도로 확인해야 합니다.

## 캐시 미스 동작

기본값은 `strict_cache: false` 로, 캐시에 없으면 평가 프로세스가 그 자리에서 전사합니다(Whisper 모델은 최초 필요 시점에 1회만 로드). 재현성을 위해 선행 전사를 강제하려면 config에서 `strict_cache: true`로 두면 캐시 미스가 에러가 됩니다.

WorldSense와 OmniVideoBench는 adapter의 전처리 단계에서 `.wav`를 생성하므로, 그 전처리 전에 `prepare_asr.py`를 돌리면 해당 파일들이 아직 없습니다. 두 벤치마크는 첫 런에서 인라인 전사로 채워지거나, 전처리를 한 번 돌린 뒤 `prepare_asr.py`를 재실행하면 됩니다.
