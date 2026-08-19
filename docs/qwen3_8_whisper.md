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

## 실험 구성

| # | 구성 | `audio_mode` | Config |
| --- | --- | --- | --- |
| E1 | Qwen3.8-27B + Whisper | `asr_text` (Video-MME는 `none`) | `configs/models/qwen3_8_27b_whisper.yaml` |

`audio_mode: none`은 Video-MME에서 쓰이는 벤치마크 단위 값으로 남아 있습니다. audio를 전부 제거한 모델 단위 baseline은 측정하지 않습니다.

## 벤치마크별 audio 경로

adapter는 전부 `VllmChatClient.complete()` 하나만 호출하므로, audio 처리는 client 계층에서만 교체됩니다. **adapter 코드는 수정하지 않았습니다.**

| Benchmark | 비주얼 | audio 소스 | cascade 처리 |
| --- | --- | --- | --- |
| AV-SpeakerBench | `video_path` | 영상 컨테이너 내 오디오 트랙 | ffmpeg로 demux 후 전사 |
| OmniDCBench | `video_path` + `use_audio_in_video` | 영상 컨테이너 내 오디오 트랙 | ffmpeg로 demux 후 전사 · 플래그는 false로 강제 |
| WorldSense | `image_urls` (프레임) | `audio_path` (.wav) | 그대로 전사 |
| OmniVideoBench | `video_url` (data URL) | `audio_path` (.wav) | 그대로 전사 |
| Video-MME | `image_urls`만 | **없음** | **주입 없음 — 기존 4모델 런과 입력 동일** |

Video-MME는 `audio_path`도 `video_path`도 넘기지 않으므로 client가 자동으로 통과시키고, 그 위에 `configs/benchmarks/default.yaml`에서 `audio_mode: none`으로 못박아 두었습니다. 기존 4모델 수치(70.19 등)와 입력이 완전히 동일해 직접 비교가 성립합니다.

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
미디어 파트: `video_url` (2.0 fps · 최대 120장 data URI) · system prompt 별도

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
bash scripts/serve_qwen3_8_27b.sh
```

### 3. 평가

```bash
uv run omni-bench run --config configs/models/qwen3_8_27b_whisper.yaml
```

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
| thinking | chat template이 `<think>`를 기본으로 염 — 껐다 (아래 참고) |
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
- 크레딧 패턴 환각 의심 14건 / 발화 8,309건 (0.17%)

### thinking 비활성화

Qwen3.8-27B의 chat template은 `<think>`를 기본으로 엽니다. 동일 질문 실측:

| 설정 | 응답 | 출력 토큰 |
| --- | --- | --- |
| 기본 (thinking ON) | `We need answer multiple choice... green blue red black. Correct B... </think>\n\nB` | 35 |
| `enable_thinking: false` | `B` | 2 |

켜두면 official parser가 추론 텍스트에 등장하는 선택지 문자열을 먼저 매칭할 위험이 있고, OmniDCBench의 JSON 배열 출력도 깨집니다. baseline인 Qwen3-Omni-30B-A3B-Instruct가 non-thinking인 점도 근거입니다. model config의 `extra_body.chat_template_kwargs`로 지정하므로 코드 변경은 없습니다.

### 알려진 문제 — Whisper 환각

발화가 없는 클립에서 Whisper가 자막 크레딧 패턴을 출력하는 경우가 있습니다. `audio_class: ['Music']`인 WorldSense 영상(`dvOkwKAs`, 60초 고쟁 연주)에서 `vad_filter: true` 상태로도 `© transcript Emily Beynon`이 나왔습니다.

발화가 없는 클립에 존재하지 않는 정보가 주입되므로 cascade에 불리하게 작용합니다. `no_speech_threshold`, `hallucination_silence_threshold` 조정과 크레딧 패턴 후처리가 후보이며 아직 적용하지 않았습니다.

## 서버 위임 frame sampling은 재현되지 않는다

`configs/benchmarks/default.yaml`에서 AV-SpeakerBench와 OmniDCBench는 frame 수를 지정하지 않고 서버(모델 프로세서)의 기본 sampling에 맡깁니다. 같은 모델을 같은 코드로 다시 측정해 보면 이 두 benchmark만 리포트 수치와 어긋납니다.

| Benchmark | 재현 | 리포트 | 차이 | frame sampling |
| --- | --- | --- | --- | --- |
| OmniVideoBench | 41.10 | 41.20 | −0.10 | 클라이언트 |
| WorldSense | 51.48 | 51.36 | +0.12 | 클라이언트 |
| Video-MME | 69.67 | 70.19 | −0.52 | 클라이언트 |
| OmniDCBench F1 | 68.53 | 71.99 | −3.46 | **서버** |
| AV-SpeakerBench | 50.78 | 55.98 | −5.20 | **서버** |

vLLM 버전(lock이 0.24.0으로 고정), HF dataset(최신 커밋이 7개월 전), 로컬 미디어, adapter/config, `--moe-backend`, `--max-model-len`, serve 시점 `mm-processor-kwargs`를 모두 배제했습니다. `--max-model-len`을 지정하지 않아도 vLLM은 65536을 유도하고 `prompt_tokens` 평균이 4,728로 불변이라 frame 수가 바뀌지 않았습니다.

즉 이 두 benchmark의 비주얼 입력은 harness가 고정하지 못하는 값에 달려 있습니다. **비교는 같은 시점에 측정한 값끼리만 유효합니다.** 서로 다른 날의 수치를 대조하려면 frame 수를 config에 명시해야 합니다 — 다만 그렇게 하면 기존 리포트 수치와의 비교가 끊어집니다.

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

AV-SpeakerBench가 정확히 동일하다는 점이 이것이 파서 일반의 문제가 아니라 특정 응답 형태에서만 발생함을 보여줍니다.

기존 4개 omni 모델은 지시를 따라 단답을 냈으므로 이 함정에 걸리지 않았을 가능성이 큽니다. 즉 **결함이 신규 모델만 깎는 방향**으로 작동합니다. 두 수치를 함께 보고하고, 기존 모델의 저장된 응답이 남아 있으면 동일 기준으로 재채점해 비교하는 것이 가장 엄격합니다.

## frame당 토큰이 모델마다 다르다

`max_pixels: 602112`는 프레임당 면적 상한이지 토큰 상한이 아닙니다. 토큰 수는 모델의 patch/merge 설정으로 갈립니다.

| Model | patch · merge | 602,112px 프레임당 토큰 |
| --- | --- | --- |
| Qwen3.8-27B | 16 · 2 | **588** |
| Qwen3-Omni | 28px 그리드 | 192 |

Video-MME에서 Qwen3.8-27B의 prompt는 평균 34,141 토큰이었습니다(프레임 64장 × 약 532). 즉 **비주얼 입력량이 동일하지 않고, 신규 모델이 3배 가까이 많이 받았습니다.** 그 조건에서도 뒤졌다는 점은 결론을 약화시키지 않습니다.

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
