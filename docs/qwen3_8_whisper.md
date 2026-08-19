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
cache/asr/<strategy>__<model-slug>/<key[:2]>/<key>.json
```

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
| faster-whisper 실제 전사 | AV-SpeakerBench 클립 3건 성공. 모델 로드 포함 첫 건 87.5초, 이후 파일당 0.6~1.2초 |
| 캐시 재조회 | 0.2ms (엔진 미호출) |
| 프롬프트 조립 | WorldSense 실제 영상 1편으로 확인 — official prompt 바이트 보존 |

### 알려진 문제 — Whisper 환각

발화가 없는 클립에서 Whisper가 자막 크레딧 패턴을 출력하는 경우가 있습니다. `audio_class: ['Music']`인 WorldSense 영상(`dvOkwKAs`, 60초 고쟁 연주)에서 `vad_filter: true` 상태로도 `© transcript Emily Beynon`이 나왔습니다.

발화가 없는 클립에 존재하지 않는 정보가 주입되므로 cascade에 불리하게 작용합니다. `no_speech_threshold`, `hallucination_silence_threshold` 조정과 크레딧 패턴 후처리가 후보이며 아직 적용하지 않았습니다.

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
