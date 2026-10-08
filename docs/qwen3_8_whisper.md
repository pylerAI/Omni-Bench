# Qwen3.8-27B + Whisper (Cascaded ASR)

The experiment plan and result tables are in [Qwen3.8-27B + Whisper Evaluation](qwen3.8_plan.md). This document covers the implementation and how to run it.

## Purpose

On AAII bench, Qwen3.8-27B scored 52, matching GPT-5.6-Luna in language performance. The goal of this experiment is to check whether the same model's multimodal ability exceeds dedicated omni models (Qwen3-Omni, Nemotron-3-Nano-Omni), and if it does, to consider swapping the backbone of downstream tasks.

## Model

`Qwen3.8-27B` (`Qwen3_5ForConditionalGeneration`) is a vision + text model. It has `image_token_id` · `video_token_id` but **no audio encoder**. To compare it with omni models on the same benchmarks, audio must enter through another path, so text transcribed by Whisper is injected into the prompt.

This experiment is therefore a comparison of **native omni vs cascaded VLM + ASR**.

| Item | Value |
| --- | --- |
| Architecture | `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) |
| Text backbone | 64-layer hybrid (linear attention 3 : full attention 1) · hidden 5120 · GQA 24/4 |
| Context | 262,144 (mRoPE interleaved) |
| Vision tower | 27 layers · hidden 1152 · patch 16 |
| Audio | **None** — replaced by a Whisper cascade |

## Evaluation Method

### Recommend Config

Two conditions, **non-think / thinking**, are measured with the Qwen-recommended settings. The earlier evalkit-based setup (fixed frames · `temperature 0`) has been moved to the archive.

| Condition | `enable_thinking` | temp / top_p / presence | `max_tokens` | Config |
| --- | --- | --- | --- | --- |
| non-think | false | 0.7 / 0.80 / 1.5 | 4,096 | `qwen3_8_27b_whisper_nothink.yaml` + `bench_nothink.yaml` |
| thinking | **true** | 1.0 / 0.95 / 0.0 | **32,768** | `qwen3_8_27b_whisper_thinking.yaml` + `bench_thinking.yaml` |

Both conditions use `top_k 20` · `min_p 0.0` · `repetition_penalty 1.0` · `inference.frames: server`. The configs are in `configs/recommend/`.

`inference.audio` is `asr_text` on 4 benchmarks and `none` only on Video-MME — the official protocol does not use audio as input, and the comparison models run under the same condition.

**Thinking proves to be the only effective variable.** The sampling parameters (−0.13 ~ +2.90) and frame sampling (−1.00 ~ +0.15) are within noise; only thinking yields +9.80 ~ +15.01. See the [evaluation doc](qwen3.8_plan.md#results) for the numbers.

### Audio Path per Benchmark

Every adapter calls only `VllmChatClient.complete()`, so audio handling is swapped only in the client's inference pipeline (`src/omni_bench/inference/`). Adapters do not branch on the audio · frames · transport settings.

The recommend config uses `inference.frames: server` on all 5 benchmarks — the original video is passed as is.

| Benchmark | Visual | Audio source | Cascade handling |
| --- | --- | --- | --- |
| AV-SpeakerBench | `video_path` | Audio track inside the video container | Demuxed with ffmpeg, then transcribed |
| OmniDCBench | `video_path` | Audio track inside the video container | Demuxed with ffmpeg, then transcribed · `use_audio_in_video` forced to false |
| WorldSense | `video_path` | `audio_path` (.wav) | Transcribed as is |
| OmniVideoBench | `video_path` | `audio_path` (.wav) | Transcribed as is |
| Video-MME | `video_path` | **None** | **No injection** |

AV-SpeakerBench and OmniDCBench only send the original video, so `frames` is fixed to `server` and no key is needed. The other three (default `client`) disable client-side frame extraction with `inference.frames: server` in the benchmark config.

Video-MME passes no `audio_path`, and on top of that `configs/benchmarks/default.yaml` pins it with `inference.audio: none` — the official protocol does not use audio as input, and the comparison models run under the same condition.

### Audio Precedence

`inference.audio` (legacy key `audio_mode`) can be set in both the model config and the benchmark config, and **the benchmark value overrides the model value**. Whether audio is used is a property of the model, but also of the benchmark protocol — a benchmark whose official setting excludes audio must stay audio-free even when the model runs in `asr_text` mode.

```yaml
# configs/models/qwen3_8_27b_whisper.yaml
models:
  - name: qwen3.8-27b-whisper
    inference:
      audio: asr_text           # model default

# configs/benchmarks/default.yaml
benchmarks:
  - name: videomme
    inference:
      audio: none               # exception for this benchmark only
```

A benchmark config's `asr:` block is also merged over the model's `asr:` block. Only the needed entries have to be written; the engine settings need not be repeated.

```yaml
  - name: omnidcbench
    asr:
      max_chars: 4000           # everything else inherits the model value
```

The STT engine is managed by `AsrCommandPool`, so it is **loaded only once** across benchmarks that share the same engine settings. The engine is shared even when prompt-format options (`max_chars`, etc.) differ per benchmark.

### Assembled Prompts

The `asr_text` strategy (`AsrTextAudio`) prepends the transcript block **before** the official prompt, with one blank line between them. The official prompt string stays byte-identical, so the official parser is unaffected.

The examples below are for a 15-second clip with 3 detected utterances. The transcript block is identical across benchmarks; only the official prompt below it differs.

#### AV-SpeakerBench

Media part: `video_url` (original video · server default frame sampling)

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

#### WorldSense

Media part: `image_url` × 8 (JPEG data URI)

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

#### OmniVideoBench

Media part: `video_path` (original video · server frame sampling) · separate system prompt

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

#### OmniDCBench

Media part: `video_url` (original video · server default frame sampling)

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

#### Video-MME — No Injection

Media part: `image_url` × 64 (JPEG data URI · per-frame cap of 602,112 px)

With `inference.audio: none`, no transcript is attached. The prompt is identical to the earlier 4-model runs.

```text
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
What is shown at the end of the video?
A. a cake
B. a car
C. a dog
D. a book
The best answer is:
```

#### When There Is No Speech

If there is no audio track or VAD finds no speech, the block is replaced by a single line.

```text
Audio transcript: (no speech detected)

<official prompt>
```

- When `max_chars` is set, the beginning is cut and `...(truncated)...` is shown — the end is kept because in long clips the answer is often in the latter part
- The transcript always goes **after** the media parts, at the very start of the text part

### ASR Task — Fixed to `transcribe`

Whisper supports two tasks: `transcribe` (keeps the source language) and `translate` (translates into English). faster-whisper defaults to `transcribe`, and this project keeps that default.

`translate` is not used for fairness. Omni models hear the audio in its source language with no translation step, so adding translation only to the cascade would add a processing step the comparison targets do not have. The fact that the benchmark questions are in English, so `translate` might favor the cascade, is precisely the reason for this choice.

If a change is needed, add `task: translate` to the config's `options`. It is passed through via `**options`, so no code change is required.

The `language` and `language_probability` of each transcribed clip remain per file in the cache JSON, so the non-English share can be aggregated after a batch to check the impact of this choice.

## Implementation

### Architecture

```text
src/omni_bench/asr/
  schema.py      TranscriptionRequest / Transcription — shared I/O schema for every command and strategy
  strategies.py  SttStrategy ABC + registry (strategy pattern)
  audio.py       ffmpeg demux · audio-track detection
  cache.py       transcript disk cache keyed by media identity
  commands.py    Command ABC · TranscribeCommand · BatchTranscribeCommand (command pattern)
  format.py      Transcription -> prompt block
src/omni_bench/inference/audio.py
  NoAudio             inference.audio: none
  AsrTextAudio        inference.audio: asr_text (engine from asr.strategy.name)
  AsrCommandPool      loads the STT engine once per engine config
src/omni_bench/client.py
  build_chat_client   config -> client (inference pipeline) factory
```

**Strategy pattern** — STT engines are swapped behind a single `SttStrategy` interface. Commands do not know which engine is running.

| Strategy name | Engine | Use |
| --- | --- | --- |
| `faster_whisper` | CTranslate2 | **Default** — fastest for large offline transcription |
| `transformers_whisper` | HF pipeline | For reference comparison |

A new engine is registered by subclassing `SttStrategy` and adding `@register_strategy("name")`.

**Command pattern** — every transcription takes a `TranscriptionRequest` and returns a `Transcription`. Only the commands know about the cache, and the JSON they produce is both the on-disk cache format and the contract for downstream analysis scripts.

### Cache

```text
/gpfs/public/artifacts/ail/omni-bench/cache/asr/<strategy>__<model-slug>/<key[:2]>/<key>.json
```

The repository lives on NFS (`/home/ail`), which is the wrong place for a store this hot, and transcripts must outlive any particular checkout, so the cache lives on gpfs.

`key` is the SHA-1 of `resolved path + size + mtime`. When an upstream artifact (e.g. a re-extracted `.wav`) changes, the entry is invalidated automatically; changing the strategy or Whisper model changes the namespace, so they never mix.

### Cache-Miss Behavior

The default is `strict_cache: false`: on a cache miss, the evaluation process transcribes inline (the Whisper model is loaded once, at first need). To enforce pre-transcription for reproducibility, set `strict_cache: true` in the config and a cache miss becomes an error.

WorldSense and OmniVideoBench produce their `.wav` files in the adapter's preprocessing step, so running `prepare_asr.py` before that preprocessing finds those files missing. For these two benchmarks, either let the first run fill them via inline transcription, or run the preprocessing once and re-run `prepare_asr.py`.

## Running Evaluations

### 1. Pre-transcription (recommended)

```bash
uv run python scripts/prepare_asr.py \
  --asr-config configs/asr/whisper_large_v3.yaml \
  --gpus 0,1,2,3 --threads-per-gpu 2
```

One worker process per GPU, each with a thread pool. Cached files are skipped, so the job is **resumable**. The summary is saved to `cache/asr/prepare_asr_report.json`.

Main arguments:

| Argument | Description |
| --- | --- |
| `--benchmark` | Only specific benchmarks. Repeatable |
| `--media-root` | Additional directories/files |
| `--strategy` · `--model` · `--language` | Override config values |
| `--dry-run` | Only count the target files and exit |

### 2. Serving

```bash
MAX_MODEL_LEN=131072 VLLM_BIN=$UV_PROJECT_ENVIRONMENT/bin/vllm \
  bash scripts/serve_qwen3_8_27b.sh
```

`MAX_MODEL_LEN` is raised to fit both the thinking output (up to 32,768) and the server frame-sampling input (up to 25,760). With the default, thinking runs exceed the context.

### 3. Evaluation

```bash
export ALLOWED_LOCAL_MEDIA_PATH=/gpfs/public
# bench = videomme | worldsense | omnivideobench | av_speakerbench

uv run omni-bench run --config configs/recommend/qwen3_8_27b_whisper_thinking.yaml --benchmark <bench>   # thinking
uv run omni-bench run --config configs/recommend/qwen3_8_27b_whisper_nothink.yaml --benchmark <bench>    # non-think
# Video-MME: configs/recommend/qwen3_8_27b_videomme_{thinking,nothink}.yaml
```

Video-MME does not inject ASR, so it uses separate model configs (`configs/recommend/qwen3_8_27b_videomme_thinking.yaml` · `..._nothink.yaml`). Each measurement config names its benchmark config with `benchmark_config:` (`bench_thinking.yaml` · `bench_nothink.yaml` · `bench_videomme_thinking.yaml` · `bench_videomme_nothink.yaml`); `--benchmark-config` still overrides it.

### 4. Backfilling Truncated Items

```bash
python scripts/strip_truncated.py all   # --dry-run counts them first
```

This removes records whose final answer was lost by hitting `max_tokens`. Re-running step 3 afterwards makes the adapter's resume logic retry **only the removed items**.

There are two criteria.

| Criterion | Reason |
| --- | --- |
| `completion_tokens >= max_tokens` | Explicit truncation |
| `<think>` opened but no `</think>` | Catches adapters that use a different cap — the first criterion missed OmniVideoBench when it hit 1,024 |

For OmniDCBench, items whose `prediction_json` is `null` are removed as well. In captioning, a sample scores 0 if no JSON can be extracted, even with tokens left over.

**The backfill effect is small.** Truncated questions are hard ones where the model wandered without concluding, so even with more tokens their accuracy is below average — Video-MME 77.22 → 77.33, WorldSense 61.13 → 61.32. For AV-SpeakerBench, the 68.14 computed over normally finished items alone was actually 65.35.

### 5. Rescoring Thinking Runs

The local vLLM is served without a reasoning parser, so a thinking reply arrives as `<reasoning>\n</think>\n\nB` in `content`. The official parser takes the first `[ABCD]` character, so it reads letters from the reasoning. The thinking configs therefore set `inference.reasoning: think_tag`: the text before the last `</think>` is stored as `reasoning`, the official parser sees only what follows, and the raw content is kept as `response_raw`.

Runs recorded before this setting existed are re-scored without inference by re-applying the strategy and the adapter's own parser and summary code. The source run is only read.

```bash
uv run omni-bench rescore --run-dir results/qwen3.8-27b-whisper-srvthink \
    --config configs/recommend/qwen3_8_27b_whisper_thinking.yaml \
    --reasoning think_tag --out-dir results_rescored
```

| Benchmark | official parser on raw content | `think_tag` + official parser | robust parser (`rescore_mcq.py`) |
| --- | --- | --- | --- |
| Video-MME | 27.70 | 77.44 | 77.33 |
| WorldSense | 23.33 | 61.07 | 61.32 |
| OmniVideoBench | 0.00 | 52.30 | 52.50 |
| AV-SpeakerBench | 38.39 | 65.19 | 65.35 |

The reported thinking numbers are the robust-parser values. `scripts/rescore_mcq.py` remains available for that comparison:

```bash
python scripts/rescore_mcq.py results/<model>/<benchmark>/records.jsonl options
```

## Findings

### Verification Status

| Item | Result |
| --- | --- |
| vLLM architecture support | vLLM 0.24.0 registers `Qwen3_5ForConditionalGeneration` |
| Serving | Starts in 6 min on 1 GPU, 11 min with DP=4. If `ninja` (for FlashInfer JIT) is not on PATH, it fails when the first sampling kernel is built after weight loading |
| thinking | The chat template opens `<think>` by default. Both non-think / thinking conditions are measured (see below) |
| Prompt assembly | Checked with 1 real WorldSense video — official prompt preserved byte for byte |
| Cache re-lookup | 0.2ms (engine not called) |

### ASR Batch Measurements (full transcription completed)

4 GPUs · 8 threads · faster-whisper large-v3 fp16. **308.8 hours** of audio were transcribed in **31.03 GPU-hours** (RTF 10x), about 40 minutes of wall-clock time. 0 failures.

| Dataset | Files | Speech | No speech | No audio | Segments | Audio | RTF | Non-English |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AV-SpeakerBench | 6,159 | 4,098 | 2,061 | 2,053 | 21,729 | 23.8h | 8x | 0% |
| OmniVideoBench | 1,884 | 1,707 | 177 | 0 | 143,564 | 201.7h | 11x | 15% |
| OmniDCBench | 1,122 | 980 | 142 | 2 | 22,825 | 18.1h | 7x | **89%** |
| WorldSense | 1,662 | 1,524 | 138 | 0 | 43,316 | 65.2h | 9x | 3% |
| Total | 10,827 | 8,309 | 2,518 | 2,055 | 231,434 | 308.8h | 10x | — |

- AV-SpeakerBench's "no audio" count comes from the dataset including a `visual_only/` variant
- **89% of OmniDCBench is Chinese** (zh 862 / en 99). The prompt is in English and requires English captions, so keeping `transcribe` means the model must write English captions from a Chinese transcript. Qwen3-Omni runs under the same condition, so fairness holds, but this must be kept in mind when interpreting results
- Credit-pattern hallucinations are fully counted in a separate section below (214 cases / 1.69%)

### Thinking — The Variable That Decides Accuracy

Qwen3.8-27B's chat template opens `<think>` by default. It was initially disabled, but the recommend config measurements showed **thinking is the only effective variable** (+9.80 ~ +15.01). It is set via `extra_body.chat_template_kwargs.enable_thinking`, so no code change is involved.

| Condition | Generated tokens | Response shape |
| --- | --- | --- |
| `enable_thinking: false` | 2 ~ 5 | `B` |
| `enable_thinking: true` | 1,869 ~ 3,914 | `...reasoning... </think>\n\nB` |

**The official parser cannot be used on thinking outputs.** It picks the first `[ABCD]` character, so wrong answers are picked up from the reasoning text — on the Video-MME thinking run, official 27.70 vs improved parser 77.33. `robust_extract` in `scripts/rescore_mcq.py` prefers explicit markers and otherwise takes the last standalone letter, so it correctly picks the final answer after `</think>` (extraction failures 2/2,700, 98.5% with explicit markers).

Thinking has two costs.

| Item | Detail |
| --- | --- |
| Run time | 4 ~ 25x (AV-SpeakerBench 8.1 min → 206.1 min) |
| Truncation | At `max_tokens` 8,192, 4.4 ~ 13.9% lose the final answer → must be raised to 32,768. OmniDCBench reaches 60.8% because the caption JSON adds to it |

### Known Issue — Whisper Hallucinations (1.69%)

On clips without speech, Whisper outputs subtitle-credit patterns. Full survey of the 12,660 cache entries:

| Category | Count | Share |
| --- | --- | --- |
| Correctly handled as no speech (0 segments) | 2,677 | 21.1% |
| **Confirmed hallucination** (total speech length > video length) | **214** | **1.69%** |
| Same-sentence repetition loop | 6 | 0.05% |

```text
video  8.9 s / speech 30.0 s (3.4x) · "Thank you for watching!"
video 13.2 s / speech 30.0 s (2.3x) · "© transcript Emily Beynon"
```

The telltale sign is a speech length of **exactly 30.0 s** — it fills Whisper's whole 30-second window, and 30 seconds of speech in a 9-second video is impossible. Because the training data contains a large amount of auto-generated subtitles, Whisper produces "things commonly said at the end of a video" on silence.

**Not addressed.** A hallucinated transcript is one sentence (about 10 tokens) against a prompt of 5,290 ~ 25,760, so its contribution is negligible, and its content carries no answer cues. It does, however, wrongly signal "there is speech" for a silent video, so in theory it hurts questions that ask for the number of speakers.

A single rule, `total speech length > video length × 1.05`, catches all 214 cases. For downstream use with many silent clips, it is best placed in front of `format_transcript`.

### Environment Setup — Do Not Use the README Command As Is

The README's flash-attn install command has two pitfalls.

```bash
# README as is — risky
uv pip install "https://.../flash_attn-2.8.3+cu130torch2.11-...whl"

# the form to actually use
uv pip install --python <venv>/bin/python --no-deps "https://.../flash_attn-...whl"
```

Without `--no-deps`, the wheel does not pin torch, so uv pulls in the latest torch. When actually run, it planned to install `torch==2.13.0` and `triton==3.7.1` — vLLM 0.24.0 is built against torch 2.11.0, so proceeding would break serving. The `+cu130torch2.11` in the wheel filename already says it is for torch 2.11 only.

`--python` is needed too. `uv pip` does not consult `UV_PROJECT_ENVIRONMENT` (that variable is only for `uv sync`/`uv run`), so in environments where the venv lives elsewhere because home is on NFS, the install target goes wrong.

#### flash-attn Does Not Change the Attention Backend

The serving logs are identical before and after installation.

| Layer | Before install | After install |
| --- | --- | --- |
| ViT attention | `FLASH_ATTN` | `FLASH_ATTN` |
| MMEncoderAttention | `FLASH_ATTN` | `FLASH_ATTN` |
| Main attention | `FLASHINFER` | `FLASHINFER` |

vLLM uses its bundled kernels (`vllm-flash-attn`), so the external package makes no difference. AV-SpeakerBench accuracy is also essentially unchanged at 50.78 → 50.90, and `prompt_tokens` stays at 4,728. In other words, this install is not a variable for reproducibility.

### Frame Sampling — Server Delegation Adopted

`inference.frames: server` does not extract frames on the client; it sends the original video as is and lets the model processor decide. The values are exactly those in the model weights' `video_preprocessor_config.json` — `fps 2` · `max_frames 768` · `min_frames 4` · `size.longest_edge 25,165,824` (pixel budget for **the whole video**).

The key point is that the pixel budget applies to the whole video, not per frame. In terms of resolution:

| Resolution | px per frame | Frames that fit the budget | Tokens per frame |
| --- | --- | --- | --- |
| 1280×720 (720p source) | 921,600 | **27** | 450 |
| 854×480 (480p) | 409,920 | 61 | 200 |
| 640×360 (360p) | 230,400 | 109 | 112 |
| 480×256 | 122,880 | 204 | 60 |
| 224×128 | 28,672 | 877 | 14 |

Keeping 720p allows only 27 frames; filling 768 frames requires shrinking to 224×128. The processor picks the latter — **it trades spatial resolution for temporal resolution.** Measured on a 720p source:

| Video length | Requested frames (fps 2) | Actual frames | Frame resolution | Video tokens |
| --- | --- | --- | --- | --- |
| 97 s | 194 | 194 | 256×480 | 11,640 |
| 500 s | 1,000 | 768 (cap) | 128×224 | 10,752 |
| 2,820 s | 5,639 | 768 (cap) | 128×224 | 10,752 |

**Beyond 6.4 min (384 s), the frame count stops at 768 and only the resolution keeps dropping** — 500 s and 2,820 s get the same resolution.

#### Accuracy Impact — No Net Effect, Large Side Effects

Compared on Video-MME with the evalkit setup (64 fixed frames), overall accuracy is −1.00, within noise. It does, however, split by duration.

| Bucket | evalkit 64 frames | server | Diff |
| --- | --- | --- | --- |
| short | 78.67 | 80.22 | +1.55 |
| medium | 65.33 | 66.67 | +1.34 |
| long | 59.67 | 53.78 | **−5.89** |

The drop on long comes from the **collapse of spatial resolution** (128×224). By task_type, it concentrates on perception tasks.

| Type | Representative tasks | Diff |
| --- | --- | --- |
| Perception (what is visible) | Spatial Reasoning −18.2 · Object Recognition −16.7 · OCR −14.3 · Counting −12.5 | **−12 ~ −18** |
| Reasoning (what happens) | Action Reasoning +3.9 · Information Synopsis −2.5 · Temporal Reasoning −4.4 | −4 ~ +4 |

At 128×224 the model can still reason "a person walks and then sits down", but cannot read "what does that sign say". **Thinking compensates for this loss** — on long, non-think 53.78 → thinking 67.78 (+14.00).

#### Why It Was Adopted — The Parsing Defect Disappears

The reason is measurement quality, not accuracy.

| Item | evalkit 64 frames | server |
| --- | --- | --- |
| Long-form answers | 30.9% | **0.2%** |
| official vs robust parser gap | **+12.30** | **+0.07** |
| prompt tokens | 34,141 | **13,734** |
| wall (Video-MME 2,700 items) | 45.4 min | **20.5 min** |

In the evalkit setup, the choice of parser swung the score by 12 points (55.59 vs 67.89), and since Qwen3-Omni had 0 long-form answers, that loss applied only to our model. In the server setup the two parsers agree, so **cross-model comparisons do not depend on the parser choice.**

### Whisper and vLLM Must Not Share a GPU

The two stages must be separated in time. Running them overlapped once killed the vLLM EngineCore.

```
EngineCore encountered a fatal error
  shm_broadcast.py ... self._spin_condition.wait(timeout_ms=...)
  TimeoutError
Dumping scheduler stats: num_running_reqs=6
```

With vLLM occupying 165GB of GPU 0 at `gpu_memory_utilization 0.9`, Whisper was loaded onto the same GPU; the contention starved that GPU's vLLM worker and caused an shm dequeue timeout. It was not an OOM, but the result was engine death, and every later request became an `APIConnectionError`. Of 2,700 Video-MME items, 1,495 (all of long, 594 of medium) failed this way.

Looking only at free memory (18GB left out of 183GB) it seems to fit, but the problem is not memory — it is **response latency from SM contention**.

#### Resuming from a Failed Run

Re-running the same command skips finished items and retries only the error rows. Records carrying `error` are not treated as done; they are moved from `records.jsonl` to `records.errors.jsonl` (kept for diagnosis) before the retry (`load_resumable_records` in `src/omni_bench/io.py`).

### Video-MME Parsing Defect

Of 2,700 Video-MME items, in **540 (20%) Qwen3.8-27B ignores the "letter only" instruction and answers in prose.**

```
Based on the video, the stages of human evolution are presented in
chronological order. The first stage shown is **Dryopithecus**...
```

The adapter's `extract_answer` strips prefixes and then takes the first A-D character anywhere with `re.search(r"[ABCD]")`. It catches the **B** of `Based`. **539 of the 540 were parsed as `B`**, and given the gold distribution of B 204 / C 146 / A 95 / D 95, this group's accuracy of 37.96% is essentially the expected value of "always B" (37.8%) — it is not model performance.

`scripts/rescore_mcq.py` rescores from the stored responses alone (no inference needed). It first looks for explicit answer markers (`answer is X`, `**X**`, `(X)`, `X.` at line start), and otherwise uses the **last** standalone A-D on word boundaries.

| Benchmark | Long-form answers | official | robust | Diff |
| --- | --- | --- | --- | --- |
| Video-MME | 540 (20.0%) | 56.78 | **67.26** | **+10.48** |
| WorldSense | 103 (3.2%) | 46.97 | 47.67 | +0.69 |
| OmniVideoBench | 3 (0.3%) | 41.70 | 41.70 | 0.00 |
| AV-SpeakerBench | 0 (0%) | 50.47 | 50.47 | 0.00 |

AV-SpeakerBench being exactly identical shows this is not a general parser problem but one that arises only with a specific response shape. Qwen3-Omni had 0 long-form answers, so **the defect worked in the direction of penalizing only our model**.

#### Resolved by Server Frame Sampling

The numbers above were measured with the evalkit setup (64 frames sent as `image_url`). Switching to the recommend config makes the defect disappear.

| Condition | Long-form answers | official | robust | Diff |
| --- | --- | --- | --- | --- |
| evalkit 64 frames | 30.9% | 55.59 | 67.89 | **+12.30** |
| **server (non-think)** | **0.2%** | **66.81** | **66.89** | **+0.07** |

When sent as a video, the model follows the instruction and answers briefly. The cause was not confirmed, but listing 64 images appears to have interfered with instruction following.

**It is needed again for thinking runs.** Thinking output is 100% long-form, so the official parser gives 27.70 and `robust_extract` gives 77.33. Since a short answer follows `</think>`, 98.5% include an explicit marker, and extraction fails on 2 of 2,700 items.

### Tokens per Frame — Same for Both Models

The two models have the same vision patch settings. `Qwen3.8-27B`'s `video_preprocessor_config.json` and Qwen3-Omni's `Qwen2VLVideoProcessor` both use `patch_size 16` · `merge_size 2` · `temporal_patch_size 2`. Frames of the same area therefore yield the same number of tokens.

| Model | patch · merge · temporal | Video-MME prompt mean |
| --- | --- | --- |
| Qwen3.8-27B | 16 · 2 · 2 | 34,141 |
| Qwen3-Omni | 16 · 2 · 2 | 34,131 |

The measurements agree within 10 tokens. **The visual input volume is the same.**

`max_pixels: 602112` is a per-frame area cap, and tokens are determined by patch/merge — the two models are the same, so this value acts identically on both. The only per-model difference is the whole-video pixel budget (`size.longest_edge`, 25,165,824 vs 12,845,056), and Video-MME frames are far below it, so no difference shows up.

#### Sending Frames as Images Removes Temporal Merging

In the evalkit setup, the Video-MME adapter sent 64 frames as 64 `image_url` parts (0 `video_url`). The recommend config sends `video_path`, so it does not have the problem below. `image_processing_qwen2_vl.py` fills a single image by duplicating it `temporal_patch_size` times and fixes the temporal grid at 1, so no merging across frames happens.

| Transport | temporal grid | Tokens |
| --- | --- | --- |
| `image_url` × 64 (evalkit) | 64 | about 34,100 |
| `video_path` (server, measured) | 97 | **13,734** |

So the evalkit setup fed the same information with 2x the tokens, and the encoder never received the temporal relation between frames. mRoPE's temporal axis is also used properly only for video input. With the recommend config, frames grew 3x to 194 while tokens dropped to 40% — the combined result of temporal merging and the whole-video pixel budget.

## Smoke Tests

These verify the logic only, without dependencies (faster-whisper, a vLLM server). A fake STT strategy and a stubbed OpenAI SDK check prompt assembly, the cache, and audio · frames strategy dispatch.

```bash
uv run python tests/smoke_asr.py        # strategy registry · cache · demux · batch · format
uv run python tests/smoke_client.py     # 3 audio modes · 2 frames modes · transcript injection position
uv run python tests/smoke_override.py   # benchmark overrides · asr merge · shared engine pool · config-key warnings
uv run python tests/smoke_transport.py  # file/base64 transport · re-encoding cache · inference block = legacy keys
uv run python tests/smoke_reasoning.py  # reasoning strategies · rescore (source untouched)
```

The paths that use real Whisper weights and a vLLM server must be checked separately.
