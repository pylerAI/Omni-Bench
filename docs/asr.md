# ASR Transcripts

## Purpose

Some models under evaluation have no audio encoder (Qwen3.8-27B, Nemotron 3.5 Super VL). To compare them with omni models on audio-visual benchmarks, the audio is transcribed by a separate ASR engine and the transcript is injected into the prompt — a cascaded VLM + ASR setup.

This is selected per model with `inference.audio: asr_text` (see [Inference Strategies](inference_strategies.md)); the engine and transcript options come from the model config's `asr:` block. Transcripts are cached on disk, so each media file is transcribed once and every later run only looks it up.

## Engines

The engine is chosen by `asr.strategy.name`. All engines share one interface (`SttStrategy` in `src/omni_bench/asr/strategies.py`) and one output schema, so the cache and the prompt formatting do not depend on the engine.

| Engine | What it is | Key fields | Example config |
| --- | --- | --- | --- |
| `faster_whisper` | Whisper on CTranslate2. Default; fastest for bulk offline transcription | `model`, `device`, `device_index`, `compute_type`, `language`; `options` passed to `WhisperModel.transcribe` (defaults `beam_size 5`, `vad_filter true`, `condition_on_previous_text false`, `word_timestamps false`) plus `num_workers` | `configs/asr/whisper_large_v3.yaml`, and the `asr:` block of `configs/models/qwen3_8_27b_whisper*.yaml` / `configs/models/nemotron_3_5_super_vl_*.yaml` |
| `transformers_whisper` | Hugging Face `automatic-speech-recognition` pipeline. Slower; for cross-checking | `model`, `device`, `device_index`, `compute_type`; `options.chunk_length_s` (30), `options.batch_size` (16) | none shipped — set `strategy.name: transformers_whisper` in an `asr:` block |
| `vllm_asr` | An ASR model (e.g. Qwen3-ASR-1.7B) behind an OpenAI-compatible vLLM endpoint. Audio is cut into `chunk_s` pieces; each segment's timestamp is its chunk offset | `options`: `base_url`, `served_model_name`, `chunk_s` (30), `limit_s` (cap on audio read), `max_tokens` (448), `temperature` (0), `system_prompt`, `timeout_s` (300), `api_key` | `configs/asr/qwen3_asr.yaml` (server: `scripts/serve_qwen3_asr.sh`) |
| `nemo_streaming` | NVIDIA Nemotron streaming ASR (`nvidia/nemotron-3.5-asr-streaming-0.6b`). **Cache-only**: transcripts are produced elsewhere (NeMo pipeline in `riva_asr/`) and only read here; it never transcribes and errors on a cache miss | `model` (selects the cache namespace); use with `strict_cache: true` | `asr:` block of `configs/models/nemotron_3_5_super_vl_bf16_nothink_nemotron_asr.yaml` |

`asr:` block fields (`AsrSettings` in `src/omni_bench/asr/__init__.py`):

| Field | Default | Meaning |
| --- | --- | --- |
| `strategy` | `faster_whisper` · `Systran/faster-whisper-large-v3` · `cuda` · index 0 · `float16` | Engine spec: `name`, `model`, `device`, `device_index`, `compute_type`, `language` (`null` = auto-detect), `options`. Unknown keys are folded into `options` |
| `cache_dir` | `<repo>/cache/asr` | Cache root. Relative paths resolve against the repository root |
| `strict_cache` | `false` | `true`: a cache miss at evaluation time is an error |
| `with_timestamps` | `true` | Prefix each segment with `[mm:ss]` |
| `max_chars` | `null` | Keep only the last N characters of the transcript |
| `max_end_s` | `null` | Drop segments starting after this many seconds |
| `header` / `empty_text` | see [Prompt Injection](#prompt-injection) | Block header and no-speech line |

A benchmark config may carry its own `asr:` block; it is merged over the model's block, so only the changed fields need to be written (for example `max_chars` for long clips).

```yaml
# configs/asr/whisper_large_v3.yaml (excerpt)
asr:
  strategy:
    name: faster_whisper
    model: Systran/faster-whisper-large-v3
    device: cuda
    compute_type: float16
    language: null
    options:
      beam_size: 5
      vad_filter: true
      condition_on_previous_text: false
  cache_dir: /gpfs/public/artifacts/ail/omni-bench/cache/asr
  strict_cache: false
  with_timestamps: true
  max_chars: null
```

A new engine subclasses `SttStrategy`, implements `_transcribe`, and registers itself with `@register_strategy("name")`.

## Creating Transcripts

### Ahead of evaluation — `scripts/prepare_asr.py`

```bash
# every benchmark with audio in configs/benchmarks/default.yaml, all visible GPUs
uv run python scripts/prepare_asr.py --asr-config configs/asr/whisper_large_v3.yaml

# selected benchmarks and GPUs
uv run python scripts/prepare_asr.py --asr-config configs/asr/whisper_large_v3.yaml \
    --benchmark worldsense --benchmark omnivideobench --gpus 0,1,2,3 --threads-per-gpu 8

# count the work only
uv run python scripts/prepare_asr.py --asr-config configs/asr/whisper_large_v3.yaml --dry-run
```

- **GPU distribution**: files are split round-robin over `--gpus` (default: all visible). Each GPU gets one worker process with its own engine instance and a thread pool of `--threads-per-gpu` (default 8); for `faster_whisper` this also sets CTranslate2 `num_workers` unless the config pins it.
- **Resume**: files already in the cache are skipped, so an interrupted job can be re-run as is. Failures do not stop the batch; they are listed in the report (`<cache_dir>/prepare_asr_report.json`, or `--report`) and the exit code is 1.
- **Media collected** (from `--benchmark-config`, default `configs/benchmarks/default.yaml`):

  | Benchmark | Media roots |
  | --- | --- |
  | AV-SpeakerBench | `data_path` |
  | OmniDCBench | `video_dir` |
  | OmniVideoBench | `video_dir` + `preprocess_cache_dir` (default `<video_dir>/preprocess_cache`) |
  | WorldSense | `video_dir` + `preprocess_cache_dir` (default `<data_path>/preprocess_cache`) |
  | Video-MME | not collected — no transcript is injected |

  WorldSense and OmniVideoBench hand the client a `.wav` demuxed into the adapter's preprocess cache, not the original video, so those files are collected too. They exist only after the adapter has preprocessed once; the script prints a note when the directory is missing. `--media-root` adds more files or directories.
- **Overrides**: `--strategy`, `--model`, `--language`, `--cache-dir`, `--limit`.

### At evaluation time

For every request the client transcribes the audio the model would otherwise have received: the request's `audio_path`, else the original video (its audio track is demuxed with ffmpeg). With client-sampled frames the original video is not sent, so it is not transcribed either.

| `strict_cache` | On a cache miss |
| --- | --- |
| `false` (default) | Transcribe inline and write the cache. The engine is loaded once, at first need, and shared across benchmarks with the same engine settings |
| `true` | Fail the request (`FileNotFoundError: No cached transcript ...`); the error row is retried on the next run. Use after `prepare_asr.py` for deterministic runs |

## Storage

```text
<cache_dir>/<engine>__<model>/<key[:2]>/<key>.json
```

- `<engine>__<model>` is the namespace: the engine name, then `strategy.model` with leading/trailing `/` stripped and every `/` replaced by `__`. Each engine/model pair has its own namespace, so transcripts from different engines never mix.
- `key` is the SHA-1 of `"<resolved path>\0<size in bytes>\0<mtime_ns>"`. Moving, copying, re-extracting, or touching a media file changes the key, and the old entry is no longer found (it is not deleted).
- Writes go to a temporary file in the same directory followed by an atomic rename, so concurrent workers never see a partial file.
- Video transport (`inference.transport: base64`) may send a transcoded copy, but ASR always reads the original path, so the cache key is unaffected.

Shared cache root: `/gpfs/public/artifacts/ail/omni-bench/cache/asr/`. It lives on gpfs rather than in the repository (`/home/ail` is NFS) because the store is hot and the transcripts must outlive any checkout.

| Namespace | Files |
| --- | --- |
| `faster_whisper__Systran__faster-whisper-large-v3` | 12,660 |
| `nemo_streaming__nvidia__nemotron-3.5-asr-streaming-0.6b` | 11,367 |

## File Format

Each file is one `Transcription` (`src/omni_bench/asr/schema.py`), schema version `asr/1`.

| Field | Meaning |
| --- | --- |
| `schema_version` | `asr/1` |
| `media` | `path` (resolved), `size_bytes`, `mtime_ns`, `key` |
| `backend` / `model` | Engine name and model id |
| `language` / `language_probability` | Detected (or forced) language and its probability, when the engine reports them |
| `duration_s` | Media duration |
| `elapsed_s` | Transcription time (model loading excluded; it is recorded as `params.model_load_s`) |
| `params` | Engine settings used (device, compute type, options). Media without an audio stream get an empty transcript with `params.empty_reason: no_audio_stream` |
| `text` | All segment texts joined by spaces |
| `segments` | List of `index`, `start_s`, `end_s`, `text` |

Example (text and segments trimmed):

```json
{
  "schema_version": "asr/1",
  "media": {
    "path": "/gpfs/public/datasets/OmniVideoBench/videos/video_106.mp4",
    "size_bytes": 81378309,
    "mtime_ns": 1782960416725197286,
    "key": "c70ecc17255ac0be6def690184c2dbe65f12ba52"
  },
  "backend": "faster_whisper",
  "model": "Systran/faster-whisper-large-v3",
  "language": "en",
  "language_probability": 0.9951171875,
  "duration_s": 207.552,
  "elapsed_s": 13.770572671666741,
  "params": {"device": "cuda", "device_index": 1, "compute_type": "float16", "language": null,
             "beam_size": 5, "vad_filter": true, "condition_on_previous_text": false, "num_workers": 8},
  "text": "wait to get behind these guys ladies and gentlemen number 21 is the worst ranked swimmer ...",
  "segments": [
    {"index": 0, "start_s": 18.9, "end_s": 27.29,
     "text": "wait to get behind these guys ladies and gentlemen number 21 is the worst ranked"},
    {"index": 1, "start_s": 27.29, "end_s": 33.05,
     "text": "swimmer in American history in this field they range from number three"}
  ]
}
```

## Prompt Injection

`format_transcript` (`src/omni_bench/asr/format.py`) turns a transcript into one text block that is prepended to the benchmark's official prompt with one blank line between them. The official prompt itself is unchanged, so the official parsers are unaffected.

```text
Audio transcript of the media (speech recognised automatically):
[00:01] So the first thing you want to do is preheat the oven.
[00:05] Meanwhile, mix the flour and the sugar together.

<official prompt>
```

- **Header**: `header` (default above). Timestamps are `[mm:ss]` (`[hh:mm:ss]` past one hour), floored to the second; `with_timestamps: false` gives the plain text.
- **Length caps**: `max_end_s` drops segments that start after that time. `max_chars` keeps the **last** N characters and prefixes `...(truncated)...`, since answers in long clips tend to come late.
- **No speech**: no audio track, no detected speech, or nothing left after `max_end_s` → the block is the single line `empty_text` (default `Audio transcript: (no speech detected)`).
- **Per benchmark**: the transcript replaces the audio input. Video-MME runs with `inference.audio: none` in `configs/benchmarks/default.yaml` (the official protocol does not use audio), so nothing is injected there by default. AV-SpeakerBench and OmniDCBench transcribe the video's own audio track; WorldSense and OmniVideoBench transcribe their demuxed `.wav`.
- **Recorded per request**: `asr_chars`, the length of the injected block, is stored in each record (`null` when nothing was injected).

Assembled text part per benchmark, for a clip with one detected utterance (`<block>` is the transcript block above; options shortened):

| Benchmark | Media parts | Text part |
| --- | --- | --- |
| AV-SpeakerBench | `video_url` (original video) | `<block>` + `Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\nHow many people speak in the video?\nA. one ...\nThe best answer is:` |
| WorldSense | `image_url` × 8 (client frames, default) or `video_url` (`frames: server`) | `<block>` + `These are the frames of a video and the corresponding audio. Select the best answer ... \nQuestion: What is the speaker preparing?\nA. bread ...\nAnswer: ` |
| OmniVideoBench | JPEG-sequence `video_url` (client, default) or original `video_url` (`frames: server`); system prompt from the adapter | `<block>` + `You are given a video. Based on the content of the video, answer the following question:\n\nQuestion:\n... \n\nOptions:\nA. ...\n\nAnswer with the option's letter directly(e.g., A, B, C, or D)...` |
| OmniDCBench | `video_url` (original video) | `<block>` + `Thoroughly describe everything in the video, capturing every detail. ...` + the timestamped-JSON instruction and clip-length cap |
| Video-MME | `image_url` × 64 (client, default) or `video_url` (`frames: server`) | No block (`inference.audio: none`): `Select the best answer ... \n<question>\nA. ...\nThe best answer is:` |

A clip with no speech gets `Audio transcript: (no speech detected)` + blank line + the official prompt.
