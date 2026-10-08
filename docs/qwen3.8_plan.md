# Qwen3.8-27B + Whisper Evaluation

## Purpose

On AAII bench, Qwen3.8-27B scored 52, matching GPT-5.6-Luna in language performance. This evaluation checks whether the same model's multimodal ability exceeds that of dedicated omni models, and if it does, whether it should replace the backbone of the downstream task (AiD Video).

Qwen3.8-27B is a vision + text model with **no audio encoder**. Audio is replaced by a Whisper transcript, so this experiment compares **native omni vs cascaded VLM + ASR**.

> **Conclusion — with thinking on, the swap is worth it.** With thinking, it beats Qwen3-Omni-30B on all 4 benchmarks and reaches Qwen3.5-Omni-Flash level. However, **all of the gain comes from thinking** — non-thinking is on par or behind. The cost is 4~25x the run time.

Implementation details are in [Qwen3.8-27B + Whisper (Cascaded ASR)](qwen3_8_whisper.md).

## Model

| Item | Value |
| --- | --- |
| Architecture | `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) |
| Text backbone | 64-layer hybrid (linear attention 3 : full attention 1) · hidden 5120 · GQA 24/4 |
| Context | 262,144 (mRoPE interleaved) |
| Vision tower | 27 layers · hidden 1152 · patch 16 · out 5120 |
| Parameters | **dense 27B** (the comparison target Qwen3-Omni is MoE 30B-A3B, active 3B) |
| Audio | None — replaced by a Whisper large-v3 cascade |
| Weights | `/gpfs/public/artifacts/models/Qwen/Qwen3.8-27B/` |

## Evaluation Protocol

Only the Qwen-recommended settings (the recommend config) are used — not the earlier evalkit-based setup (fixed frames · `temperature 0`).

### Sampling

| Item | non-think | thinking |
| --- | --- | --- |
| `temperature` | 0.7 | 1.0 |
| `top_p` / `top_k` | 0.80 / 20 | 0.95 / 20 |
| `min_p` | 0.0 | 0.0 |
| `presence_penalty` | 1.5 | 0.0 |
| `repetition_penalty` | 1.0 | 1.0 |
| `enable_thinking` | false | **true** |
| `max_tokens` | 4,096 | **32,768** |

For thinking, `max_tokens` started at 8,192, but **4.4~13.9% were truncated** (cut off mid-reasoning, losing the final answer), so it was raised to 32,768 and everything was re-measured → 0 truncations.

### Input Handling — `inference.frames: server`

Frames are not extracted on the client. **The original video is sent as is** and the model processor decides. The values are exactly those in the model weights' `video_preprocessor_config.json`.

| Item | Value | Meaning |
| --- | --- | --- |
| `fps` | 2 | 2 frames per second |
| `max_frames` | 768 | Frame-count cap — hit beyond 384 s (6.4 min) |
| `min_frames` | 4 | Lower bound |
| `size.longest_edge` | 25,165,824 px | Pixel budget for **the whole video** (not per frame) |
| `patch_size` / `merge_size` | 16 / 2 | 1 token = 32×32 px |
| `temporal_patch_size` | 2 | 2 adjacent frames grouped into one 3D patch → half the tokens |

The pixel budget applies to the whole video. In terms of resolution:

| Resolution | px per frame | Frames that fit the budget | Tokens per frame |
| --- | --- | --- | --- |
| 1280×720 (720p source) | 921,600 | **27** | 450 |
| 854×480 (480p) | 409,920 | 61 | 200 |
| 640×360 (360p) | 230,400 | 109 | 112 |
| 480×256 | 122,880 | 204 | 60 |
| 224×128 | 28,672 | 877 | 14 |

**Keeping 720p allows only 27 frames; filling 768 frames requires shrinking to 224×128.** The processor picks the latter — it trades spatial resolution for temporal resolution. Measured values (720p source):

| Video length | Requested frames (fps 2) | Actual frames | Frame resolution | Video tokens |
| --- | --- | --- | --- | --- |
| 97 s | 194 | 194 | 256×480 | 11,640 |
| 500 s | 1,000 | 768 (cap) | 128×224 | 10,752 |
| 2,820 s | 5,639 | 768 (cap) | 128×224 | 10,752 |

- **Beyond 6.4 min the frame count stops and only the resolution keeps dropping** — 500 s and 2,820 s get the same resolution
- The earlier evalkit setup: 64 fixed frames · per-frame `max_pixels 602,112` (about 768×720) · 64 `image_url` parts (each image is independent, so the `temporal_patch` merging gain is 0) → 34,141 prompt tokens
- The server setup: **13,734** prompt tokens (40%) · **2.2x faster** wall time · **long-form answers drop from 30.9% to 0.2%, removing the parsing defect** (official/improved parser gap +12.30 → +0.07)

### Other Settings

| Item | Value |
| --- | --- |
| Serving | `MAX_MODEL_LEN` 131,072 · B200×4 (DP=4) · TP=1 · `gpu_memory_utilization` 0.9 |
| Concurrency | 12 |
| ASR | faster-whisper `large-v3` · fp16 · `beam_size 5` · `vad_filter true` · language auto · **task = transcribe** |
| Video-MME | **No ASR injection** — the official protocol does not use audio as input, and the comparison models run under the same condition |

`translate` is not used because it would give only the cascade a translation step that omni models do not have.

## Results

| Benchmark | Our model | non-think | thinking | Qwen3-Omni 30B | Qwen3.5-Omni Flash | Qwen3.5-Omni Plus | Gemini |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Video-MME (w/o sub) | Qwen3.8-27B | 66.89 | **77.33** | 70.5 | 77.0 | 81.9 | 79.6 (2.5-Flash-Thinking) |
| WorldSense | Qwen3.8-27B + Whisper | 47.48 | **61.32** | 54.0 | 57.9 | 62.8 | 65.5 (3.1 Pro) |
| OmniVideoBench | Qwen3.8-27B + Whisper | 42.70 | **52.50** | 38.40 | — | — | 57.83 (2.5-Pro) |
| AV-SpeakerBench | Qwen3.8-27B + Whisper | 50.34 | **65.35** | — | 65.2 | 71.3 | 75.1 (3.1 Pro) |

Qwen3.5-Omni and Gemini have **closed weights · paid APIs**, so they are not backbone candidates. The only replaceable counterpart is **Qwen3-Omni-30B**, and with thinking our model beats it on all 4 benchmarks.

### All of the Gain Comes from Thinking

| Benchmark | non-think | thinking | Δ |
| --- | --- | --- | --- |
| Video-MME | 66.89 | **77.33** | **+10.44** |
| WorldSense | 47.48 | **61.32** | **+13.84** |
| OmniVideoBench | 42.70 | **52.50** | **+9.80** |
| AV-SpeakerBench | 50.34 | **65.35** | **+15.01** |

The net effect was isolated by changing one variable at a time.

| Variable | Effect |
| --- | --- |
| Sampling parameters (temp 0 → recommended) | −0.13 ~ +2.90 → **no effect** |
| Frame sampling (fixed → server default) | −1.00 ~ +0.15 → **no effect** |
| **thinking** | **+9.80 ~ +15.01** |

**For Qwen3-Omni, thinking actually hurts** — Video-MME 70.5 → 69.7 (−0.8) · LVBench 50.2 → 49.0 · MLVU 75.2 → 72.9. This appears to be the difference between an active-3B MoE and a dense 27B. **Whether inference time can be converted into accuracy is what separates the two models.**

### Cost

| Benchmark | n | non-think | thinking | Ratio | Generated tokens |
| --- | --- | --- | --- | --- | --- |
| Video-MME | 2,700 | 20.5 min | 138.9 min | 6.8× | 2.5 → 1,891 |
| WorldSense | 3,172 | 21.8 min | 127.4 min | 5.8× | 5.2 → 1,869 |
| OmniVideoBench | 1,000 | 25.6 min | 86.7 min | 3.4× | 2.0 → 3,914 |
| AV-SpeakerBench | 3,212 | 8.1 min | 206.1 min | **25.4×** | 2.0 → 3,667 |

0 errors. GPU utilization is about 8% — the bottleneck is **CPU video decoding** in the vLLM APIServer (NVDEC is not used). Raising concurrency from 12→48 halves throughput (thrashing).

### Sources of Comparison Numbers

- Qwen3-Omni 30B — [Qwen3-Omni Technical Report](https://arxiv.org/html/2509.17765) Table 9 (Video-MME) · Table 11 (WorldSense) · Table 10 (Instruct vs Thinking)
- Qwen3.5-Omni Flash · Plus — [Qwen3.5-Omni Technical Report](https://arxiv.org/html/2604.15804v1) Table 6 (Video-MME) · Table 7 (WorldSense · AV-SpeakerBench)
- Gemini — the same tables of the two reports above; OmniVideoBench only from [OmniVideoBench](https://arxiv.org/html/2510.10689) Table 3
- There is no Qwen3-Omni number for AV-SpeakerBench — the benchmark was released (2025-12) after the report (2025-09)

As for inference mode, **only Qwen3-Omni is confirmed non-thinking** (Instruct variant); the reports do not state it for Qwen3.5-Omni or Gemini. The Gemini column uses a different generation per row, so it cannot be compared vertically; only the best version per benchmark is listed.

### Not Measured — OmniDCBench

**60.8% of thinking outputs were truncated** (the long caption JSON and the reasoning both consume `max_tokens`), and backfilling is unfinished. A checkpoint of 712/1,122 is preserved. Non-thinking scores F1 0.483, and this is the only benchmark where the transcript has a negative (−) effect — 89% of the audio is Chinese while English captions and timestamp JSON are required.

## Running Evaluations

```bash
export UV_PROJECT_ENVIRONMENT=/gpfs/private/ail/venvs/omni-bench
export ALLOWED_LOCAL_MEDIA_PATH=/gpfs/public
export OMNI_BENCH_RESULT_DIR=/gpfs/public/artifacts/ail/omni-bench/runs/results

# 1. Transcribe ASR ahead of time — must not overlap with vLLM in time (sharing a GPU kills the EngineCore)
uv run python scripts/prepare_asr.py \
    --asr-config configs/asr/whisper_large_v3.yaml \
    --gpus 0,1,2,3 --threads-per-gpu 8

# 2. Serving
MAX_MODEL_LEN=131072 VLLM_BIN=$UV_PROJECT_ENVIRONMENT/bin/vllm \
    bash scripts/serve_qwen3_8_27b.sh

# 3. Evaluation (bench = videomme | worldsense | omnivideobench | av_speakerbench)
uv run omni-bench run --config configs/recommend/qwen3_8_27b_whisper_thinking.yaml \
    --benchmark-config configs/recommend/bench_thinking.yaml   --benchmark <bench>   # thinking
uv run omni-bench run --config configs/recommend/qwen3_8_27b_whisper_nothink.yaml \
    --benchmark-config configs/recommend/bench_nothink.yaml --benchmark <bench>   # non-think

# 4. Backfill truncated items — after removal, re-running step 3 retries only those items
python scripts/strip_truncated.py all
```

## Result Locations

```text
/gpfs/public/artifacts/ail/omni-bench/runs/
  results/qwen3.8-27b-srv-think/            Video-MME thinking
  results/qwen3.8-27b-srv-nothink/          Video-MME non-think
  results/qwen3.8-27b-whisper-srvthink/     4 benchmarks, thinking
  results/qwen3.8-27b-whisper-srvnothink/   4 benchmarks, non-think
  results/qwen3-omni-repro/                 Qwen3-Omni reproduction (comparison baseline)
  archive/                                  exploration and intermediate experiment outputs
```

Each `<benchmark>/` holds `records.jsonl` + `summary.json`. The `perf` block of `summary.json` records `wall_s` · `samples_per_s` · `latency_s` (mean/p50/p90/p99/max) · the `prompt_tokens`/`completion_tokens` distributions. `wall_s` includes frame decoding and transcript lookup and predicts re-run time; `latency_s` includes server queueing, so compare it only within the same run.

The 12,660 entries of the ASR cache `/gpfs/public/artifacts/ail/omni-bench/cache/asr/` can be reused.
