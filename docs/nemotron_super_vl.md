# Nemotron 3.5 Super VL

> **NDA** — This is an NVIDIA Early Access checkpoint (GA expected 2026-10-15). Do not share the model name, specs, or results outside the company.

## Purpose

Nemotron 3.5 Super VL, received through NVIDIA Early Access, is compared with the existing candidates (Qwen3.8-27B, Nemotron-3-Nano-Omni, Qwen3-Omni) on the same benchmarks. The evaluation also covers:

- the effect of think / no-think
- quantization loss (BF16 vs NVFP4)
- the effect of swapping the ASR (Whisper vs Nemotron ASR)

Super VL is a vision + text model with **no audio encoder**. As with Qwen3.8, audio is replaced by a Whisper transcript (cascaded VLM + ASR).

> **Conclusion — on par with the existing open models, and not clearly ahead even with think.**
> - No-think is on par with Qwen3.8-27B and Nemotron-Omni.
> - The think gain is +1.6 ~ +6.6, smaller than Qwen3.8 (+9.8 ~ +15.0).
> - NVFP4 quantization loss is within ±1.2, which is negligible.

## Model

| Item | Value |
| --- | --- |
| Model | Nemotron 3.5 Super VL — 120B MoE · active 12B · supports thinking / tool calling |
| Checkpoint | GA candidate build 2026-10-01 (`nemotron_3_5_super_ga_candidate_mtp_boosted[_nvfp4]_20261001_vv0.1`) |
| Input | text · image · video (**no audio**) |
| Serving | Remote vLLM 0.31.0 run by the Platform team (sel2 `vllm` namespace) · `max_model_len` 65,536 |
| Endpoint | BF16 `http://vllm-nemotron-3-5-super-vl-bf16.vllm:8000/v1` (B200×2) · NVFP4 `http://vllm-nemotron-3-5-super-vl-nvfp4.vllm:8000/v1` (B200×1) |

`*.vllm.idc.k8s` addresses do not resolve in DNS inside cluster Pods, so the in-cluster names above are used.

## Evaluation Method

The input conditions are the same as the Qwen3.8 recommend config ([qwen3.8_plan.md](qwen3.8_plan.md)). Only the model sampling and the transport required by the remote server's constraints were changed.

### Sampling

Super VL has no public model card, so the official recommended values of the same family, Nemotron 3 Super, are used (shared by think and no-think).

| Item | no-think | think |
| --- | --- | --- |
| `temperature` / `top_p` | 1.0 / 0.95 | 1.0 / 0.95 |
| Others (`top_k`, `presence_penalty`, ...) | Not sent (server defaults) | Same |
| `enable_thinking` | false | **true** |
| `max_tokens` | 4,096 | **32,768** (from the start) |

For think responses, the server's reasoning parser splits the output into `message.reasoning` and `content`. Only the answer remains in `content`, so unlike Qwen3.8 think, scoring uses **the official parser directly** (`rescore_mcq.py` is not needed). The configs therefore keep the default `inference.reasoning: server`.

No-think occasionally emits inline reasoning ending in `</think>` inside `content` (Video-MME: BF16 20 items, NVFP4 1). The reported numbers keep the `server` setting. Re-scoring with `omni-bench rescore --reasoning think_tag` gives Video-MME BF16 67.30 → 67.52 and NVFP4 67.11 → 67.07; no other benchmark changes.

### Remote Server Constraints

| Constraint | Handling |
| --- | --- |
| Cannot read `file://` paths (no `--allowed-local-media-path`) | `inference.transport: base64` — videos are sent as base64 data URLs |
| Request bodies too large (Video-MME up to 922MB) | Only files above 200MB are re-encoded to ≤720p and cached (fps kept, audio removed). Affected: Video-MME 156 · OmniVideoBench 87 |
| `mm_processor_kwargs` / `media_io_kwargs` cause HTTP 400 | `inference.strip_mm_kwargs: true` — those keys are removed from the request |
| Audio input rejected | `inference.audio: asr_text` (Video-MME is `none` in the benchmark config) |
| No per-request frame settings | Server defaults are used. **Fixed at about 4.6k tokens per video** (presumably vLLM's default 32 frames, unconfirmed) |

The video input volume is less than half of Qwen3.8's (`inference.frames: server`, about 10~13k tokens). Only the Platform team can change the server setting (`--media-io-kwargs`).

### Other Settings

| Item | Value |
| --- | --- |
| `inference.frames` | `server` (benchmark config · same as Qwen3.8) |
| ASR | Reuses the existing faster-whisper `large-v3` cache (`strict_cache: true`, 12,660 entries) |
| Video-MME | No ASR injection · no subtitles (official protocol) |
| OmniVideoBench system prompt | Adapter default as is (same condition as the Qwen3.8 · Nemotron-Omni runs) |
| Concurrency | no-think 24 · think 48 (no effect on results) |

## Results

All runs had 0 errors. Sources of the comparison models:

- Nemotron-Omni · Qwen3-Omni: [report.html](../report.html)
- Qwen3.8: [qwen3.8_plan.md](qwen3.8_plan.md)

### Model Comparison (BF16 · no-think)

| Benchmark | Super VL | Nemotron-Omni (BF16) | Qwen3.8-27B | Qwen3-Omni 30B |
| --- | --- | --- | --- | --- |
| Video-MME (w/o sub) | 67.30 | 67.81 | 66.89 | **70.19** |
| WorldSense | **52.21** | 50.63 | 47.48 | 51.36 |
| OmniVideoBench | 42.40 | 40.10 | **42.70** | 41.20 |
| AV-SpeakerBench | 47.42 | 50.28 | 50.34 | **55.98** |

AV-SpeakerBench is the lowest among the compared models. Many of its questions identify the speaker, which penalizes a cascade whose transcript carries no speaker information.

### think / no-think (BF16)

| Benchmark | Super VL no-think | Super VL think | Δ | Qwen3.8 no-think | Qwen3.8 think | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Video-MME | 67.30 | 69.19 | +1.89 | 66.89 | 77.33 | +10.44 |
| WorldSense | 52.21 | 55.42 | +3.22 | 47.48 | 61.32 | +13.84 |
| OmniVideoBench | 42.40 | 44.00 | +1.60 | 42.70 | 52.50 | +9.80 |
| AV-SpeakerBench | 47.42 | 54.02 | +6.60 | 50.34 | 65.35 | +15.01 |

The think gain is much smaller than Qwen3.8's. Accuracy dropped the longer the model reasoned: on WorldSense, generations under 300 tokens score 60.2% and those over 4k tokens score 37.0%.

One likely cause is the visual input being capped at about 4.6k tokens. Scenes missing from the input cannot be recovered by longer reasoning.

### Quantization (Δ = NVFP4 − BF16)

| Benchmark | BF16 think | NVFP4 think | Δ | BF16 no-think | NVFP4 no-think | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Video-MME | 69.19 | 68.30 | −0.89 | 67.30 | 67.11 | −0.19 |
| WorldSense | 55.42 | 54.57 | −0.85 | 52.21 | 51.61 | −0.60 |
| OmniVideoBench | 44.00 | 45.20 | +1.20 | 42.40 | 42.50 | +0.10 |
| AV-SpeakerBench | 54.02 | 54.33 | +0.31 | 47.42 | 47.20 | −0.22 |

All 8 entries are within ±1.2. Serving with NVFP4 (B200×1) costs almost no accuracy.

### ASR Comparison (BF16 · no-think, Δ = Nemotron − Whisper)

| Benchmark | Whisper large-v3 | nemotron-3.5-asr-streaming-0.6b | Δ |
| --- | --- | --- | --- |
| WorldSense | 52.21 | 50.25 | −1.95 |
| OmniVideoBench | 42.40 | 41.10 | −1.30 |
| AV-SpeakerBench | 47.42 | 46.86 | −0.56 |

A qualitative comparison of transcript quality:

| Model | Strengths | Weaknesses |
| --- | --- | --- |
| Nemotron ASR | No hallucinations, no translation into the wrong language, no long dropped spans | Spells numbers out as English words; weak on overlapping speech and non-English (Chinese, etc.) |
| Whisper | Strong on proper nouns, numbers, multi-speaker dialogue | Hallucinates on silent/music segments ("Thank you for watching", etc.) |

4 videos in the Nemotron cache (OmniVideoBench `video_93/532/534/539`) were generated without timestamps. They affect 5 of 1,000 questions.

### Truncation (think, `finish_reason == length`, max_tokens 32,768)

| Benchmark | BF16 | NVFP4 |
| --- | --- | --- |
| Video-MME | 35 | 30 |
| WorldSense | 14 | 8 |
| OmniVideoBench | 13 | 18 |
| AV-SpeakerBench | 6 | 8 |

This is 0.2~1.8% of the total. Truncated items were counted as wrong (not re-measured).

### Cost

| Item | no-think | think |
| --- | --- | --- |
| Generated tokens (mean) | 2~15 | 1,500~2,900 |
| Total duration | about 5~6 hours per configuration | about 12~14 hours per configuration |

The bottleneck is **CPU video decoding** in the server's API processes. Video-MME long videos (30~60 min) in particular took about 4 minutes per question.

- Raising concurrency does not increase the number of requests reaching the GPU. 20~40 requests were in flight, and KV cache usage was below 10%.
- Fixing this requires raising the server-side `--api-server-count` (a Platform team setting).

## Implementation (Changes vs `qwen-3.8`)

Scoring logic (answer extraction · accuracy aggregation · reports) is the same as `main`. The changes are limited to request transport and run conveniences.

| Feature | File | Description |
| --- | --- | --- |
| base64 video transport | `src/omni_bench/inference/transport.py`, `src/omni_bench/video_transport.py` | Model config `inference.transport: file \| base64`. With base64, files over the `transcode:` threshold use the re-encoding cache. The cache key is path · size · mtime · re-encoding parameters, and concurrent access is safe |
| Pre-building re-encodes | `scripts/pretranscode_videos.py` | Builds the cache in parallel before evaluation. On a cache miss the client builds the entry on the spot |
| mm kwargs removal | `src/omni_bench/inference/__init__.py` | With `inference.strip_mm_kwargs: true`, `mm_processor_kwargs` · `media_io_kwargs` are removed from the request |
| Reasoning capture | `src/omni_bench/client.py`, `adapters/*` | Stores `reasoning` · `finish_reason` · token counts in the records |
| Error-row retry | `src/omni_bench/io.py`, `adapters/*` | A failed request no longer stops the whole benchmark; it is recorded as an error row. Re-running retries only the error rows |
| Run options | `src/omni_bench/cli.py` | `--limit N`, `--limit-mode head\|spread` (N items spread across the whole set), `--result-dir`, run config snapshot `config_used.json` (including the git commit) |
| Truncation cleanup | `scripts/strip_truncated.py` | Supports `--run-dir`; judges by `finish_reason == length` |
| Nemotron ASR | `src/omni_bench/asr/nemo_streaming.py` | ASR strategy `nemo_streaming` that only reads the cache. Errors if the cache is missing |

## Running Evaluations

```bash
export UV_PROJECT_ENVIRONMENT=/gpfs/private/ail/venvs/omni-bench

# 0. (optional) pre-build the re-encoding cache for large videos
uv run --no-sync python scripts/pretranscode_videos.py \
    --config configs/nemotron_super_vl/bf16_nothink.yaml \
    --video-dir /gpfs/public/datasets/Video-MME/data \
    --video-dir /gpfs/public/datasets/OmniVideoBench/videos

# 1. Smoke — 20 items per benchmark, spread across the whole set
uv run --no-sync omni-bench run --config configs/nemotron_super_vl/bf16_nothink.yaml \
    --benchmark-config configs/nemotron_super_vl/bench_nothink.yaml --benchmark worldsense \
    --limit 20 --limit-mode spread --result-dir /tmp/smoke

# 2. Full run — one measurement config file = one run. One benchmark per --benchmark (omit for the whole benchmark config)
#    model config: {bf16,nvfp4}_{nothink,think}.yaml / bench config: bench_{nothink,think}.yaml
uv run --no-sync omni-bench run --config configs/nemotron_super_vl/bf16_nothink.yaml \
    --benchmark-config configs/nemotron_super_vl/bench_nothink.yaml --benchmark worldsense

# 3. ASR comparison (Nemotron ASR cache, BF16 · no-think, excluding Video-MME)
uv run --no-sync omni-bench run --config configs/nemotron_super_vl/bf16_nothink_nemotron_asr.yaml \
    --benchmark-config configs/nemotron_super_vl/bench_nothink.yaml --benchmark worldsense
```

The model is served remotely only, so `--serve` is not used. During measurement, `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` was set to block outbound traffic, and the 4 benchmarks (videomme · worldsense · omnivideobench · av_speakerbench) were run sequentially with the commands above. Long runs go to the background with `nohup … > <log> 2>&1 &`.

| Config | Purpose |
| --- | --- |
| `configs/models/nemotron_3_5_super_vl_{bf16,nvfp4}.yaml` | Base model settings (endpoint · transport · sampling · Whisper ASR cache). no-think by default |
| `configs/nemotron_super_vl/{bf16,nvfp4}_{nothink,think}.yaml` | Settings used for measurement (the counterpart of Qwen3.8's `configs/recommend/`). Same request content as the model settings above; only the result location and think flag differ |
| `configs/nemotron_super_vl/bench_{nothink,think}.yaml` | Benchmark settings (max_tokens · temperature · concurrency) |
| `configs/nemotron_super_vl/bf16_nothink_nemotron_asr.yaml` | Same settings with only the ASR switched to `nemo_streaming` |

The model settings under `configs/models/` were verified to produce the same requests as the measurement settings: with the OpenAI SDK replaced by a stub, the actual request bodies were compared and matched for both BF16 and NVFP4.

Re-running the same command skips finished items and retries only the error rows.

## Result Locations

```text
/gpfs/public/artifacts/ail/nemotron-compare/
  README.md · summary.json                          full results summary (including machine-readable JSON)
  omni-bench/
    nemotron-3.5-super-vl-bf16-nothink/              2.1 · 2.2 · 2.3 · 2.4(Whisper)
    nemotron-3.5-super-vl-bf16-think/                2.2 · 2.3
    nemotron-3.5-super-vl-nvfp4-nothink/             2.3
    nemotron-3.5-super-vl-nvfp4-think/               2.3
    nemotron-3.5-super-vl-bf16-nothink-nemotron-asr/ 2.4(Nemotron ASR)
      <benchmark>/records.jsonl · summary.json · config_used.json
    logs/ · _smoke/
  cache/transcode/                                   re-encoded copies of videos over 200MB
```

The `2.x` labels refer to the Results subsections in order: 2.1 Model Comparison · 2.2 think / no-think · 2.3 Quantization · 2.4 ASR Comparison.

Shared ASR caches:

- Whisper: `/gpfs/public/artifacts/ail/omni-bench/cache/asr/faster_whisper__Systran__faster-whisper-large-v3/`
- Nemotron: `/gpfs/public/artifacts/ail/omni-bench/cache/asr/nemo_streaming__nvidia__nemotron-3.5-asr-streaming-0.6b/`
