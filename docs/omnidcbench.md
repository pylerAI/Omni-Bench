# OmniDCBench

## Official Links

- Official GitHub: [yaolinli/TimeChat-Captioner](https://github.com/yaolinli/TimeChat-Captioner)
- arXiv: [TimeChat-Captioner: Scripting Multi-Scene Videos with Time-Aware and Structural Audio-Visual Captions](https://arxiv.org/abs/2602.08711)
- Project page: [TimeChat-Captioner](https://timechat-captioner.github.io/)
- Benchmark dataset: [yaolily/OmniDCBench](https://huggingface.co/datasets/yaolily/OmniDCBench)

![TimeChat-Captioner overview](assets/timechat_captioner_overview.png)

## Purpose

OmniDCBench evaluates omni dense captioning. It measures the ability to generate a continuous, timestamp-aware, fine-grained audio-visual narrative for multi-scene video.

## Dataset

- Local path: `/gpfs/public/datasets/OmniDCBench/`
- Official repo: `submodules/TimeChat-Captioner`
- Annotation: `/gpfs/public/datasets/OmniDCBench/ours_gt_file.json`
- Keypoint annotation: `/gpfs/public/datasets/OmniDCBench/ours_gt_file_keypoints.json`

The video archive must be extracted once before evaluation.

```bash
bash scripts/extract_omnidcbench_videos.sh
```

## Evaluation Method

The prompt found in the official TimeChat-Captioner inference code is the following dense caption prompt.

```text
Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.
```

There is no separate prompt in the official code instructing the model to emit timestamped JSON. Instead, `Infer/readme.md` and the evaluator assume the model output is already a JSON list containing timestamps. That assumes TimeChat-Captioner has been trained via SFT/GRPO to produce structured dense captions for the prompt above.

A general omni model given only that prompt may emit a natural-language paragraph, so the local adapter explicitly requests structured timestamped output that the official evaluator can read.

```text
Return only a valid JSON array. Do not include markdown fences or any extra text. Each array item must describe one temporal segment and include:
- "timestamp": a string in "MM:SS-MM:SS" format, relative to the start of this clip.
- "caption": a detailed audio-visual caption for that segment.
Use enough segments to cover the full video from beginning to end.
```

The adapter writes `predictions.jsonl`, preserving the fields the official `Eval` script expects:

- the original ground-truth fields
- `prediction`
- `prediction_json`, when the model output parses as JSON

Frame count and resolution follow the vLLM server's default video sampling. Per-request `num_frames`/`fps`/`max_pixels` are ignored by the server (changing them leaves the prompt token count unchanged, as verified), and OmniDCBench clips are short (≤70s), so they fit within the context window under default sampling. Only `use_audio_in_video` is applied per request. The model output is expected to be a structured dense caption JSON string with timestamps. Earlier natural-language paragraph predictions have no timestamp segments, so their F1/mIoU are 0 and they need to be re-inferred.

## Metrics

TimeChat-Captioner provides metric scripts under `submodules/TimeChat-Captioner/Eval`:

- `eval_sodam.py`
- `eval_time.py`

The local adapter runs `eval_time.py` and records the metrics in `summary.json`. The default configuration is `enable_sodam=false`, which computes only the temporal metrics (F1/mIoU) without a Gemini judge. To also compute Gemini-judge-based SODA_M, set the Gemini credentials (`metric_credentials`) and run with `enable_sodam=true`.

| Metric | What it measures |
| --- | --- |
| `SODA_M` | Temporal alignment and caption content matching of dense captions, jointly |
| F1 | How well predicted timestamp/caption segments correspond to GT segments |
| mean IoU | Mean overlap between predicted and GT timestamp spans |
| Checklist score | How much visual/acoustic/speech/camera-state information the caption covers, based on keypoints |

OmniDCBench annotations cover the following dimensions.

| Dimension | What it captures |
| --- | --- |
| `segment_detail_caption` | Whether the main visual/audio events within a segment are described in detail |
| `video_background` | Whether place, background, and scene context are described |
| `acoustics_content` | Whether audio information such as background music, sound, and tone is described |
| `shooting_style` | Whether camera/editing/shot style is described |
| `speech_content` | Whether spoken content is included accurately |
| `camera_state` | Whether camera motion/state such as pan, tilt, zoom, and shot scale is described |

## Final Output Metric Table

Final results produced through the official `Eval` script are reported in the following table.

| Model | Size | SODA_M | F1 | mean IoU | Segment Detail | Background | Acoustics | Shooting Style | Speech Content | Camera State | Avg Latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Omni | 30B | - | - | - | - | - | - | - | - | - | - |
| Nemotron-3-Nano-Omni | 30B | - | - | - | - | - | - | - | - | - | - |

The adapter writes `predictions.jsonl` for use as input to the official evaluator.
