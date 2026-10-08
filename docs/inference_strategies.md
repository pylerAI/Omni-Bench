# Inference Strategies

## Purpose

Models differ in how they take inputs and return answers: some accept audio and others need an ASR transcript, some sample video frames on the server and others need client-sampled frames, and some servers read local files while remote ones need base64 data. These choices are set in the model config's `inference:` block, so the run command stays the same for every model:

```bash
uv run omni-bench run \
  --config configs/models/<model>.yaml \
  --benchmark-config configs/benchmarks/default.yaml \
  --benchmark worldsense
```

Adapters do not branch on these choices. Prompts, answer parsing, and scoring are the same for every strategy.

## Configuration

```yaml
models:
  - name: nemotron-3.5-super-vl-bf16
    inference:
      audio: asr_text
      frames: server
      transport: base64
      strip_mm_kwargs: true
      reasoning: server
```

| Axis | Options | Default | Meaning |
| --- | --- | --- | --- |
| `audio` | `native` · `none` · `asr_text` | `native` | Send audio as is · drop it · inject an ASR transcript (engine from `asr.strategy.name`). A benchmark config may override it. |
| `frames` | `client` · `server` | per benchmark | Client-sampled frames · original video sampled by the model server. Benchmark config > model config > benchmark default. |
| `transport` | `file` · `base64` | `file` | `file://` URL · base64 data URL (large files via the `transcode:` cache). Model config only. |
| `strip_mm_kwargs` | `true` · `false` | `false` | Drops `mm_processor_kwargs` / `media_io_kwargs`, which some remote servers reject. Model config only. |
| `reasoning` | `server` · `think_tag` · `none` | `server` | Server already splits reasoning · answer is the text after the last `</think>` (servers without a reasoning parser) · no processing. Model config only. |

The old flat keys `audio_mode`, `frame_sampling`, and `video_transport` are still accepted. Unknown config keys print a warning, and typos inside `inference:` raise an error. The resolved values are printed per benchmark and saved in `config_used.json`.

## ASR Transcripts

For `audio: asr_text`, pre-fill the transcript cache so evaluation is not blocked on ASR:

```bash
uv run python scripts/prepare_asr.py --asr-config configs/asr/whisper_large_v3.yaml
```

Available ASR engines: `faster_whisper`, `transformers_whisper`, `vllm_asr`, and `nemo_streaming` (reads an existing cache only). See [qwen3_8_whisper.md](qwen3_8_whisper.md) for details.

## Re-scoring Stored Runs

`omni-bench rescore` applies a reasoning strategy to stored records and re-runs each adapter's parser and summary without inference. It never modifies the source directory.

```bash
uv run omni-bench rescore \
  --run-dir results/<model> \
  --out-dir results_rescored \
  --reasoning think_tag
```

## Adding a Strategy

Each axis is a registry in `src/omni_bench/inference/`. Subclass the axis base class, then register it by name to make it selectable from YAML:

```python
@TRANSPORTS.register("s3")
class S3Transport(Transport):
    def video_url(self, path): ...
```
