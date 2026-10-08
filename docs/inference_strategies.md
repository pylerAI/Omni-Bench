# Inference Strategies

## Purpose

Models differ in how they take inputs and return answers: some accept audio and others need an ASR transcript, some sample video frames on the server and others need client-sampled frames, and some servers read local files while remote ones need base64 data. These choices are set in the model config's `inference:` block, so the run command stays the same for every model:

```bash
uv run omni-bench run \
  --config configs/models/<model>.yaml \
  --benchmark worldsense
```

A model config may also name its benchmark config with the top-level key `benchmark_config:` (path relative to the model config file), so one `--config` describes a whole measurement. Precedence: `--benchmark-config` on the command line > `benchmark_config:` > `configs/benchmarks/default.yaml`. The resolved path is saved as `benchmark_config_path` in `config_used.json`.

```yaml
benchmark_config: bench_thinking.yaml
models:
  - name: qwen3.8-27b-whisper-srvthink
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

For `audio: asr_text`, the engine, the transcript cache, pre-transcription with `scripts/prepare_asr.py`, and how the transcript enters the prompt are described in [ASR Transcripts](asr.md).

## Re-scoring Stored Runs

`omni-bench rescore` applies a reasoning strategy to stored records and re-runs each adapter's parser and summary without inference. It never modifies the source directory. For runs without `config_used.json`, pass the model config with `--config` (its `benchmark_config:` is used) or `--benchmark-config`.

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
