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
benchmark_config: ../benchmarks/qwen3_8_27b_thinking.yaml
models:
  - name: qwen3.8-27b-whisper-srvthink
```

### Shared bases (`extends:`)

A model config can build on base configs with the top-level key `extends:`, a path or a list of paths relative to the file. Bases can extend further; a cycle is an error.

- Merge order: the bases in list order, then the file itself. A later value wins.
- Mappings merge key by key (deep); any other value, lists included, is replaced wholesale.
- `models:` when the base and the child each hold exactly one model, the child's entry is deep-merged onto the base's (so a child usually writes only `name`, `display_name`, and what differs). Otherwise the child's list replaces the base's.
- Path keys (`benchmark_config`, `global.result_dir`) resolve against the file that declares them.
- Unknown keys are checked on the merged result, so a typo in a child is still reported.
- `config_used.json` records the merged settings and `config_chain`, every file that contributed (base first).

Bases live in `configs/models/_base/` and are not run on their own. For example, a Super VL measurement config is the Super VL base plus the Whisper mixin plus a few lines:

```yaml
extends: [_base/nemotron_3_5_super_vl.yaml, _base/asr_whisper_large_v3.yaml]
benchmark_config: ../benchmarks/nemotron_3_5_super_vl_think.yaml
models:
- name: nemotron-3.5-super-vl-bf16-think
  display_name: "Nemotron-3.5-Super-VL-BF16 + Whisper (thinking)"
  served_model_name: nemotron-3.5-super-vl-bf16
  base_url: http://vllm-nemotron-3-5-super-vl-bf16.vllm:8000/v1
  extra_body:
    chat_template_kwargs:
      enable_thinking: true
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
      reasoning: as_is
```

| Axis | Options | Default | Meaning |
| --- | --- | --- | --- |
| `audio` | `native` · `none` · `asr_text` | `native` | Send audio as is · drop it · inject an ASR transcript (engine from `asr.strategy.name`). A benchmark config may override it. |
| `frames` | `client` · `server` | per benchmark | Client-sampled frames · original video sampled by the model server. Benchmark config > model config > benchmark default. |
| `transport` | `file` · `base64` | `file` | `file://` URL · base64 data URL (large files via the `transcode:` cache). Model config only. |
| `strip_mm_kwargs` | `true` · `false` | `false` | Drops `mm_processor_kwargs` / `media_io_kwargs`, which some remote servers reject. Model config only. |
| `reasoning` | `as_is` · `split` | `as_is` | Content is the answer as received (a separate reasoning field from the server is recorded) · answer is the text after the last `</think>`, the rest is recorded as reasoning (servers without a reasoning parser). Model config only. |

The old flat keys `audio_mode`, `frame_sampling`, and `video_transport` are still accepted. Unknown config keys print a one-line `warning:` to stderr, and typos inside `inference:` raise an error. The resolved values are printed per benchmark and saved in `config_used.json`.

## ASR Transcripts

For `audio: asr_text`, the engine, the transcript cache, pre-transcription with `scripts/prepare_asr.py`, and how the transcript enters the prompt are described in [ASR Transcripts](asr.md).

## Re-scoring Stored Runs

`omni-bench rescore` applies a reasoning strategy to stored records and re-runs each adapter's parser and summary without inference. It never modifies the source directory. For runs without `config_used.json`, pass the model config with `--config` (its `benchmark_config:` is used) or `--benchmark-config`.

```bash
uv run omni-bench rescore \
  --run-dir results/<model> \
  --out-dir results_rescored \
  --reasoning split
```

Related scripts for thinking runs:

- `scripts/strip_truncated.py <benchmark|all> --run-dir results/<model> [--dry-run]` removes records whose final answer was cut off at `max_tokens` (`finish_reason == length`, or `<think>` without `</think>`; OmniDCBench also drops rows whose `prediction_json` is null) and backs the originals up to `~/trash`. Re-running the same `omni-bench run` command then retries only those items.
- `scripts/rescore_mcq.py results/<model>/<benchmark>/records.jsonl [options]` re-scores multiple-choice records from stored responses with a stricter letter extractor (explicit answer markers first, otherwise the last standalone A-D) and prints it next to the official parser's score.

## Adding a Strategy

Each axis is a registry in `src/omni_bench/inference/`. Subclass the axis base class, then register it by name to make it selectable from YAML:

```python
@TRANSPORTS.register("s3")
class S3Transport(Transport):
    def video_url(self, path): ...
```
