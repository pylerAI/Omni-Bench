from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from omni_bench.adapters.base import BenchmarkAdapter, apply_limit
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.io import append_jsonl, read_jsonl_records, write_json, write_jsonl


PROMPT = (
    "Thoroughly describe everything in the video, capturing every detail. "
    "Include as much information from the audio as possible, and ensure that "
    "the descriptions of both audio and video are well-coordinated.\n\n"
    "Return only a valid JSON array. Do not include markdown fences or any extra text. "
    "Each array item must describe one temporal segment and include:\n"
    '- "timestamp": a string in "MM:SS-MM:SS" format, relative to the start of this clip.\n'
    '- "caption": a detailed audio-visual caption for that segment.\n'
    "Use enough segments to cover the full video from beginning to end."
)


def build_prompt(duration_sec: Any = None) -> str:
    """PROMPT with an explicit clip-length cap to prevent runaway timestamps."""
    try:
        total = int(round(float(duration_sec)))
    except (TypeError, ValueError):
        total = 0
    if total <= 0:
        return PROMPT
    cap = f"{total // 60:02d}:{total % 60:02d}"
    return (
        PROMPT
        + f"\nThis clip is exactly {total} seconds long. Every timestamp must lie within "
        f"00:00-{cap}, and no timestamp may exceed {cap}. Do not describe or invent any "
        "segment beyond the end of the clip; stop once the clip ends."
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class OmniDCBenchAdapter(BenchmarkAdapter):
    name = "omnidcbench"

    def run(
        self,
        *,
        benchmark: BenchmarkConfig,
        model: ModelConfig,
        client: VllmChatClient,
        output_dir: Path,
    ) -> dict[str, Any]:
        if not benchmark.annotation_file:
            raise ValueError("OmniDCBench requires annotation_file in config.")

        data_root = Path(benchmark.data_path or ".").expanduser()
        video_dir = Path(benchmark.video_dir or data_root / "Video").expanduser()
        rows = apply_limit(read_jsonl(Path(benchmark.annotation_file).expanduser()), benchmark.limit)

        concurrency = max(1, int(benchmark.extra.get("concurrency", 8)))

        prediction_path = output_dir / "predictions.jsonl"
        records = dedupe_records_by_clip(read_jsonl_records(prediction_path))
        done = {
            str(record.get("clip_path"))
            for record in records
            if is_completed_prediction(record)
        }
        pending_rows = [row for row in rows if str(row.get("clip_path")) not in done]

        def process_row(row: dict[str, Any]) -> dict[str, Any]:
            video_path = video_dir / row["clip_path"]
            if not video_path.exists():
                return {
                    **row,
                    "prediction": "FAILED",
                    "prediction_json": None,
                    "error": f"Video file not found: {video_path}",
                }

            prompt = build_prompt(row.get("duration"))
            try:
                completion = client.complete(
                    prompt,
                    video_path=video_path,
                    max_tokens=benchmark.max_tokens,
                    temperature=benchmark.temperature,
                    extra_body=omnidcbench_extra_body(use_audio_in_video=True),
                )
            except Exception as exc:
                if not should_retry_without_audio(exc):
                    return failed_record(row, exc)
                try:
                    completion = client.complete(
                        prompt,
                        video_path=video_path,
                        max_tokens=benchmark.max_tokens,
                        temperature=benchmark.temperature,
                        extra_body=omnidcbench_extra_body(use_audio_in_video=False),
                    )
                    fallback_reason = f"{type(exc).__name__}: {exc}"
                except Exception as fallback_exc:
                    return failed_record(
                        row,
                        fallback_exc,
                        fallback_reason=f"{type(exc).__name__}: {exc}",
                    )
            else:
                fallback_reason = None

            prediction = completion.text
            record = {
                **row,
                "prediction": prediction,
                "prediction_json": parse_prediction_json(prediction),
                "use_audio_in_video": fallback_reason is None,
                "latency_s": completion.latency_s,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.total_tokens,
            }
            if fallback_reason:
                record["audio_fallback_reason"] = fallback_reason
            return record

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_row, row) for row in pending_rows]
            progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{model.name}/OmniDCBench",
            )
            for future in progress:
                record = future.result()
                records.append(record)
                append_jsonl(prediction_path, record)
                done.add(str(record.get("clip_path")))

        records = dedupe_records_by_clip(records)
        write_jsonl(prediction_path, records)
        metric_result = run_official_metrics(
            benchmark=benchmark,
            data_root=data_root,
            output_dir=output_dir,
            prediction_path=prediction_path,
        )
        summary = {
            "total": len(records),
            "failed": sum(1 for row in records if row.get("prediction") == "FAILED" or row.get("error")),
            "prediction_file": str(output_dir / "predictions.jsonl"),
            "audio_fallbacks": sum(1 for row in records if row.get("use_audio_in_video") is False),
        }
        summary.update(metric_result)
        write_json(output_dir / "records.json", records)
        write_json(output_dir / "summary.json", summary)
        return summary


def omnidcbench_extra_body(*, use_audio_in_video: bool) -> dict[str, Any]:
    """Per-request body for OmniDCBench video requests.

    The vLLM server honours ``use_audio_in_video`` but ignores per-request video
    sampling kwargs (num_frames/fps/max_pixels — verified: varying them leaves the
    prompt token count unchanged), so only the audio flag is set here. Frame count
    and resolution fall back to the server's default video sampling, which stays
    within the context window because OmniDCBench clips are short (<=70s).
    """
    return {"mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video}}


def is_completed_prediction(record: dict[str, Any]) -> bool:
    return bool(
        record.get("clip_path")
        and str(record.get("prediction", "")).strip()
        and record.get("prediction") != "FAILED"
        and not record.get("error")
    )


def dedupe_records_by_clip(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_clip: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        clip = record.get("clip_path")
        if clip is None:
            continue
        key = str(clip)
        if key not in latest_by_clip:
            order.append(key)
        latest_by_clip[key] = record
    return [latest_by_clip[key] for key in order]


def should_retry_without_audio(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    audio_error_markers = (
        "audio",
        "no audio",
        "no audio found",
        "audio stream",
        "unsupported audio",
        "invalid or unsupported audio file",
        "use_audio_in_video",
    )
    return any(marker in message for marker in audio_error_markers)


def failed_record(
    row: dict[str, Any],
    exc: Exception,
    *,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    record = {
        **row,
        "prediction": "FAILED",
        "prediction_json": None,
        "error": f"{type(exc).__name__}: {exc}",
    }
    if fallback_reason:
        record["audio_fallback_reason"] = fallback_reason
    return record


def run_official_metrics(
    *,
    benchmark: BenchmarkConfig,
    data_root: Path,
    output_dir: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    if not bool(benchmark.extra.get("run_metrics", True)):
        return {
            "metrics": None,
            "metric_note": "Metric execution skipped because run_metrics=false.",
        }

    gt_file = Path(
        benchmark.extra.get("metric_gt_file") or data_root / "ours_gt_file_keypoints.json"
    ).expanduser()
    if not gt_file.exists():
        return {
            "metrics": None,
            "metric_error": f"Metric ground-truth file not found: {gt_file}",
        }

    evaluator_path = Path(
        benchmark.extra.get("metric_evaluator")
        or Path("submodules/TimeChat-Captioner/Eval/eval_time.py")
    )
    if not evaluator_path.is_absolute():
        evaluator_path = PROJECT_ROOT / evaluator_path
    if not evaluator_path.exists():
        return {
            "metrics": None,
            "metric_error": f"Metric evaluator not found: {evaluator_path}",
        }

    log_path = output_dir / "official_eval.log"
    raw_metric_path = output_dir / "official_metrics.json"
    max_workers = int(benchmark.extra.get("max_workers", 4))
    enable_sodam = bool(benchmark.extra.get("enable_sodam", False))
    cache_dir = output_dir / "metric_cache"
    metric_output_dir = output_dir / "metric_outputs"

    try:
        evaluator = load_python_module(evaluator_path, "omnidcbench_official_eval")
        configure_official_evaluator(
            evaluator,
            enable_sodam=enable_sodam,
            credentials=benchmark.extra.get("metric_credentials"),
        )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            raw_metrics = evaluator.eval_with_files(
                str(prediction_path),
                str(gt_file),
                tmponly=not enable_sodam,
                max_workers=max_workers,
                cache_dir=str(cache_dir),
                output_dir=str(metric_output_dir),
            )
        log_path.write_text(stream.getvalue(), encoding="utf-8")
        write_json(raw_metric_path, raw_metrics)
        return {
            "metrics": summarize_official_metrics(raw_metrics, enable_sodam=enable_sodam),
            "official_metric_file": str(raw_metric_path),
            "official_eval_log": str(log_path),
            "enable_sodam": enable_sodam,
        }
    except Exception as exc:
        log_path.write_text(
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        return {
            "metrics": None,
            "metric_error": f"{type(exc).__name__}: {exc}",
            "official_eval_log": str(log_path),
            "enable_sodam": enable_sodam,
        }


def load_python_module(path: Path, module_name: str) -> Any:
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_official_evaluator(
    evaluator: Any,
    *,
    enable_sodam: bool,
    credentials: str | None,
) -> None:
    original_scorer = evaluator.Checklist_Score
    if not enable_sodam:

        class DummyChecklistScore:
            DIM_KEYS = getattr(original_scorer, "DIM_KEYS", [])

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        evaluator.Checklist_Score = DummyChecklistScore
        return

    if not credentials:
        return

    class ConfiguredChecklistScore(original_scorer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["credentials"] = str(Path(credentials).expanduser())
            super().__init__(*args, **kwargs)

    evaluator.Checklist_Score = ConfiguredChecklistScore


def summarize_official_metrics(raw_metrics: dict[str, Any], *, enable_sodam: bool) -> dict[str, Any]:
    soda = raw_metrics.get("SODA_m")
    soda_m = (soda.get("SODA_m_total") if isinstance(soda, dict) else soda) if enable_sodam else None
    return {
        "f1": raw_metrics.get("F1_Score"),
        "miou": raw_metrics.get("mIoU"),
        "soda_m": soda_m,
        "precision_mean": raw_metrics.get("Precision_Mean"),
        "recall_mean": raw_metrics.get("Recall_Mean"),
        "num_videos": raw_metrics.get("num_videos"),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_prediction_json(text: str) -> Any:
    stripped = text.strip()
    for candidate in (stripped, extract_json_block(stripped)):
        if not candidate:
            continue
        try:
            return normalize_segments(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return None


def normalize_segments(value: Any) -> list[dict[str, Any]] | None:
    """Coerce a parsed prediction to a list of segment dicts, or None.

    The model sometimes returns a single segment object instead of an array; the
    official evaluator then iterates a dict/string and crashes. Wrap a lone dict
    in a list and drop any non-dict elements so the evaluator always sees a list
    of segment dicts.
    """
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, list):
        segments = [segment for segment in value if isinstance(segment, dict)]
        return segments or None
    return None


def extract_json_block(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return text[start : end + 1]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return None
