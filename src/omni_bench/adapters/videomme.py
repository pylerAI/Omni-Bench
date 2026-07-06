from __future__ import annotations

import base64
import math
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from omni_bench.adapters.base import BenchmarkAdapter
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.io import append_jsonl, read_json, read_jsonl_records, write_json


class VideoMMEAdapter(BenchmarkAdapter):
    name = "videomme"

    def run(
        self,
        *,
        benchmark: BenchmarkConfig,
        model: ModelConfig,
        client: VllmChatClient,
        output_dir: Path,
    ) -> dict[str, Any]:
        if not benchmark.annotation_file:
            raise ValueError("Video-MME requires annotation_file in config.")
        video_dir = Path(benchmark.video_dir or benchmark.data_path or ".").expanduser()
        official = deepcopy(load_videomme_annotation(benchmark.annotation_file))
        flat = self._flatten(official, video_dir)
        if benchmark.limit is not None:
            flat = flat[: benchmark.limit]

        num_frames = int(benchmark.extra.get("max_frames", 64))
        max_pixels = int(benchmark.extra.get("max_pixels", 768 * 28 * 28))

        records_path = output_dir / "records.jsonl"
        records = read_jsonl_records(records_path)
        done = {str(record.get("question_id")) for record in records}
        existing_response = {str(record.get("question_id")): record.get("response") for record in records}
        for item in flat:
            qid = str(item["question_id"])
            if qid in existing_response:
                item["question_ref"]["response"] = existing_response[qid]
        frame_cache: dict[str, list[str]] = {}
        for item in tqdm(flat, desc=f"{model.name}/Video-MME"):
            if str(item["question_id"]) in done:
                continue
            video_path = item["video_path"]
            # The vLLM server ignores per-request video sampling kwargs, so sample
            # frames client-side (uniform, resized to a per-frame pixel budget) and
            # send them as images — the bounded, deterministic equivalent of
            # VLMEvalKit's frame-based Video-MME setting. Frames are cached per video.
            if video_path not in frame_cache:
                frame_cache = {video_path: sample_video_frames(video_path, num_frames, max_pixels)}
            prompt = self._prompt(item, use_subtitles=bool(benchmark.extra.get("use_subtitles", False)))
            completion = client.complete(
                prompt,
                image_urls=frame_cache[video_path],
                max_tokens=benchmark.max_tokens,
                temperature=benchmark.temperature,
            )
            response = completion.text
            item["question_ref"]["response"] = response
            record = {
                "video_id": item["video_id"],
                "duration": item["duration"],
                "domain": item["domain"],
                "sub_category": item["sub_category"],
                "question_id": item["question_id"],
                "task_type": item["task_type"],
                "answer": item["answer"],
                "response": response,
                "latency_s": completion.latency_s,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.total_tokens,
                "video_path": item["video_path"],
            }
            records.append(record)
            append_jsonl(records_path, record)
            done.add(str(item["question_id"]))

        throughput_summary = summarize_throughput(records)
        write_json(output_dir / "official_results.json", official)
        write_json(output_dir / "records.json", records)
        write_json(output_dir / "throughput_summary.json", throughput_summary)
        summary = {
            "total": len(records),
            "official_results_file": str(output_dir / "official_results.json"),
            "throughput_summary_file": str(output_dir / "throughput_summary.json"),
            "throughput": throughput_summary,
            "note": "Run the official Video-MME evaluator on official_results.json for accuracy.",
        }
        write_json(output_dir / "summary.json", summary)
        return summary

    @staticmethod
    def _flatten(data: list[dict[str, Any]], video_dir: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for video in data:
            video_id = str(video.get("video_id"))
            video_path = resolve_video_path(video_dir, str(video.get("video") or video_id))
            for question in video.get("questions", []):
                rows.append(
                    {
                        "video_id": video_id,
                        "duration": video.get("duration"),
                        "domain": video.get("domain"),
                        "sub_category": video.get("sub_category"),
                        "subtitles": video.get("subtitles") or video.get("subtitle") or "",
                        "question_id": question.get("question_id"),
                        "task_type": question.get("task_type"),
                        "question": question.get("question"),
                        "options": question.get("options", []),
                        "answer": question.get("answer"),
                        "question_ref": question,
                        "video_path": str(video_path),
                    }
                )
        return rows

    @staticmethod
    def _prompt(item: dict[str, Any], *, use_subtitles: bool) -> str:
        prefix = ""
        if use_subtitles and item.get("subtitles"):
            prefix = f"This video's subtitles are listed below:\n{item['subtitles']}\n"
        return (
            prefix
            + "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            f"{item['question']}\n"
            f"{chr(10).join(item['options'])}\n"
            "The best answer is:"
        )


def sample_video_frames(video_path: str | Path, num_frames: int, max_pixels: int) -> list[str]:
    """Uniformly sample and resize frames, returned as JPEG ``data:`` URIs.

    Frames are resized so each has at most ``max_pixels`` pixels (rounded to the
    28-pixel patch grid Qwen uses), bounding the token count per frame to
    ``max_pixels / (28*28) / 4``. This keeps the request within the context
    window regardless of clip length, which sending the raw video does not.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise ValueError(f"Video has no frames: {video_path}")
    sample_count = min(num_frames, total_frames)
    indices = [0] if sample_count == 1 else [
        round(i * (total_frames - 1) / (sample_count - 1)) for i in range(sample_count)
    ]

    image_urls: list[str] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        frame = _resize_to_pixel_budget(frame, max_pixels)
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            image_urls.append("data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii"))
    capture.release()
    if not image_urls:
        raise ValueError(f"Failed to sample frames from video: {video_path}")
    return image_urls


def _resize_to_pixel_budget(frame: Any, max_pixels: int, factor: int = 28) -> Any:
    height, width = frame.shape[:2]
    scale = min(1.0, math.sqrt(max_pixels / float(height * width)))
    new_h = max(factor, int(round(height * scale / factor)) * factor)
    new_w = max(factor, int(round(width * scale / factor)) * factor)
    if (new_h, new_w) != (height, width):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def load_videomme_annotation(path: str | Path) -> list[dict[str, Any]]:
    """Return the official nested (video -> questions) structure.

    Accepts the official template JSON (already nested) or the dataset's flat,
    per-question form (HuggingFace parquet or a flat JSON list), grouping the
    latter by ``video_id`` so the adapter and ``official_results.json`` keep the
    Video-MME evaluator's expected shape.
    """
    src = Path(path).expanduser()
    if src.suffix == ".parquet":
        import pandas as pd

        return _group_questions(pd.read_parquet(src).to_dict("records"))
    data = read_json(src)
    if isinstance(data, list) and data and isinstance(data[0], dict) and "questions" in data[0]:
        return data
    return _group_questions(data)


def _group_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    videos: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        video_id = str(row.get("video_id"))
        if video_id not in videos:
            videos[video_id] = {
                "video_id": video_id,
                "duration": row.get("duration") or row.get("question_type"),
                "domain": row.get("domain"),
                "sub_category": row.get("sub_category"),
                "url": row.get("url"),
                "video": row.get("videoID") or row.get("video") or video_id,
                "questions": [],
            }
            order.append(video_id)
        options = row.get("options")
        if options is None:
            options = row.get("candidates")
        options = list(options) if options is not None else []
        questions = videos[video_id]["questions"]
        questions.append(
            {
                "question_id": row.get("question_id") or f"{video_id}-{len(questions) + 1}",
                "task_type": row.get("task_type"),
                "question": row.get("question"),
                "options": [str(option) for option in options],
                "answer": row.get("answer"),
                "response": "",
            }
        )
    return [videos[video_id] for video_id in order]


def resolve_video_path(video_dir: Path, video_name: str) -> Path:
    path = Path(video_name)
    if path.suffix:
        return video_dir / path
    for suffix in (".mp4", ".mkv", ".webm", ".mov"):
        candidate = video_dir / f"{video_name}{suffix}"
        if candidate.exists():
            return candidate
    return video_dir / f"{video_name}.mp4"


def summarize_throughput(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_s"]) for row in records if row.get("latency_s") is not None]
    total_wall_time_s = sum(latencies)
    unique_videos = {row.get("video_id") for row in records if row.get("video_id") is not None}

    prompt_tokens = sum_optional_int(row.get("prompt_tokens") for row in records)
    completion_tokens = sum_optional_int(row.get("completion_tokens") for row in records)
    total_tokens = sum_optional_int(row.get("total_tokens") for row in records)

    return {
        "samples": len(records),
        "unique_videos": len(unique_videos),
        "total_wall_time_s": round(total_wall_time_s, 6),
        "avg_latency_s": round(sum(latencies) / len(latencies), 6) if latencies else 0.0,
        "p50_latency_s": percentile(latencies, 50),
        "p95_latency_s": percentile(latencies, 95),
        "samples_per_sec": safe_rate(len(records), total_wall_time_s),
        "videos_per_hour": safe_rate(len(unique_videos), total_wall_time_s / 3600.0),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_per_sec": safe_rate(prompt_tokens, total_wall_time_s),
        "completion_tokens_per_sec": safe_rate(completion_tokens, total_wall_time_s),
        "total_tokens_per_sec": safe_rate(total_tokens, total_wall_time_s),
    }


def sum_optional_int(values: Any) -> int:
    return sum(int(value) for value in values if value is not None)


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, ceil((pct / 100.0) * len(ordered)) - 1))
    return round(ordered[index], 6)
