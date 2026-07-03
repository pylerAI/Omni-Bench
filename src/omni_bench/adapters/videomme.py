from __future__ import annotations

from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from omni_bench.adapters.base import BenchmarkAdapter
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.io import read_json, write_json, write_jsonl


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
        official = deepcopy(read_json(benchmark.annotation_file))
        flat = self._flatten(official, video_dir)
        if benchmark.limit is not None:
            flat = flat[: benchmark.limit]

        records: list[dict[str, Any]] = []
        for item in tqdm(flat, desc=f"{model.name}/Video-MME"):
            prompt = self._prompt(item, use_subtitles=bool(benchmark.extra.get("use_subtitles", False)))
            completion = client.complete(
                prompt,
                video_path=item["video_path"],
                max_tokens=benchmark.max_tokens,
                temperature=benchmark.temperature,
            )
            response = completion.text
            item["question_ref"]["response"] = response
            records.append(
                {
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
            )

        throughput_summary = summarize_throughput(records)
        write_json(output_dir / "official_results.json", official)
        write_json(output_dir / "records.json", records)
        write_jsonl(output_dir / "records.jsonl", records)
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
