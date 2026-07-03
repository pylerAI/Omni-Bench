from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from omni_bench.adapters.base import BenchmarkAdapter, apply_limit
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.io import summarize_accuracy, write_json, write_jsonl


class WorldSenseAdapter(BenchmarkAdapter):
    name = "worldsense"

    def run(
        self,
        *,
        benchmark: BenchmarkConfig,
        model: ModelConfig,
        client: VllmChatClient,
        output_dir: Path,
    ) -> dict[str, Any]:
        data_root = Path(benchmark.data_path or ".").expanduser()
        annotation_file = Path(benchmark.annotation_file or data_root / "worldsense_qa.json").expanduser()
        video_dir = Path(benchmark.video_dir or data_root / "videos").expanduser()

        rows = apply_limit(self._flatten(read_worldsense_json(annotation_file), video_dir), benchmark.limit)
        records: list[dict[str, Any]] = []

        for row in tqdm(rows, desc=f"{model.name}/WorldSense"):
            prompt = build_prompt(row["question"], row["candidates"])
            video_path = row["video_path"]
            if not Path(video_path).exists():
                records.append(
                    {
                        **row,
                        "prompt": prompt,
                        "response": "",
                        "parsed_answer": "",
                        "is_correct": False,
                        "latency_s": None,
                        "error": f"Video file not found: {video_path}",
                    }
                )
                continue

            completion = client.complete(
                prompt,
                video_path=video_path,
                max_tokens=benchmark.max_tokens,
                temperature=benchmark.temperature,
            )
            response = completion.text
            parsed = parse_answer(response, valid_options=row["valid_options"])
            records.append(
                {
                    **row,
                    "prompt": prompt,
                    "response": response,
                    "parsed_answer": parsed,
                    "is_correct": parsed == row["answer"],
                    "latency_s": completion.latency_s,
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "total_tokens": completion.total_tokens,
                }
            )

        summary = summarize_accuracy(records, ("domain", "sub_category", "task_domain", "task_type", "duration"))
        summary["missing_videos"] = sum(1 for row in records if row.get("error"))
        summary["note"] = (
            "WorldSense official repo points to VLMEvalKit and does not include a standalone evaluator; "
            "this adapter evaluates the released multiple-choice annotation directly."
        )
        write_json(output_dir / "records.json", records)
        write_jsonl(output_dir / "records.jsonl", records)
        write_json(output_dir / "summary.json", summary)
        return summary

    @staticmethod
    def _flatten(data: dict[str, Any], video_dir: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for video_id, video in data.items():
            for key, task in video.items():
                if not key.startswith("task") or not isinstance(task, dict):
                    continue
                candidates = task.get("candidates", [])
                rows.append(
                    {
                        "video_id": video.get("video_id", video_id),
                        "video_duration": video.get("video_duration"),
                        "duration": video.get("duration"),
                        "domain": video.get("domain"),
                        "sub_category": video.get("sub_category"),
                        "audio_class": video.get("audio_class", []),
                        "video_caption": video.get("video_caption"),
                        "task_id": key,
                        "task_domain": task.get("task_domain"),
                        "task_type": task.get("task_type"),
                        "question": task.get("question"),
                        "answer": task.get("answer"),
                        "candidates": candidates,
                        "valid_options": candidate_letters(candidates),
                        "video_path": str(resolve_video_path(video_dir, str(video.get("video_id", video_id)))),
                    }
                )
        return rows


def read_worldsense_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"WorldSense annotation must be a JSON object: {path}")
    return data


def build_prompt(question: str, candidates: list[str]) -> str:
    return (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter of the correct option.\n"
        f"{question}\n"
        f"{chr(10).join(candidates)}\n"
        "The best answer is:"
    )


def parse_answer(response: str, valid_options: set[str]) -> str:
    match = re.search(r"\b([A-Z])\b", response.strip().upper())
    if match and match.group(1) in valid_options:
        return match.group(1)
    return ""


def candidate_letters(candidates: list[str]) -> set[str]:
    letters = set()
    for candidate in candidates:
        match = re.match(r"\s*([A-Z])\.", str(candidate))
        if match:
            letters.add(match.group(1))
    return letters


def resolve_video_path(video_dir: Path, video_id: str) -> Path:
    for suffix in (".mp4", ".mkv", ".webm", ".mov"):
        candidate = video_dir / f"{video_id}{suffix}"
        if candidate.exists():
            return candidate
    return video_dir / f"{video_id}.mp4"
