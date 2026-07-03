from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from omni_bench.adapters.base import BenchmarkAdapter, apply_limit
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.io import summarize_accuracy, write_json, write_jsonl


SYS = (
    "Carefully watch this video and pay attention to every detail. "
    "Based on your observations, select the best option that accurately addresses the question."
)

FRAMES_TMPL_AUDIO = (
    "These are the frames of a video and the corresponding audio. "
    "Select the best answer to the following multiple-choice question based on the video. "
    "Respond with only the letter (A, B, C, or D) of the correct option."
)

DURATIONS = ["<1min", "1-2min", "2-4min", "4-6min", "6-8min", ">8min"]
DOMAINS = [
    "Tech & Science",
    "Culture & Politics",
    "Daily Life",
    "Film & TV",
    "Performance",
    "Games",
    "Sports",
    "Music",
]
TASK_DOMAINS = ["Recognition", "Understanding", "Reasoning"]
TASK_CATEGORIES = [
    "Anomaly Recognition",
    "Event Recognition",
    "Attribute Recognition",
    "Human Interaction",
    "Temporal Localization",
    "Video Emotions",
    "Event Sorting",
    "Hallucination",
    "Text and Diagram Understanding",
    "Attribute Reasoning",
    "Causal Reasoning",
    "Object Counting",
    "Action Counting",
    "Temporal Prediction",
    "Emotion Change",
    "Audio Counting",
    "Scene Recognition",
    "Human-object Interaction",
    "Human Emotions",
    "Object State Change",
    "Relation Reasoning",
    "Spatial Relation",
    "Audio Source Localization",
    "Audio Recognition",
    "Object Existence Recognition",
    "Audio Change",
]
AUDIO_CLASSES = ["Speech", "Event", "Music"]


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
        num_frames = int(benchmark.extra.get("num_frames", 8))

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
                system_prompt=SYS,
                extra_body={
                    "media_io_kwargs": {"video": {"num_frames": num_frames}},
                    "mm_processor_kwargs": {"use_audio_in_video": True},
                },
            )
            response = completion.text
            parsed = extract_characters_regex(response)
            records.append(
                {
                    **row,
                    "prompt": prompt,
                    "response": response,
                    "parsed_answer": parsed,
                    "is_correct": parsed == row["answer"],
                    "score": int(parsed == row["answer"]) if parsed else -1,
                    "latency_s": completion.latency_s,
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "total_tokens": completion.total_tokens,
                }
            )

        summary = summarize_accuracy(records, ("domain", "sub_category", "task_domain", "task_type", "duration"))
        summary["missing_videos"] = sum(1 for row in records if row.get("error"))
        summary["vlmeval_rating_file"] = str(output_dir / "vlmeval_rating.json")
        summary["note"] = (
            "WorldSense evaluation mirrors VLMEvalKit exact matching and dimension aggregation. "
            "Judge-based fallback extraction is not used."
        )
        write_json(output_dir / "records.json", records)
        write_jsonl(output_dir / "records.jsonl", records)
        write_json(output_dir / "vlmeval_rating.json", get_dimension_rating(records))
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
        f"{FRAMES_TMPL_AUDIO}\n"
        f"Question: {question}\n"
        f"{chr(10).join(candidates)}\n"
        "Answer: "
    )


def extract_characters_regex(response: str) -> str:
    normalized = response.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
    ]
    for answer_prefix in answer_prefixes:
        normalized = normalized.replace(answer_prefix, "")

    if len(normalized.split()) > 10 and not re.search("[ABCD]", normalized):
        return ""
    match = re.search(r"[ABCD]", normalized)
    return "" if match is None else match.group(0)


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


def get_dimension_rating(records: list[dict[str, Any]]) -> dict[str, Any]:
    sub_categories = sorted({str(row.get("sub_category")) for row in records if row.get("sub_category")})
    rating: dict[str, Any] = {}
    for duration in ["overall"] + DURATIONS:
        rating[duration] = {
            "overall": "",
            "domain": {key: [] for key in DOMAINS},
            "sub_category": {key: [] for key in sub_categories},
            "task_domain": {key: [] for key in TASK_DOMAINS},
            "task_type": {key: [] for key in TASK_CATEGORIES},
            "audio_class": {key: [] for key in AUDIO_CLASSES},
        }

    for row in records:
        score = float(row.get("score", -1))
        duration = row.get("duration")
        targets = ["overall"]
        if duration in DURATIONS:
            targets.append(duration)
        for target in targets:
            append_score(rating[target]["domain"], row.get("domain"), score)
            append_score(rating[target]["sub_category"], row.get("sub_category"), score)
            append_score(rating[target]["task_domain"], row.get("task_domain"), score)
            append_score(rating[target]["task_type"], row.get("task_type"), score)
            for audio_class in row.get("audio_class", []) or []:
                append_score(rating[target]["audio_class"], audio_class, score)

    for duration in ["overall"] + DURATIONS:
        all_domain_scores = []
        for scores in rating[duration]["domain"].values():
            all_domain_scores.extend(scores)
        rating[duration]["overall"] = format_mean(all_domain_scores)
        for group in ("domain", "sub_category", "task_domain", "task_type", "audio_class"):
            for key, scores in rating[duration][group].items():
                rating[duration][group][key] = format_mean(scores)
    return rating


def append_score(bucket: dict[str, list[float]], key: object, score: float) -> None:
    if key in bucket:
        bucket[str(key)].append(score)


def format_mean(scores: list[float]) -> str:
    valid = [score for score in scores if score >= 0]
    if not valid:
        return "nan"
    mean = sum(valid) / len(valid)
    if math.isnan(mean):
        return "nan"
    return f"{mean:.3f}"
