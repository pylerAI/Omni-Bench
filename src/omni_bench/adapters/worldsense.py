from __future__ import annotations

import base64
import math
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
from PIL import Image
from tqdm import tqdm

from omni_bench.adapters.base import BenchmarkAdapter, apply_limit
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.io import (
    append_jsonl,
    read_jsonl_records,
    summarize_accuracy,
    summarize_throughput,
    write_json,
)


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
        cache_dir = Path(benchmark.extra.get("preprocess_cache_dir", data_root / "preprocess_cache")).expanduser()

        rows = apply_limit(self._flatten(read_worldsense_json(annotation_file), video_dir), benchmark.limit)
        records_path = output_dir / "records.jsonl"
        records = read_jsonl_records(records_path)
        done = {worldsense_key(record) for record in records}
        num_frames = int(benchmark.extra.get("num_frames", 8))
        concurrency = max(1, int(benchmark.extra.get("concurrency", 8)))
        pending = [row for row in rows if worldsense_key(row) not in done]

        def process_row(row: dict[str, Any]) -> dict[str, Any]:
            prompt = build_prompt(row["question"], row["candidates"])
            video_path = row["video_path"]
            if not Path(video_path).exists():
                return {
                    **row,
                    "prompt": prompt,
                    "response": "",
                    "parsed_answer": "",
                    "is_correct": False,
                    "latency_s": None,
                    "error": f"Video file not found: {video_path}",
                }
            # Frame sampling + audio extraction is CPU/IO-bound; running rows
            # concurrently overlaps it with GPU inference and uses all DP replicas.
            media = prepare_worldsense_media(Path(video_path), cache_dir, num_frames)
            completion = client.complete(
                prompt,
                image_urls=media["image_urls"],
                audio_path=media["audio_path"],
                max_tokens=benchmark.max_tokens,
                temperature=benchmark.temperature,
                system_prompt=SYS,
            )
            response = completion.text
            parsed = extract_characters_regex(response)
            return {
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
                "preprocess_cache_path": str(media["cache_path"]),
            }

        write_lock = threading.Lock()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_row, row) for row in pending]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{model.name}/WorldSense"):
                record = future.result()
                with write_lock:
                    records.append(record)
                    append_jsonl(records_path, record)
        wall_time_s = time.perf_counter() - started

        summary = summarize_accuracy(records, ("domain", "sub_category", "task_domain", "task_type", "duration"))
        summary["throughput"] = summarize_throughput(records, wall_time_s=wall_time_s if pending else None)
        summary["missing_videos"] = sum(1 for row in records if row.get("error"))
        summary["vlmeval_rating_file"] = str(output_dir / "vlmeval_rating.json")
        summary["note"] = (
            "WorldSense evaluation mirrors VLMEvalKit exact matching and dimension aggregation. "
            "Judge-based fallback extraction is not used."
        )
        write_json(output_dir / "records.json", records)
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


def prepare_worldsense_media(video_path: Path, cache_dir: Path, num_frames: int) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{video_path.stem}_{num_frames}frames_v1.json"
    audio_path = cache_dir / f"{video_path.stem}.wav"
    if not cache_path.exists():
        data = sample_video_frames_as_images(video_path, num_frames)
        tmp_path = cache_path.with_suffix(".tmp")
        import json

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp_path.replace(cache_path)
    if not audio_path.exists():
        extract_audio_to_wav(video_path, audio_path)

    import json

    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["audio_path"] = str(audio_path)
    data["cache_path"] = str(cache_path)
    return data


def sample_video_frames_as_images(video_path: Path, num_frames: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise ValueError(f"Video has no frames: {video_path}")
    sample_count = min(num_frames, total_frames)
    indices = [0] if sample_count == 1 else [
        round(i * (total_frames - 1) / (sample_count - 1))
        for i in range(sample_count)
    ]

    image_urls = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        with BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=90)
            image_urls.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    capture.release()
    if not image_urls:
        raise ValueError(f"Failed to sample frames from video: {video_path}")
    return {
        "image_urls": image_urls,
        "num_frames": len(image_urls),
        "frame_indices": indices[: len(image_urls)],
        "total_frames": total_frames,
    }


def extract_audio_to_wav(video_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(command, check=True)


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


def worldsense_key(row: dict[str, Any]) -> str:
    return f"{row.get('video_id')}::{row.get('task_id')}"
