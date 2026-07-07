from __future__ import annotations

import base64
import json
import re
import string
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

from omni_bench.adapters.base import BenchmarkAdapter, apply_limit
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.io import append_jsonl, read_json, read_jsonl_records, summarize_accuracy, write_json


class OmniVideoBenchAdapter(BenchmarkAdapter):
    name = "omnivideobench"

    def run(
        self,
        *,
        benchmark: BenchmarkConfig,
        model: ModelConfig,
        client: VllmChatClient,
        output_dir: Path,
    ) -> dict[str, Any]:
        if not benchmark.annotation_file:
            raise ValueError("OmniVideoBench requires annotation_file in config.")
        video_dir = Path(benchmark.video_dir or benchmark.data_path or ".").expanduser()
        items = self._flatten(load_annotation(Path(benchmark.annotation_file)), video_dir)
        items = apply_limit(items, benchmark.limit)
        max_frames = int(benchmark.extra.get("max_frames", benchmark.extra.get("num_frames", 120)))
        fps = float(benchmark.extra.get("fps", 2.0))
        max_workers = int(benchmark.extra.get("max_workers", 2))
        preprocess_workers = int(benchmark.extra.get("preprocess_workers", 4))
        cache_dir = Path(
            benchmark.extra.get("preprocess_cache_dir", output_dir / "preprocess_cache")
        ).expanduser()

        records_path = output_dir / "records.jsonl"
        existing_records = read_jsonl_records(records_path)
        done = {str(record.get("question_id")) for record in existing_records}
        pending_items = [item for item in items if str(item.get("question_id")) not in done]

        cache_by_video = preprocess_videos(
            [Path(item["video_path"]) for item in pending_items],
            cache_dir=cache_dir,
            max_frames=max_frames,
            fps=fps,
            max_workers=preprocess_workers,
        )

        records: list[dict[str, Any]] = list(existing_records)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_item,
                    item=item,
                    benchmark=benchmark,
                    client=client,
                    cache_path=cache_by_video[Path(item["video_path"])],
                    prompt=self._prompt(item["question"], item["options"]),
                ): item
                for item in pending_items
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{model.name}/OmniVideoBench",
            ):
                record = future.result()
                records.append(record)
                append_jsonl(records_path, record)

        summary = summarize_accuracy(records, ("video_type", "question_type", "audio_type"))
        summary["max_workers"] = max_workers
        summary["preprocess_workers"] = preprocess_workers
        summary["preprocess_cache_dir"] = str(cache_dir)
        write_json(output_dir / "records.json", records)
        write_json(output_dir / "summary.json", summary)
        return summary

    @staticmethod
    def _flatten(data: list[dict[str, Any]], video_dir: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for video in data:
            if "question" in video and "options" in video:
                video_name = str(video.get("video") or video.get("video_id"))
                rows.append(
                    {
                        "video": video_name,
                        "video_type": video.get("video_type"),
                        "duration": video.get("duration"),
                        "question_id": video.get("question_id") or f"{video_name}-{len(rows)}",
                        "question": video.get("question"),
                        "question_type": video.get("question_type"),
                        "audio_type": video.get("audio_type"),
                        "options": video.get("options", []),
                        "answer": video.get("correct_option") or video.get("answer"),
                        "video_path": str(resolve_video_path(video_dir, video_name)),
                    }
                )
                continue

            video_name = str(video.get("video") or video.get("video_id"))
            video_path = resolve_video_path(video_dir, video_name)
            for idx, question in enumerate(video.get("questions", [])):
                rows.append(
                    {
                        "video": video_name,
                        "video_type": video.get("video_type"),
                        "duration": video.get("duration"),
                        "question_id": question.get("question_id", f"{video_name}-{idx}"),
                        "question": question.get("question"),
                        "question_type": question.get("question_type"),
                        "audio_type": question.get("audio_type"),
                        "options": question.get("options", []),
                        "answer": question.get("correct_option") or question.get("answer"),
                        "video_path": str(video_path),
                    }
                )
        return rows

    @staticmethod
    def _prompt(question: str, options: list[str]) -> str:
        return (
            "You are given a video. Based on the content of the video, answer the following question:\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{chr(10).join(options)}\n\n"
            "Answer with the option's letter directly(e.g., A, B, C, or D)."
            "If your access to the video content is limited, at least one option that is more likely than the others must be chosen."
            "Mustn't give any other reason for can not choose!"
        )


def resolve_video_path(video_dir: Path, video_name: str) -> Path:
    path = Path(video_name)
    if path.suffix:
        direct = video_dir / path
        if direct.exists():
            return direct
        return video_dir / path.name
    for suffix in (".mp4", ".mkv", ".webm", ".mov"):
        candidate = video_dir / f"{video_name}{suffix}"
        if candidate.exists():
            return candidate
    return video_dir / f"{video_name}.mp4"


def load_annotation(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    return read_json(path)


def run_single_item(
    *,
    item: dict[str, Any],
    benchmark: BenchmarkConfig,
    client: VllmChatClient,
    cache_path: Path,
    prompt: str,
) -> dict[str, Any]:
    sampled_video = read_json(cache_path)
    audio_path = sampled_video.get("audio_path")
    try:
        completion = client.complete(
            prompt,
            video_url=sampled_video["data_url"],
            audio_path=audio_path if audio_path else None,
            max_tokens=int(benchmark.extra.get("max_tokens", 1024)),
            temperature=float(benchmark.extra.get("temperature", 0.7)),
            top_p=benchmark.extra.get("top_p"),
            do_sample=bool(benchmark.extra.get("do_sample", True)),
            system_prompt=benchmark.extra.get(
                "system_prompt",
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            ),
            extra_body={
                "media_io_kwargs": {
                    "video": {
                        "num_frames": sampled_video["num_frames"],
                        "fps": sampled_video["sample_fps"],
                        "total_num_frames": sampled_video["total_frames"],
                        "frames_indices": sampled_video["frame_indices"],
                        "duration": sampled_video["duration_s"],
                    }
                },
                "mm_processor_kwargs": {"use_audio_in_video": False},
            },
        )
        response = completion.text
        parsed = extract_model_answer(response, prompt)
        return {
            **item,
            "prompt": prompt,
            "response": response,
            "parsed_answer": parsed,
            "is_correct": clean_text(parsed) == clean_text(item["answer"]),
            "latency_s": completion.latency_s,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.total_tokens,
            "sampled_num_frames": sampled_video["num_frames"],
            "audio_path": audio_path,
            "preprocess_cache_path": str(cache_path),
        }
    except Exception as exc:
        return {
            **item,
            "prompt": prompt,
            "response": "",
            "parsed_answer": "",
            "is_correct": False,
            "latency_s": None,
            "sampled_num_frames": sampled_video.get("num_frames"),
            "audio_path": audio_path,
            "preprocess_cache_path": str(cache_path),
            "error": repr(exc),
        }


def preprocess_videos(
    video_paths: list[Path],
    *,
    cache_dir: Path,
    max_frames: int,
    fps: float,
    max_workers: int,
) -> dict[Path, Path]:
    unique_paths = list(dict.fromkeys(video_paths))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = {path: sampled_video_cache_path(path, cache_dir, max_frames, fps) for path in unique_paths}
    missing = [path for path in unique_paths if not cache_paths[path].exists()]
    if missing:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    write_sampled_video_cache,
                    video_path=path,
                    cache_path=cache_paths[path],
                    max_frames=max_frames,
                    fps=fps,
                ): path
                for path in missing
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Preprocessing OmniVideoBench videos",
            ):
                future.result()
    return cache_paths


def sampled_video_cache_path(video_path: Path, cache_dir: Path, max_frames: int, fps: float) -> Path:
    fps_tag = str(fps).replace(".", "p")
    return cache_dir / f"{video_path.stem}_{fps_tag}fps_max{max_frames}_v3.json"


def write_sampled_video_cache(video_path: Path, cache_path: Path, max_frames: int, fps: float) -> None:
    data = sample_video_as_jpeg_sequence(video_path, max_frames=max_frames, fps=fps)
    audio_path = extract_audio_to_wav(video_path, cache_path.with_suffix(".wav"))
    data["audio_path"] = str(audio_path) if audio_path else None
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp_path.replace(cache_path)


def extract_audio_to_wav(video_path: Path, output_path: Path) -> Path | None:
    if output_path.exists():
        return output_path
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
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return output_path


def sample_video_as_jpeg_sequence(video_path: Path, max_frames: int, fps: float) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    duration_s = total_frames / source_fps if source_fps > 0 else 0.0
    if total_frames <= 0:
        capture.release()
        raise ValueError(f"Video has no frames: {video_path}")

    target_frames = int(duration_s * fps) if duration_s > 0 else max_frames
    sample_count = min(max(target_frames, 1), max_frames, total_frames)
    if sample_count == 1:
        indices = [0]
    else:
        indices = [
            round(i * (total_frames - 1) / (sample_count - 1))
            for i in range(sample_count)
        ]

    encoded_frames: list[str] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        with BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=90)
            encoded_frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    capture.release()

    if not encoded_frames:
        raise ValueError(f"Failed to sample frames from video: {video_path}")

    sample_fps = len(encoded_frames) / duration_s if duration_s > 0 else 1.0
    return {
        "data_url": "data:video/jpeg;base64," + ",".join(encoded_frames),
        "num_frames": len(encoded_frames),
        "frame_indices": indices[: len(encoded_frames)],
        "total_frames": total_frames,
        "sample_fps": sample_fps,
        "duration_s": duration_s,
        "requested_fps": fps,
        "source_fps": source_fps,
        "max_frames": max_frames,
    }


def extract_model_answer(response_text: str, prompt: str | None = None) -> str:
    """OmniVideoBench official-style answer extractor."""
    if "assistant" in response_text:
        model_answer = response_text.split("assistant")[-1].strip()
    elif prompt:
        model_answer = response_text.split(prompt)[-1].strip()
    else:
        model_answer = response_text.strip()

    box_match = re.search(r"/box\{([^}]+)\}", model_answer)
    if box_match:
        model_answer = box_match.group(1).strip()

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", model_answer)
    if boxed_match:
        model_answer = boxed_match.group(1).strip()

    return model_answer


def clean_text(text: object) -> str:
    """OmniVideoBench official clean_text comparison helper."""
    if not isinstance(text, str):
        return ""
    normalized = text.lower()

    box_match = re.search(r"/box\{([^}]+)\}", normalized)
    if box_match:
        normalized = box_match.group(1)

    if normalized and len(normalized) > 1 and normalized[1] == ".":
        normalized = normalized[0]

    translator = str.maketrans("", "", string.punctuation)
    return " ".join(normalized.translate(translator).split())
