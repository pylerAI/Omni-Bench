from __future__ import annotations

import base64
import re
import string
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
from omni_bench.io import read_json, summarize_accuracy, write_json, write_jsonl


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
        records: list[dict[str, Any]] = []
        num_frames = int(benchmark.extra.get("num_frames", 120))

        for item in tqdm(items, desc=f"{model.name}/OmniVideoBench"):
            prompt = self._prompt(item["question"], item["options"])
            sampled_video = sample_video_as_jpeg_sequence(Path(item["video_path"]), num_frames)
            completion = client.complete(
                prompt,
                video_url=sampled_video["data_url"],
                audio_path=item["video_path"],
                max_tokens=benchmark.max_tokens,
                temperature=benchmark.temperature,
                system_prompt=benchmark.extra.get("system_prompt"),
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
            records.append(
                {
                    **item,
                    "prompt": prompt,
                    "response": response,
                    "parsed_answer": parsed,
                    "is_correct": clean_text(parsed) == clean_text(item["answer"]),
                    "latency_s": completion.latency_s,
                    "sampled_num_frames": sampled_video["num_frames"],
                }
            )

        summary = summarize_accuracy(records, ("video_type", "question_type", "audio_type"))
        write_json(output_dir / "records.json", records)
        write_jsonl(output_dir / "records.jsonl", records)
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
            "Answer with the option's letter directly (e.g., A, B, C, or D)."
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


def sample_video_as_jpeg_sequence(video_path: Path, num_frames: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    duration_s = total_frames / fps if fps > 0 else 0.0
    if total_frames <= 0:
        capture.release()
        raise ValueError(f"Video has no frames: {video_path}")

    sample_count = min(num_frames, total_frames)
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
