from __future__ import annotations

import ast
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from omni_bench.adapters.base import BenchmarkAdapter, apply_limit
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.io import append_jsonl, load_existing_keys, read_jsonl_records, summarize_accuracy, write_json


class AVSpeakerBenchAdapter(BenchmarkAdapter):
    name = "av_speakerbench"

    def run(
        self,
        *,
        benchmark: BenchmarkConfig,
        model: ModelConfig,
        client: VllmChatClient,
        output_dir: Path,
    ) -> dict[str, Any]:
        dataset_name = benchmark.extra.get("dataset_name", "plnguyen2908/Holistic_AVQA_bench")
        rows = list(load_dataset(dataset_name, split=benchmark.split))
        rows = self._filter(rows, benchmark)
        rows = apply_limit(rows, benchmark.limit)

        data_root = Path(benchmark.data_path or ".").expanduser()
        records_path = output_dir / "records.jsonl"
        records = read_jsonl_records(records_path)
        done = load_existing_keys(records_path, "question_id")
        pending = [row for row in rows if str(row.get("question_id")) not in done]
        concurrency = max(1, int(benchmark.extra.get("concurrency", 8)))

        def process_row(row: dict[str, Any]) -> dict[str, Any]:
            choices = ast.literal_eval(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
            prompt = (
                "Select the best answer to the following multiple-choice question based on the video. "
                "Respond with only the letter (A, B, C, or D) of the correct option.\n"
                f"{row['question']}\n"
                f"{chr(10).join(choices)}\n"
                "The best answer is:"
            )
            media_path = self._media_path(data_root, row, benchmark.mode)
            completion = client.complete(
                prompt,
                video_path=media_path if benchmark.mode != "audio" else None,
                audio_path=media_path if benchmark.mode == "audio" else None,
                max_tokens=benchmark.max_tokens,
                temperature=benchmark.temperature,
                extra_body=extra_body_for_mode(benchmark.mode),
            )
            response = completion.text
            parsed = extract_characters_regex(response)
            return {
                "question_id": row.get("question_id"),
                "video_id": row.get("video_id"),
                "category": row.get("category"),
                "sub_category": row.get("sub_category"),
                "task_id": row.get("task_id"),
                "prompt": prompt,
                "answer": row.get("answer"),
                "response": response,
                "parsed_answer": parsed,
                "is_correct": parsed == row.get("answer"),
                "latency_s": completion.latency_s,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.total_tokens,
                "media_path": str(media_path),
            }

        write_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_row, row) for row in pending]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{model.name}/AV-SpeakerBench"):
                record = future.result()
                with write_lock:
                    records.append(record)
                    append_jsonl(records_path, record)

        summary = summarize_accuracy(records, ("category", "sub_category", "task_id"))
        write_json(output_dir / "records.json", records)
        write_json(output_dir / "summary.json", summary)
        return summary

    @staticmethod
    def _filter(rows: list[dict[str, Any]], benchmark: BenchmarkConfig) -> list[dict[str, Any]]:
        category = benchmark.extra.get("category")
        sub_category = benchmark.extra.get("sub_category")
        task_id = benchmark.extra.get("task_id")
        return [
            row
            for row in rows
            if (category is None or row.get("category") == category)
            and (sub_category is None or row.get("sub_category") == sub_category)
            and (task_id is None or row.get("task_id") == task_id)
        ]

    @staticmethod
    def _media_path(data_root: Path, row: dict[str, Any], mode: str) -> Path:
        if mode == "audio":
            key = "audio_path"
        elif mode == "visual":
            key = "visual_path"
        else:
            key = "audio_visual_path"
        return data_root / row[key]


def extra_body_for_mode(mode: str) -> dict[str, Any] | None:
    if mode == "av":
        return {"mm_processor_kwargs": {"use_audio_in_video": True}}
    return None


ANSWER_PREFIXES = [
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
    "The correct answer",
    "The correct option",
    "Based",
    "Correct answer",
    "\u261e",
    "<|im_end|>",
]


def extract_characters_regex(text: str | None) -> str:
    """AV-SpeakerBench official answer parser."""
    if text is None:
        return ""

    normalized = text.strip()
    for answer_prefix in ANSWER_PREFIXES:
        normalized = normalized.replace(answer_prefix, "")

    normalized = re.sub(r"[.,:!'\";/\?`~@#\$%\^&\*\(\)\[\]\{\}\\|<>\n]", " ", normalized)
    for token in normalized.split():
        if token in {"A", "B", "C", "D", "E"}:
            return token
    return ""
