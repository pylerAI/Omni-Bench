from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from omni_bench.adapters.base import BenchmarkAdapter, apply_limit
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.io import write_json, write_jsonl


PROMPT = (
    "Thoroughly describe everything in the video, capturing every detail. "
    "Include as much information from the audio as possible, and ensure that "
    "the descriptions of both audio and video are well-coordinated."
)


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

        max_frames = int(benchmark.extra.get("max_frames", 160))
        fps = float(benchmark.extra.get("fps", 2.0))
        max_pixels = int(benchmark.extra.get("max_pixels", 297920))

        records: list[dict[str, Any]] = []
        for row in tqdm(rows, desc=f"{model.name}/OmniDCBench"):
            video_path = video_dir / row["clip_path"]
            if not video_path.exists():
                records.append(
                    {
                        **row,
                        "prediction": "FAILED",
                        "prediction_json": None,
                        "error": f"Video file not found: {video_path}",
                    }
                )
                continue

            completion = client.complete(
                PROMPT,
                video_path=video_path,
                max_tokens=benchmark.max_tokens,
                temperature=benchmark.temperature,
                extra_body={
                    "media_io_kwargs": {
                        "video": {
                            "num_frames": max_frames,
                            "fps": fps,
                        },
                        "image": {
                            "max_pixels": max_pixels,
                        },
                    },
                    "mm_processor_kwargs": {"use_audio_in_video": True},
                },
            )
            prediction = completion.text
            records.append(
                {
                    **row,
                    "prediction": prediction,
                    "prediction_json": parse_prediction_json(prediction),
                    "latency_s": completion.latency_s,
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "total_tokens": completion.total_tokens,
                }
            )

        summary = {
            "total": len(records),
            "failed": sum(1 for row in records if row.get("prediction") == "FAILED" or row.get("error")),
            "prediction_file": str(output_dir / "predictions.jsonl"),
            "note": (
                "OmniDCBench official metrics are implemented in TimeChat-Captioner/Eval. "
                "Use predictions.jsonl with the official eval scripts for SODA_M/F1."
            ),
        }
        write_jsonl(output_dir / "predictions.jsonl", records)
        write_json(output_dir / "records.json", records)
        write_json(output_dir / "summary.json", summary)
        return summary


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
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
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
