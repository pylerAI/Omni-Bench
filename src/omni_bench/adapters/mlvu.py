from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from omni_bench.adapters.base import BenchmarkAdapter, apply_limit
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.io import read_json, summarize_accuracy, write_json, write_jsonl


DEFAULT_DATA_LIST = {
    "count": ("4_count.json", "video/count"),
    "ego": ("3_ego.json", "video/ego"),
    "needle": ("2_needle.json", "video/needle"),
    "order": ("5_order.json", "video/order"),
    "plotQA": ("1_plotQA.json", "video/plotQA"),
    "anomaly_reco": ("6_anomaly_reco.json", "video/anomaly_reco"),
    "topic_reasoning": ("7_topic_reasoning.json", "video/topic_reasoning"),
}


class MLVUAdapter(BenchmarkAdapter):
    name = "mlvu"

    def run(
        self,
        *,
        benchmark: BenchmarkConfig,
        model: ModelConfig,
        client: VllmChatClient,
        output_dir: Path,
    ) -> dict[str, Any]:
        data_dir = Path(benchmark.annotation_file or benchmark.extra.get("data_dir") or ".").expanduser()
        video_root = Path(benchmark.video_dir or benchmark.data_path or ".").expanduser()
        data_list = benchmark.extra.get("data_list", DEFAULT_DATA_LIST)
        rows = self._load_rows(data_dir, video_root, data_list)
        rows = apply_limit(rows, benchmark.limit)

        records: list[dict[str, Any]] = []
        for row in tqdm(rows, desc=f"{model.name}/MLVU"):
            completion = client.complete(
                row["prompt"],
                video_path=row["video_path"],
                max_tokens=benchmark.max_tokens,
                temperature=benchmark.temperature,
                system_prompt=benchmark.extra.get(
                    "system_prompt",
                    "Carefully watch this video and pay attention to every detail. "
                    "Based on your observations, select the best option that accurately addresses the question.",
                ),
            )
            response = completion.text
            records.append(
                {
                    **row,
                    "response": response,
                    "is_correct": check_ans(pred=response, gt=row["answer"]),
                    "latency_s": completion.latency_s,
                }
            )

        summary = summarize_accuracy(records, ("task_type",))
        write_json(output_dir / "records.json", records)
        write_jsonl(output_dir / "records.jsonl", records)
        write_json(output_dir / "summary.json", summary)
        return summary

    @staticmethod
    def _load_rows(data_dir: Path, video_root: Path, data_list: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for task_type, value in data_list.items():
            annotation_name, video_prefix = value[0], value[1]
            for item in read_json(data_dir / annotation_name):
                candidates = item["candidates"]
                answer_idx = candidates.index(item["answer"])
                options = [f"({chr(ord('A') + idx)}) {candidate}" for idx, candidate in enumerate(candidates)]
                question = f"Question: {item['question']}\nOptions:\n" + "\n".join(options)
                answer_letter = chr(ord("A") + answer_idx)
                rows.append(
                    {
                        "task_type": task_type,
                        "video_path": str(video_root / video_prefix / item["video"]),
                        "question": item["question"],
                        "prompt": question + "\nOnly give the best option.",
                        "answer": f"({answer_letter}) {item['answer']}",
                    }
                )
        return rows


def check_ans(pred: str, gt: str) -> bool:
    """MLVU official multiple-choice answer checker."""
    gt_option = gt[gt.index("(") + 1 : gt.index(")")]
    if ")" in pred:
        pred = pred[pred.index(")") - 1 : pred.index(")")]
    return pred == gt_option
