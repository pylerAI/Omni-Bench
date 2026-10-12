"""Reasoning strategies (as_is / split) and `omni-bench rescore` (no server)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from omni_bench.client import build_chat_client
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.inference import REASONING_STRATEGIES, resolve_inference
from omni_bench.rescore import rescore_run


def model(**inference):
    return ModelConfig(name="m", weight_path="/x", base_url="http://x/v1", extra={"inference": inference})


def test_split_rules():
    split = REASONING_STRATEGIES.get("split")()
    assert split.split("<think>\nstep A then B\n</think>\n\nB", None) == ("B", "step A then B")
    assert split.split("a </think> b </think> C", None) == ("C", "a </think> b")   # LAST </think>
    assert split.split("plain B", "srv") == ("plain B", "srv")                     # no tag -> unchanged
    assert split.split("x</think>D", "srv") == ("D", "srv\n\nx")                   # appended to server reasoning
    assert REASONING_STRATEGIES.get("as_is")().split("x</think>D", "srv") == ("x</think>D", "srv")
    for old in ("server", "think_tag", "none"):                                     # removed names
        with pytest.raises(ValueError):
            REASONING_STRATEGIES.get(old)


def test_client_applies_reasoning_before_the_adapter(openai_stub):
    openai_stub.content, openai_stub.reasoning = "think...\n</think>\n\nB", None
    res = build_chat_client(model(reasoning="split")).complete("Q")
    assert res.text == "B" and res.reasoning == "think..." and res.meta()["response_raw"] == openai_stub.content
    res = build_chat_client(model()).complete("Q")                                  # default as_is
    assert res.text == openai_stub.content and "response_raw" not in res.meta()
    openai_stub.content, openai_stub.reasoning = "B", "server-side"
    res = build_chat_client(model(reasoning="split")).complete("Q")
    assert res.text == "B" and res.reasoning == "server-side" and "response_raw" not in res.meta()
    res = build_chat_client(model()).complete("Q")
    assert res.text == "B" and res.reasoning == "server-side"                      # as_is keeps server reasoning
    with pytest.raises(ValueError):
        resolve_inference(ModelConfig(name="m", weight_path="/x", extra={"inference": {"reasoning": "splt"}}))
    with pytest.raises(ValueError):                                                 # model-only axis
        resolve_inference(model(), BenchmarkConfig(name="worldsense", extra={"inference": {"reasoning": "as_is"}}))


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture
def stored_run(tmp_path):
    run = tmp_path / "results" / "m"
    bench = run / "av_speakerbench"
    bench.mkdir(parents=True)
    base = {"category": "c", "sub_category": "s", "task_id": "t"}
    recs = [
        {**base, "question_id": 1, "answer": "C", "response": "Option B is wrong, so C.\n</think>\n\nC",
         "parsed_answer": "B", "is_correct": False, "reasoning": None},
        {**base, "question_id": 2, "answer": "A", "response": "A", "parsed_answer": "A", "is_correct": True,
         "reasoning": None},
        {**base, "question_id": 3, "answer": "D", "response": "", "parsed_answer": "", "is_correct": False,
         "error": "HTTP 500"},
    ]
    (bench / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (bench / "summary.json").write_text(json.dumps({"accuracy": 33.3333, "perf": {"wall_s": 1.0}}))
    (bench / "config_used.json").write_text(json.dumps({
        "inference": {"reasoning": "as_is", "frames": "server"},
        "model": {"name": "m", "weight_path": "/x", "extra": {}},
        "benchmarks": [{"name": "av_speakerbench", "extra": {}}]}))
    return run


def test_rescore_reuses_the_run_code_path_and_never_touches_the_source(stored_run, tmp_path):
    before = _tree_hash(stored_run)
    as_is = rescore_run(stored_run, tmp_path / "out_as_is")["av_speakerbench"]      # strategy from config_used
    assert as_is["accuracy"] == 33.3333                                             # official parser reads B in the reasoning
    assert as_is["perf"] == {"wall_s": 1.0}
    split = rescore_run(stored_run, tmp_path / "out_split", reasoning="split")["av_speakerbench"]
    assert split["accuracy"] == 66.6667 and split["correct"] == 2
    out = [json.loads(line) for line in (tmp_path / "out_split/m/av_speakerbench/records.jsonl").read_text().splitlines()]
    assert out[0]["response"] == "C" and out[0]["response_raw"].startswith("Option B") and out[0]["reasoning"]
    assert "response_raw" not in out[1]
    assert out[2]["parsed_answer"] == "" and out[2]["error"]                       # error rows keep failure values
    again = rescore_run(tmp_path / "out_split/m", tmp_path / "out_again", reasoning="split")["av_speakerbench"]
    assert again["accuracy"] == split["accuracy"]                                   # idempotent via response_raw
    assert _tree_hash(stored_run) == before, "source run modified"
    for bad_out in (stored_run.parent, stored_run):
        with pytest.raises(ValueError, match="overlap"):
            rescore_run(stored_run, bad_out)
