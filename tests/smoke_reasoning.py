"""Reasoning strategies (as_is / split) and `omni-bench rescore` (no server)."""
import hashlib, json, sys, tempfile, types
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

reply = {"content": "", "reasoning": None}

class _Completions:
    def create(self, **kwargs):
        msg = types.SimpleNamespace(content=reply["content"], reasoning=reply["reasoning"], model_extra={})
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")], usage=None)

class _OpenAI:
    def __init__(self, **kwargs): self.chat = types.SimpleNamespace(completions=_Completions())

openai_stub = types.ModuleType("openai"); openai_stub.OpenAI = _OpenAI; sys.modules["openai"] = openai_stub

from omni_bench.client import build_chat_client
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.inference import REASONING_STRATEGIES, resolve_inference
from omni_bench.rescore import rescore_run

# --- split rules ---
tt = REASONING_STRATEGIES.get("split")()
assert tt.split("<think>\nstep A then B\n</think>\n\nB", None) == ("B", "step A then B")
assert tt.split("a </think> b </think> C", None) == ("C", "a </think> b")          # LAST </think>
assert tt.split("plain B", "srv") == ("plain B", "srv")                            # no tag -> unchanged
assert tt.split("x</think>D", "srv") == ("D", "srv\n\nx")                          # appended to server reasoning
assert REASONING_STRATEGIES.get("as_is")().split("x</think>D", "srv") == ("x</think>D", "srv")
for old in ("server", "think_tag", "none"):                                        # old names are gone
    try:
        REASONING_STRATEGIES.get(old); raise SystemExit(f"{old} must be unknown")
    except ValueError: pass
print("split rules OK")

# --- client applies it before the adapter sees the text ---
def model(**inference):
    return ModelConfig(name="m", weight_path="/x", base_url="http://x/v1", extra={"inference": inference})

reply.update(content="think...\n</think>\n\nB", reasoning=None)
res = build_chat_client(model(reasoning="split")).complete("Q")
assert res.text == "B" and res.reasoning == "think..." and res.meta()["response_raw"] == reply["content"]
res = build_chat_client(model()).complete("Q")                                      # default as_is
assert res.text == reply["content"] and "response_raw" not in res.meta()
reply.update(content="B", reasoning="server-side")
res = build_chat_client(model(reasoning="split")).complete("Q")
assert res.text == "B" and res.reasoning == "server-side" and "response_raw" not in res.meta()
res = build_chat_client(model()).complete("Q")
assert res.text == "B" and res.reasoning == "server-side"                          # as_is keeps server reasoning
for bad in ({"inference": {"reasoning": "splt"}},):
    try:
        resolve_inference(ModelConfig(name="m", weight_path="/x", extra=bad)); raise SystemExit("typo must raise")
    except ValueError as exc: print("reasoning typo rejected:", str(exc)[:50])
try:
    resolve_inference(model(), BenchmarkConfig(name="worldsense", extra={"inference": {"reasoning": "as_is"}}))
    raise SystemExit("benchmark-level reasoning must raise")
except ValueError: pass
print("client reasoning OK")

# --- rescore: same code path as the run, source untouched ---
def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    run = tmp / "results" / "m"
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
    before = tree_hash(run)

    srv = rescore_run(run, tmp / "out_server")["av_speakerbench"]       # strategy from config_used
    assert srv["accuracy"] == 33.3333, srv                              # official parser reads the B in the reasoning
    assert srv["perf"] == {"wall_s": 1.0}
    tag = rescore_run(run, tmp / "out_tag", reasoning="split")["av_speakerbench"]
    assert tag["accuracy"] == 66.6667 and tag["correct"] == 2, tag
    out = [json.loads(l) for l in (tmp / "out_tag/m/av_speakerbench/records.jsonl").read_text().splitlines()]
    assert out[0]["response"] == "C" and out[0]["response_raw"].startswith("Option B") and out[0]["reasoning"]
    assert "response_raw" not in out[1]
    assert out[2]["parsed_answer"] == "" and out[2]["error"]           # error rows keep their failure values
    again = rescore_run(tmp / "out_tag/m", tmp / "out_again", reasoning="split")["av_speakerbench"]
    assert again["accuracy"] == tag["accuracy"]                          # idempotent via response_raw
    assert tree_hash(run) == before, "source run modified"
    for bad_out in (run.parent, run):
        try:
            rescore_run(run, bad_out); raise SystemExit("overlapping --out-dir must raise")
        except ValueError: pass
    print("rescore OK: as_is", srv["accuracy"], "split", tag["accuracy"])

print("\nALL REASONING SMOKE TESTS PASSED")
