"""Benchmark-level audio_mode override, ASR settings merge, and command pooling."""
import subprocess, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

sent = []
class _C:
    def create(self, **kw):
        sent.append(kw)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="A"))],
            usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))
class _OpenAI:
    def __init__(self, **kw): self.chat = types.SimpleNamespace(completions=_C())
m = types.ModuleType("openai"); m.OpenAI = _OpenAI; sys.modules["openai"] = m

from omni_bench.asr.schema import TranscriptionSegment
from omni_bench.asr.strategies import SttStrategy, register_strategy
from omni_bench.asr_client import (
    AsrCommandPool, AsrTextChatClient, NoAudioChatClient, build_chat_client,
    resolve_asr_settings, resolve_audio_mode,
)
from omni_bench.client import VllmChatClient
from omni_bench.config import BenchmarkConfig, ModelConfig, load_config

loads = []
@register_strategy("fake3")
class F(SttStrategy):
    def load(self): loads.append(self.spec.model)
    def _transcribe(self, audio_path, request):
        return [TranscriptionSegment(0, 0.0, 2.0, "A"*200)], {"language": "en"}

def text_of(c): return [p for p in c["messages"][-1]["content"] if p["type"]=="text"][0]["text"]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    wav = tmp/"a.wav"
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","lavfi","-i",
                    "sine=frequency=440:duration=2","-ar","16000","-ac","1",str(wav)],check=True)

    asr = {"strategy":{"name":"fake3","model":"fake/v3","device":"cpu"},"cache_dir":str(tmp/"c")}
    model = ModelConfig(name="q", weight_path="/x", base_url="http://x/v1",
                        extra={"audio_mode":"asr_text","asr":asr})

    def bm(name, **extra): return BenchmarkConfig(name=name, extra=extra)

    # 1. benchmark override beats the model setting
    assert resolve_audio_mode(model, bm("videomme", audio_mode="none")) == "none"
    assert resolve_audio_mode(model, bm("worldsense")) == "asr_text"   # no override -> model
    assert resolve_audio_mode(model, None) == "asr_text"
    assert resolve_audio_mode(ModelConfig(name="o", weight_path="/x"), bm("x")) == "native"
    # override can also turn ASR *on* for one benchmark of a native model
    omni = ModelConfig(name="omni", weight_path="/x", extra={"asr": asr})
    assert resolve_audio_mode(omni, bm("worldsense", audio_mode="asr_text")) == "asr_text"
    try:
        resolve_audio_mode(model, bm("x", audio_mode="weird")); raise SystemExit("must raise")
    except ValueError as e: print("bad override rejected:", e)

    # 2. client type follows the resolved mode
    pool = AsrCommandPool()
    c_vm = build_chat_client(model, benchmark=bm("videomme", audio_mode="none"), pool=pool)
    c_ws = build_chat_client(model, benchmark=bm("worldsense"), pool=pool)
    assert type(c_vm) is NoAudioChatClient and type(c_ws) is AsrTextChatClient

    # 3. Video-MME really gets no transcript even with audio present
    sent.clear()
    c_vm.complete("QUESTION", audio_path=wav)
    assert text_of(sent[-1]) == "QUESTION"
    assert [p["type"] for p in sent[-1]["messages"][-1]["content"]] == ["text"]
    print("videomme audio_mode=none -> no transcript, audio dropped")

    # 4. benchmark-level asr overrides merge over the model block
    merged = resolve_asr_settings(model, bm("omnidcbench", asr={"max_chars": 50,
                                  "strategy": {"language": "en"}}))
    assert merged.max_chars == 50
    assert merged.strategy.language == "en"
    assert merged.strategy.model == "fake/v3"        # untouched by the override
    assert merged.strategy.options == {}             # nested dict preserved, not wiped
    print("asr merge OK:", merged.max_chars, merged.strategy.language, merged.strategy.model)

    # 5. per-benchmark formatting differs while the engine is shared
    sent.clear()
    c_trunc = build_chat_client(model, benchmark=bm("omnidcbench", asr={"max_chars": 50}), pool=pool)
    c_trunc.complete("Q", audio_path=wav)
    assert "truncated" in text_of(sent[-1])
    sent.clear()
    c_ws.complete("Q", audio_path=wav)
    assert "truncated" not in text_of(sent[-1])
    print("per-benchmark max_chars OK; engine loads:", loads)
    assert len(loads) == 1, f"engine must load once, got {loads}"
    assert c_trunc.command is c_ws.command, "pool must hand out the same command"

    # 6. real config: videomme pinned to none, others inherit asr_text
    cfg = load_config("configs/models/qwen3_8_27b_whisper.yaml")
    modes = {b.name: resolve_audio_mode(cfg.models[0], b) for b in cfg.benchmarks}
    print("resolved modes:", modes)
    assert modes["videomme"] == "none"
    assert all(v == "asr_text" for k, v in modes.items() if k != "videomme")

    # 7. omni configs untouched
    cfgo = load_config("configs/models/qwen3_omni.yaml")
    modeso = {b.name: resolve_audio_mode(cfgo.models[0], b) for b in cfgo.benchmarks}
    print("qwen3-omni modes:", modeso)
    assert modeso["videomme"] == "none" and all(
        v == "native" for k, v in modeso.items() if k != "videomme")

print("\nALL OVERRIDE TESTS PASSED")
