"""Benchmark-level inference override, ASR settings merge, command pooling, config warnings."""
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
import warnings
from omni_bench.client import build_chat_client
from omni_bench.config import BenchmarkConfig, ConfigWarning, ModelConfig, load_config
from omni_bench.inference import AsrCommandPool, InferenceWarning, resolve_asr_settings, resolve_inference
from omni_bench.inference.audio import AsrTextAudio, NoAudio

def resolve_audio_mode(model, benchmark):
    return resolve_inference(model, benchmark).audio

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
    assert type(c_vm.pipeline.audio) is NoAudio and type(c_ws.pipeline.audio) is AsrTextAudio

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
    assert c_trunc.pipeline.audio.command is c_ws.pipeline.audio.command, "pool must hand out the same command"

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

    # 8. frames: benchmark > model > adapter default; adapter limits apply
    fm = ("client", "server")
    plain = ModelConfig(name="p", weight_path="/x")
    srvm = ModelConfig(name="s", weight_path="/x", extra={"inference": {"frames": "server"}})
    assert resolve_inference(plain, bm("worldsense"), fm).frames == "client"
    assert resolve_inference(plain, bm("worldsense", frame_sampling="server"), fm).frames == "server"
    assert resolve_inference(plain, bm("worldsense", inference={"frames": "server"}), fm).frames == "server"
    assert resolve_inference(srvm, bm("worldsense"), fm).frames == "server"
    assert resolve_inference(srvm, bm("worldsense", inference={"frames": "client"}), fm).frames == "client"
    # explicit new key on a server-only adapter -> error
    try:
        resolve_inference(plain, bm("av_speakerbench", inference={"frames": "client"}), ("server",))
        raise SystemExit("unsupported benchmark frames must raise")
    except ValueError as e: print("unsupported frames rejected:", str(e)[:60])
    # legacy key there used to be ignored -> warning + adapter default
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = resolve_inference(plain, bm("av_speakerbench", frame_sampling="client"), ("server",)).frames
    assert got == "server", got
    assert any(issubclass(w.category, InferenceWarning) and "av_speakerbench" in str(w.message)
               for w in caught), [str(w.message) for w in caught]
    print("legacy frame_sampling on server-only adapter -> warn + server")
    # model-level typo -> error (no silent fallback)
    for typo_model in (ModelConfig(name="t", weight_path="/x", extra={"inference": {"frames": "sever"}}),
                       ModelConfig(name="t", weight_path="/x", extra={"frame_sampling": "sever"})):
        try:
            resolve_inference(typo_model, bm("worldsense"), fm)
            raise SystemExit("model frames typo must raise")
        except ValueError as e: print("model frames typo rejected:", str(e)[:60])
    try:
        resolve_inference(plain, bm("worldsense", frame_sampling="sever"), fm)
        raise SystemExit("benchmark frames typo must raise")
    except ValueError as e: print("benchmark frames typo rejected:", str(e)[:60])
    # model-level known mode the adapter can't run -> fallback + warning naming everything
    cli_model = ModelConfig(name="cm", weight_path="/x", extra={"inference": {"frames": "client"}})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = resolve_inference(cli_model, bm("omnidcbench"), ("server",)).frames
    msgs = [str(w.message) for w in caught if issubclass(w.category, InferenceWarning)]
    assert got == "server" and msgs and all(t in msgs[0] for t in ("omnidcbench", "'client'", "'server'")), msgs
    print("unsupported model frames -> fallback + warning:", msgs[0][:80])
    # no warning on the normal paths
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_inference(srvm, bm("worldsense"), fm); resolve_inference(plain, bm("av_speakerbench"), ("server",))
    assert not [w for w in caught if issubclass(w.category, InferenceWarning)]
    try:
        resolve_inference(plain, bm("videomme", inference={"transport": "base64"}), fm)
        raise SystemExit("transport is model-only")
    except ValueError as e: print("benchmark transport rejected:", str(e)[:60])
    print("frames precedence OK")

    # 9. unknown config keys warn; known extras stay silent
    cfg_path = tmp / "m.yaml"
    cfg_path.write_text("models:\n- name: t\n  weight_path: /x\n  audio_mdoe: none\n"
                        "benchmarks:\n- name: worldsense\n  num_frames: 8\n  num_frame: 4\n  max_workers: 2\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_config(cfg_path, cfg_path)
    msgs = [str(w.message) for w in caught if issubclass(w.category, ConfigWarning)]
    assert any("audio_mdoe" in m for m in msgs) and any("num_frame'" in m for m in msgs), msgs
    assert not any("'num_frames'" in m.split("known")[0] or "'max_workers'" in m.split("known")[0]
                   for m in msgs), msgs
    print("unknown-key warnings OK:", len(msgs))

    # 10. benchmark config: CLI > model config `benchmark_config:` (relative to the file) > default
    import os
    from omni_bench.config import DEFAULT_BENCHMARK_CONFIG, resolve_benchmark_config
    sub = tmp / "cfgs" / "nested"; sub.mkdir(parents=True)
    (sub / "pair.yaml").write_text("benchmarks:\n- name: worldsense\n  num_frames: 3\n")
    (tmp / "cli.yaml").write_text("benchmarks:\n- name: videomme\n")
    keyed = sub / "model.yaml"
    keyed.write_text("benchmark_config: pair.yaml\nmodels:\n- name: k\n  weight_path: /x\n")
    plain_cfg = tmp / "plain.yaml"
    plain_cfg.write_text("models:\n- name: p\n  weight_path: /x\n")
    cwd = os.getcwd(); os.chdir("/")                 # relative path must not depend on the CWD
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")           # the key is known: no ConfigWarning
            got = load_config(keyed)
        assert got.benchmark_config_path == (sub / "pair.yaml").resolve(), got.benchmark_config_path
        assert [b.name for b in got.benchmarks] == ["worldsense"] and got.benchmarks[0].extra["num_frames"] == 3
        cli = load_config(keyed, tmp / "cli.yaml")    # CLI wins over the key
        assert [b.name for b in cli.benchmarks] == ["videomme"]
        assert load_config(plain_cfg).benchmark_config_path == DEFAULT_BENCHMARK_CONFIG
        assert resolve_benchmark_config(None, None) == DEFAULT_BENCHMARK_CONFIG
        assert resolve_benchmark_config(keyed, None) == (sub / "pair.yaml").resolve()
    finally:
        os.chdir(cwd)
    print("benchmark_config precedence OK")

print("\nALL OVERRIDE TESTS PASSED")
