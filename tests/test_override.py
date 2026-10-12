"""Benchmark-level overrides, ASR settings merge and engine pooling, frames
precedence, config-key warnings, and benchmark_config resolution."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from conftest import parts, text_of

from omni_bench.asr.schema import TranscriptionSegment
from omni_bench.asr.strategies import SttStrategy, register_strategy
from omni_bench.client import build_chat_client
from omni_bench.config import (
    DEFAULT_BENCHMARK_CONFIG,
    BenchmarkConfig,
    ConfigWarning,
    ModelConfig,
    load_config,
    resolve_benchmark_config,
)
from omni_bench.inference import AsrCommandPool, InferenceWarning, resolve_asr_settings, resolve_inference
from omni_bench.inference.audio import AsrTextAudio, NoAudio

REPO = Path(__file__).resolve().parents[1]
loads: list[str] = []


@register_strategy("fake3")
class F(SttStrategy):
    def load(self):
        loads.append(self.spec.model)

    def _transcribe(self, audio_path, request):
        return [TranscriptionSegment(0, 0.0, 2.0, "A" * 200)], {"language": "en"}


def bm(name, **extra):
    return BenchmarkConfig(name=name, extra=extra)


def audio_mode(model, benchmark):
    return resolve_inference(model, benchmark).audio


@pytest.fixture
def asr(tmp_path):
    return {"strategy": {"name": "fake3", "model": "fake/v3", "device": "cpu"}, "cache_dir": str(tmp_path / "c")}


@pytest.fixture
def cascade(asr):
    return ModelConfig(name="q", weight_path="/x", base_url="http://x/v1", extra={"audio_mode": "asr_text", "asr": asr})


def test_benchmark_audio_overrides_the_model(cascade, asr):
    assert audio_mode(cascade, bm("videomme", audio_mode="none")) == "none"
    assert audio_mode(cascade, bm("worldsense")) == "asr_text"
    assert audio_mode(cascade, None) == "asr_text"
    assert audio_mode(ModelConfig(name="o", weight_path="/x"), bm("x")) == "native"
    omni = ModelConfig(name="omni", weight_path="/x", extra={"asr": asr})   # override can turn ASR on
    assert audio_mode(omni, bm("worldsense", audio_mode="asr_text")) == "asr_text"
    with pytest.raises(ValueError):
        audio_mode(cascade, bm("x", audio_mode="weird"))


def test_engine_shared_while_formatting_differs_per_benchmark(openai_stub, media, cascade):
    loads.clear()
    pool = AsrCommandPool()
    c_vm = build_chat_client(cascade, benchmark=bm("videomme", audio_mode="none"), pool=pool)
    c_ws = build_chat_client(cascade, benchmark=bm("worldsense"), pool=pool)
    assert type(c_vm.pipeline.audio) is NoAudio and type(c_ws.pipeline.audio) is AsrTextAudio

    c_vm.complete("QUESTION", audio_path=media.wav)                 # no transcript, audio dropped
    assert text_of(openai_stub.last) == "QUESTION" and parts(openai_stub.last) == ["text"]

    merged = resolve_asr_settings(cascade, bm("omnidcbench", asr={"max_chars": 50, "strategy": {"language": "en"}}))
    assert merged.max_chars == 50 and merged.strategy.language == "en"
    assert merged.strategy.model == "fake/v3" and merged.strategy.options == {}

    c_trunc = build_chat_client(cascade, benchmark=bm("omnidcbench", asr={"max_chars": 50}), pool=pool)
    c_trunc.complete("Q", audio_path=media.wav)
    assert "truncated" in text_of(openai_stub.last)
    c_ws.complete("Q", audio_path=media.wav)
    assert "truncated" not in text_of(openai_stub.last)
    assert len(loads) == 1, f"engine must load once, got {loads}"
    assert c_trunc.pipeline.audio.command is c_ws.pipeline.audio.command


def test_shipped_configs_resolve_as_documented():
    cfg = load_config(REPO / "configs/models/qwen3_8_27b_whisper.yaml")
    modes = {b.name: audio_mode(cfg.models[0], b) for b in cfg.benchmarks}
    assert modes["videomme"] == "none" and all(v == "asr_text" for k, v in modes.items() if k != "videomme")
    cfgo = load_config(REPO / "configs/models/qwen3_omni.yaml")
    modeso = {b.name: audio_mode(cfgo.models[0], b) for b in cfgo.benchmarks}
    assert modeso["videomme"] == "none" and all(v == "native" for k, v in modeso.items() if k != "videomme")


FM = ("client", "server")
PLAIN = ModelConfig(name="p", weight_path="/x")
SRV = ModelConfig(name="s", weight_path="/x", extra={"inference": {"frames": "server"}})


def test_frames_precedence_benchmark_model_default():
    assert resolve_inference(PLAIN, bm("worldsense"), FM).frames == "client"
    assert resolve_inference(PLAIN, bm("worldsense", frame_sampling="server"), FM).frames == "server"
    assert resolve_inference(PLAIN, bm("worldsense", inference={"frames": "server"}), FM).frames == "server"
    assert resolve_inference(SRV, bm("worldsense"), FM).frames == "server"
    assert resolve_inference(SRV, bm("worldsense", inference={"frames": "client"}), FM).frames == "client"
    with warnings.catch_warnings():
        warnings.simplefilter("error", InferenceWarning)          # normal paths do not warn
        resolve_inference(SRV, bm("worldsense"), FM)
        resolve_inference(PLAIN, bm("av_speakerbench"), ("server",))


def test_frames_unsupported_or_misspelled():
    with pytest.raises(ValueError, match="supports only"):        # explicit new key -> error
        resolve_inference(PLAIN, bm("av_speakerbench", inference={"frames": "client"}), ("server",))
    with pytest.warns(InferenceWarning, match="av_speakerbench"):  # legacy key used to be ignored
        assert resolve_inference(PLAIN, bm("av_speakerbench", frame_sampling="client"), ("server",)).frames == "server"
    for typo in (ModelConfig(name="t", weight_path="/x", extra={"inference": {"frames": "sever"}}),
                 ModelConfig(name="t", weight_path="/x", extra={"frame_sampling": "sever"})):
        with pytest.raises(ValueError, match="Unknown frames"):
            resolve_inference(typo, bm("worldsense"), FM)
    with pytest.raises(ValueError, match="Unknown frames"):
        resolve_inference(PLAIN, bm("worldsense", frame_sampling="sever"), FM)
    cli_model = ModelConfig(name="cm", weight_path="/x", extra={"inference": {"frames": "client"}})
    with pytest.warns(InferenceWarning) as caught:
        assert resolve_inference(cli_model, bm("omnidcbench"), ("server",)).frames == "server"
    assert all(t in str(caught[0].message) for t in ("omnidcbench", "'client'", "'server'"))
    with pytest.raises(ValueError, match="unknown inference key"):
        resolve_inference(PLAIN, bm("videomme", inference={"transport": "base64"}), FM)


def test_unknown_config_keys_warn_and_known_extras_stay_silent(tmp_path):
    cfg_path = tmp_path / "m.yaml"
    cfg_path.write_text("models:\n- name: t\n  weight_path: /x\n  audio_mdoe: none\n"
                        "benchmarks:\n- name: worldsense\n  num_frames: 8\n  num_frame: 4\n  max_workers: 2\n")
    with pytest.warns(ConfigWarning) as caught:
        load_config(cfg_path, cfg_path)
    msgs = [str(w.message) for w in caught]
    assert any("audio_mdoe" in m for m in msgs) and any("num_frame'" in m for m in msgs)
    assert not any("'num_frames'" in m.split("known")[0] or "'max_workers'" in m.split("known")[0] for m in msgs)


def test_benchmark_config_precedence_and_relative_path(tmp_path, monkeypatch):
    sub = tmp_path / "cfgs" / "nested"
    sub.mkdir(parents=True)
    (sub / "pair.yaml").write_text("benchmarks:\n- name: worldsense\n  num_frames: 3\n")
    (tmp_path / "cli.yaml").write_text("benchmarks:\n- name: videomme\n")
    keyed = sub / "model.yaml"
    keyed.write_text("benchmark_config: pair.yaml\nmodels:\n- name: k\n  weight_path: /x\n")
    plain = tmp_path / "plain.yaml"
    plain.write_text("models:\n- name: p\n  weight_path: /x\n")
    monkeypatch.chdir("/")                                         # must not depend on the CWD
    with warnings.catch_warnings():
        warnings.simplefilter("error")                             # the key is known: no ConfigWarning
        got = load_config(keyed)
    assert got.benchmark_config_path == (sub / "pair.yaml").resolve()
    assert [b.name for b in got.benchmarks] == ["worldsense"] and got.benchmarks[0].extra["num_frames"] == 3
    assert [b.name for b in load_config(keyed, tmp_path / "cli.yaml").benchmarks] == ["videomme"]   # CLI wins
    assert load_config(plain).benchmark_config_path == DEFAULT_BENCHMARK_CONFIG
    assert resolve_benchmark_config(None, None) == DEFAULT_BENCHMARK_CONFIG
    assert resolve_benchmark_config(keyed, None) == (sub / "pair.yaml").resolve()


def test_report_labels_come_from_display_name(tmp_path):
    from omni_bench import report

    (tmp_path / "new-model").mkdir()
    (tmp_path / "new-model" / "config_used.json").write_text(json.dumps({"model": {"display_name": "New Model 9B"}}))
    (tmp_path / "qwen3-omni").mkdir()                              # older run: no config_used.json
    (tmp_path / "unnamed").mkdir()
    report.DISPLAY_NAMES.clear()
    report.DISPLAY_NAMES.update(report.load_display_names(tmp_path))
    assert report.model_label("new-model") == "New Model 9B"
    assert report.model_label("qwen3-omni") == "Qwen3-Omni-30B-A3B-Instruct"   # legacy fallback
    assert report.model_label("unnamed") == "unnamed"
    cfg = load_config(REPO / "configs/models/qwen3_8_27b_whisper_thinking.yaml")
    assert cfg.models[0].display_name == "Qwen3.8-27B + Whisper (thinking)"
