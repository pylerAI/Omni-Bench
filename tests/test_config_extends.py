"""`extends:` in model configs: merge rules, paths, chains, cycles, warnings."""

from __future__ import annotations

from pathlib import Path

import pytest

from omni_bench.config import deep_merge, load_config, load_raw_config

REPO = Path(__file__).resolve().parents[1]


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def bench(tmp_path):
    return write(tmp_path / "bench.yaml", "benchmarks:\n- name: worldsense\n")


def test_deep_merge_dicts_merge_lists_replace():
    base = {"a": {"x": 1, "y": [1, 2]}, "keep": True}
    over = {"a": {"y": [3], "z": 2}}
    assert deep_merge(base, over) == {"a": {"x": 1, "y": [3], "z": 2}, "keep": True}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "keep": True}           # inputs untouched


def test_single_model_is_merged_and_child_wins(tmp_path, bench):
    write(tmp_path / "base" / "b.yaml",
          "global:\n  request_timeout_s: 3600\n"
          "models:\n- weight_path: /w\n  extra_body:\n    top_p: 0.95\n    chat_template_kwargs:\n      enable_thinking: false\n"
          "  vllm:\n    extra_args: [--a, --b]\n")
    child = write(tmp_path / "c.yaml",
                  "extends: base/b.yaml\nmodels:\n- name: c\n  extra_body:\n    chat_template_kwargs:\n      enable_thinking: true\n"
                  "  vllm:\n    extra_args: [--z]\n")
    cfg = load_config(child, bench)
    m = cfg.models[0]
    assert m.name == "c" and m.weight_path == "/w" and cfg.request_timeout_s == 3600
    assert m.extra["extra_body"] == {"top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True}}
    assert m.vllm.extra_args == ["--z"]                                  # lists replaced wholesale
    assert cfg.config_chain == [(tmp_path / "base" / "b.yaml").resolve(), child.resolve()]


def test_paths_resolve_against_the_declaring_file(tmp_path, monkeypatch):
    write(tmp_path / "benchmarks" / "pair.yaml", "benchmarks:\n- name: videomme\n")
    write(tmp_path / "models" / "_base" / "b.yaml",
          "benchmark_config: ../../benchmarks/pair.yaml\nglobal:\n  result_dir: ../../out\n"
          "models:\n- weight_path: /w\n")
    child = write(tmp_path / "models" / "c.yaml", "extends: _base/b.yaml\nmodels:\n- name: c\n")
    monkeypatch.chdir("/")
    cfg = load_config(child)
    assert cfg.benchmark_config_path == (tmp_path / "benchmarks" / "pair.yaml").resolve()
    assert cfg.result_dir == (tmp_path / "out").resolve()


def test_chain_and_list_of_bases(tmp_path, bench):
    write(tmp_path / "a.yaml", "models:\n- weight_path: /a\n  inference:\n    audio: asr_text\n")
    write(tmp_path / "b.yaml", "extends: a.yaml\nmodels:\n- inference:\n    transport: base64\n")
    write(tmp_path / "mixin.yaml", "models:\n- asr:\n    strict_cache: true\n")
    child = write(tmp_path / "c.yaml", "extends: [b.yaml, mixin.yaml]\nmodels:\n- name: c\n")
    raw, chain = load_raw_config(child)
    assert [p.name for p in chain] == ["a.yaml", "b.yaml", "mixin.yaml", "c.yaml"]
    model = load_config(child, bench).models[0]
    assert model.extra["inference"] == {"audio": "asr_text", "transport": "base64"}
    assert model.extra["asr"] == {"strict_cache": True} and model.weight_path == "/a"
    assert "extends" not in raw


def test_cycle_is_an_error(tmp_path, bench):
    write(tmp_path / "a.yaml", "extends: b.yaml\nmodels:\n- name: a\n  weight_path: /x\n")
    write(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="extends cycle"):
        load_config(tmp_path / "a.yaml", bench)


def test_unknown_keys_in_a_child_are_still_warned(tmp_path, bench, capsys):
    write(tmp_path / "b.yaml", "models:\n- weight_path: /w\n")
    child = write(tmp_path / "c.yaml", "extends: b.yaml\nmodels:\n- name: c\n  audio_mdoe: none\n")
    load_config(child, bench)
    err = capsys.readouterr().err
    assert err.startswith("warning: ") and "audio_mdoe" in err


def test_shipped_measurement_configs_use_extends():
    cfg = load_config(REPO / "configs/models/nemotron_3_5_super_vl_nvfp4_think.yaml")
    m = cfg.models[0]
    assert len(cfg.config_chain) == 3 and m.name == "nemotron-3.5-super-vl-nvfp4-think"
    assert m.extra["extra_body"] == {"top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True}}
    assert m.extra["asr"]["strategy"]["name"] == "faster_whisper"
    asr = load_config(REPO / "configs/models/nemotron_3_5_super_vl_bf16_nothink_nemotron_asr.yaml").models[0]
    assert asr.extra["asr"]["strategy"] == {"name": "nemo_streaming", "model": "nvidia/nemotron-3.5-asr-streaming-0.6b"}
