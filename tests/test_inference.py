"""Inference pipeline and adapter parse/finalize contract (no server, no data)."""

from __future__ import annotations

import json
import types
import warnings

import pytest

import omni_bench.client as client_mod
from omni_bench.adapters import ADAPTER_NAMES, get_adapter
from omni_bench.client import build_chat_client
from omni_bench.config import BenchmarkConfig, ModelConfig
from omni_bench.inference import (
    AUDIO_STRATEGIES,
    FRAME_STRATEGIES,
    REASONING_STRATEGIES,
    TRANSPORTS,
    ImageFrames,
    InferencePipeline,
    InferenceWarning,
    MediaRequest,
    VideoFrames,
    resolve_inference,
)


class _Recorder:
    """Stub for ``openai.OpenAI``: records ``chat.completions.create`` kwargs."""

    def __init__(self, content: str = "A") -> None:
        self.calls: list[dict] = []
        self.content = content

    def __call__(self, **_init):
        recorder = self

        class _Completions:
            def create(self, **kwargs):
                recorder.calls.append(kwargs)
                msg = types.SimpleNamespace(content=recorder.content, model_extra={})
                usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=1, total_tokens=11)
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)

        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))


@pytest.fixture
def openai_stub(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(client_mod, "OpenAI", recorder)
    return recorder


def model(**extra) -> ModelConfig:
    return ModelConfig(name="m", weight_path="/x", base_url="http://x/v1", extra=extra)


def parts(call: dict) -> list[str]:
    return [p["type"] for p in call["messages"][-1]["content"]]


# --- registries and settings -------------------------------------------------

def test_registries_hold_the_default_options():
    assert "native" in AUDIO_STRATEGIES.names()
    assert {"client", "server"} <= set(FRAME_STRATEGIES.names())
    assert "file" in TRANSPORTS.names()
    assert "as_is" in REASONING_STRATEGIES.names()
    with pytest.raises(ValueError, match="Unknown transport"):
        TRANSPORTS.get("carrier-pigeon")


def test_defaults_without_inference_block():
    s = resolve_inference(model())
    assert (s.audio, s.frames, s.transport, s.reasoning) == ("native", "server", "file", "as_is")


def test_inference_block_typos_raise():
    with pytest.raises(ValueError, match="unknown inference key"):
        resolve_inference(model(inference={"transprot": "file"}))
    with pytest.raises(ValueError, match="Unknown audio"):
        resolve_inference(model(inference={"audio": "loud"}))
    with pytest.raises(ValueError, match="unknown inference key"):
        resolve_inference(model(), BenchmarkConfig(name="videomme", extra={"inference": {"transport": "file"}}))


def test_frames_follow_the_adapter():
    for name in ADAPTER_NAMES:
        adapter = get_adapter(name)
        assert resolve_inference(model(), BenchmarkConfig(name=name), adapter.frame_modes).frames == adapter.frame_modes[0]
    assert get_adapter("av_speakerbench").frame_modes[0] == "server"
    assert get_adapter("worldsense").frame_modes[0] == "client"


def test_frames_model_choice_the_adapter_cannot_run_warns_and_falls_back():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = resolve_inference(model(inference={"frames": "client"}), BenchmarkConfig(name="omnidcbench"), ("server",))
    assert got.frames == "server"
    assert any(issubclass(w.category, InferenceWarning) and "omnidcbench" in str(w.message) for w in caught)


def test_frames_benchmark_choice_the_adapter_cannot_run_raises():
    with pytest.raises(ValueError, match="supports only"):
        resolve_inference(model(), BenchmarkConfig(name="av_speakerbench", extra={"inference": {"frames": "client"}}), ("server",))


# --- request assembly -------------------------------------------------------

def test_server_frames_send_the_video_as_file_url(tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    built = InferencePipeline.build(model()).build_request(
        MediaRequest(prompt="Q", video_path=clip, system_prompt="S",
                     extra_body={"mm_processor_kwargs": {"use_audio_in_video": True}})
    )
    assert built.messages[0] == {"role": "system", "content": "S"}
    content = built.messages[-1]["content"]
    assert content[0] == {"type": "video_url", "video_url": {"url": clip.resolve().as_uri()}}
    assert content[-1] == {"type": "text", "text": "Q"}
    assert built.extra_body == {"mm_processor_kwargs": {"use_audio_in_video": True}}


def test_client_frames_as_images_then_audio_then_text(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    pipe = InferencePipeline.build(model(), BenchmarkConfig(name="worldsense"), frame_modes=("client",))
    built = pipe.build_request(MediaRequest(prompt="Q", video_path="/not/sent.mp4", audio_path=wav,
                                            frames=lambda: ImageFrames(["data:image/jpeg;base64,AA", "data:image/jpeg;base64,BB"])))
    content = built.messages[-1]["content"]
    assert [p["type"] for p in content] == ["image_url", "image_url", "audio_url", "text"]
    assert content[2]["audio_url"]["url"] == wav.resolve().as_uri()


def test_client_frames_as_one_video_merge_their_body_hints():
    pipe = InferencePipeline.build(model(extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
                                   frame_modes=("client",))
    hints = {"media_io_kwargs": {"video": {"num_frames": 3}}, "mm_processor_kwargs": {"use_audio_in_video": False}}
    built = pipe.build_request(MediaRequest(prompt="Q", video_path="/x.mp4", top_p=0.9, do_sample=True,
                                            frames=lambda: VideoFrames("data:video/jpeg;base64,AA", hints)))
    assert built.messages[-1]["content"][0] == {"type": "video_url", "video_url": {"url": "data:video/jpeg;base64,AA"}}
    assert built.extra_body == {"chat_template_kwargs": {"enable_thinking": False}, **hints,
                                "top_p": 0.9, "do_sample": True}


def test_client_frames_are_sampled_lazily():
    pipe = InferencePipeline.build(model())                 # server: frames never requested
    built = pipe.build_request(MediaRequest(prompt="Q", frames=lambda: pytest.fail("sampled")))
    assert built.messages[-1]["content"] == [{"type": "text", "text": "Q"}]


def test_client_complete_sends_the_built_request(openai_stub, tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    client = build_chat_client(model(), benchmark=BenchmarkConfig(name="av_speakerbench"), frame_modes=("server",))
    result = client.complete("Q", video_path=clip, max_tokens=7, temperature=0.5)
    call = openai_stub.calls[-1]
    assert call["model"] == "m" and call["max_tokens"] == 7 and call["temperature"] == 0.5
    assert parts(call) == ["video_url", "text"] and call["extra_body"] is None
    assert result.text == "A" and result.completion_tokens == 1


# --- adapter contract ----------------------------------------------------------

def test_every_adapter_implements_parse_and_finalize():
    for name in ADAPTER_NAMES:
        adapter = get_adapter(name)                         # abstract methods would raise TypeError
        assert callable(adapter.parse_record) and callable(adapter.finalize)


@pytest.mark.parametrize("name,record,expected", [
    ("av_speakerbench", {"answer": "B", "response": "The best answer is B."}, {"parsed_answer": "B", "is_correct": True}),
    ("worldsense", {"answer": "C", "response": "Answer: D"}, {"parsed_answer": "D", "is_correct": False, "score": 0}),
    ("worldsense", {"answer": "C", "response": "no idea at all"}, {"parsed_answer": "", "is_correct": False, "score": -1}),
    ("videomme", {"answer": "A", "response": "A"}, {"parsed_answer": "A", "is_correct": True}),
    ("omnivideobench", {"answer": "B. dogs", "response": "B", "prompt": "P"}, {"parsed_answer": "B", "is_correct": True}),
])
def test_parse_record_uses_the_official_parser(name, record, expected):
    assert get_adapter(name).parse_record(record) == expected


def test_omnidcbench_parse_record_reads_the_json_prediction():
    got = get_adapter("omnidcbench").parse_record({"prediction": '[{"timestamp": "00:00-00:05", "caption": "a"}]'})
    assert got["prediction_json"] and got["prediction_json"][0]["caption"] == "a"


def test_finalize_summarizes_stored_records(tmp_path):
    records = [
        {"question_id": 1, "category": "c", "sub_category": "s", "task_id": "t", "answer": "A", "response": "A"},
        {"question_id": 2, "category": "c", "sub_category": "s", "task_id": "t", "answer": "B", "response": "C"},
    ]
    adapter = get_adapter("av_speakerbench")
    for r in records:
        r.update(adapter.parse_record(r))
    summary = adapter.finalize(records, benchmark=BenchmarkConfig(name="av_speakerbench"), output_dir=tmp_path,
                               frames_mode="server")
    assert summary["total"] == 2 and summary["correct"] == 1 and summary["accuracy"] == 50.0
    assert json.loads((tmp_path / "summary.json").read_text())["accuracy"] == 50.0
    assert len(json.loads((tmp_path / "records.json").read_text())) == 2
