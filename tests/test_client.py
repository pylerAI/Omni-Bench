"""Audio strategies and frames handling through the client (stubbed OpenAI SDK)."""

from __future__ import annotations

import pytest
from conftest import parts, text_of

from omni_bench.asr.schema import TranscriptionSegment
from omni_bench.asr.strategies import SttStrategy, register_strategy
from omni_bench.client import build_chat_client
from omni_bench.config import ModelConfig
from omni_bench.inference import ImageFrames, InferenceWarning
from omni_bench.inference.audio import AsrTextAudio, NativeAudio, NoAudio


@register_strategy("fake2")
class FakeStrategy(SttStrategy):
    def _transcribe(self, audio_path, request):
        return [TranscriptionSegment(0, 1.0, 4.0, f"spoken words from {audio_path.suffix}")], {"language": "en"}


@pytest.fixture
def asr_raw(tmp_path):
    return {"strategy": {"name": "fake2", "model": "fake/v2", "device": "cpu"}, "cache_dir": str(tmp_path / "cache")}


def model(mode, asr=None):
    extra = {"audio_mode": mode}
    if asr:
        extra["asr"] = asr
    return ModelConfig(name="m", weight_path="/x", base_url="http://x/v1", extra=extra)


FRAMES = lambda: ImageFrames(["data:image/jpeg;base64,xx"])  # noqa: E731


def test_registry_dispatch(asr_raw):
    audio_of = lambda c: type(c.pipeline.audio)  # noqa: E731
    assert audio_of(build_chat_client(model("native"))) is NativeAudio
    assert audio_of(build_chat_client(model("none"))) is NoAudio
    assert audio_of(build_chat_client(model("asr_text", asr_raw))) is AsrTextAudio
    assert audio_of(build_chat_client(ModelConfig(name="m", weight_path="/x"))) is NativeAudio  # default
    with pytest.raises(ValueError):
        build_chat_client(model("bogus"))


def test_none_drops_audio_and_forces_use_audio_in_video_off(openai_stub, media):
    build_chat_client(model("none")).complete(
        "QUESTION", video_path=media.mp4, audio_path=media.wav,
        extra_body={"mm_processor_kwargs": {"use_audio_in_video": True}})
    call = openai_stub.last
    assert parts(call) == ["video_url", "text"] and text_of(call) == "QUESTION"
    assert call["extra_body"]["mm_processor_kwargs"]["use_audio_in_video"] is False


def test_asr_text_with_audio_file_keeps_the_official_prompt(openai_stub, media, asr_raw):
    c = build_chat_client(model("asr_text", asr_raw), frame_modes=("client", "server"))   # worldsense-style
    c.complete("QUESTION", video_path=media.mp4, frames=FRAMES, audio_path=media.wav)
    body = text_of(openai_stub.last)
    assert parts(openai_stub.last) == ["image_url", "text"]
    assert body.endswith("\n\nQUESTION") and "[00:01] spoken words from .wav" in body


def test_asr_text_transcribes_the_audio_inside_the_video(openai_stub, media, asr_raw):
    build_chat_client(model("asr_text", asr_raw)).complete(       # av_speakerbench: server frames
        "QUESTION", video_path=media.mp4, extra_body={"mm_processor_kwargs": {"use_audio_in_video": True}})
    call = openai_stub.last
    assert parts(call) == ["video_url", "text"] and "spoken words from .wav" in text_of(call)
    assert call["extra_body"]["mm_processor_kwargs"]["use_audio_in_video"] is False


def test_client_frames_without_audio_file_inject_nothing(openai_stub, media, asr_raw):
    c = build_chat_client(model("asr_text", asr_raw), frame_modes=("client", "server"))   # video-mme shape
    c.complete("QUESTION", video_path=media.mp4, frames=FRAMES)
    assert text_of(openai_stub.last) == "QUESTION" and parts(openai_stub.last) == ["image_url", "text"]


def test_server_frames_send_the_video_and_transcribe_its_track(openai_stub, media, asr_raw):
    srv = ModelConfig(name="m", weight_path="/x", base_url="http://x/v1",
                      extra={"inference": {"audio": "asr_text", "frames": "server"}, "asr": asr_raw})
    c = build_chat_client(srv, frame_modes=("client", "server"))
    c.complete("QUESTION", video_path=media.mp4, frames=lambda: pytest.fail("client frames sampled"))
    assert parts(openai_stub.last) == ["video_url", "text"] and "spoken words" in text_of(openai_stub.last)
    # an adapter that only sends video keeps server even if the model asks for client
    cli = ModelConfig(name="m", weight_path="/x", extra={"inference": {"frames": "client"}})
    with pytest.warns(InferenceWarning):
        assert build_chat_client(cli, frame_modes=("server",)).frames_mode == "server"


def test_strict_cache_refuses_to_transcribe_inline(openai_stub, media, asr_raw):
    strict = dict(asr_raw, strict_cache=True, cache_dir=str(media.dir / "empty-cache"))
    with pytest.raises(FileNotFoundError, match="No cached transcript"):
        build_chat_client(model("asr_text", strict)).complete("Q", audio_path=media.wav)
