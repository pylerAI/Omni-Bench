"""Shared fixtures: synthetic media (ffmpeg) and a stubbed OpenAI client."""

from __future__ import annotations

import shutil
import subprocess
import types
from pathlib import Path

import pytest

import omni_bench.client as client_mod

FFMPEG = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _run(*args: str) -> None:
    subprocess.run([*FFMPEG, *args], check=True)


@pytest.fixture
def media(tmp_path: Path) -> types.SimpleNamespace:
    """A 2 s tone WAV, a 2 s MP4 with an audio track, and a 1 s MP4 without one."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    wav, mp4, silent = tmp_path / "clip.wav", tmp_path / "clip_video.mp4", tmp_path / "silent.mp4"
    _run("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-ar", "16000", "-ac", "1", str(wav))
    _run("-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=5", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=2", "-shortest", str(mp4))
    _run("-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5", str(silent))
    return types.SimpleNamespace(dir=tmp_path, wav=wav, mp4=mp4, silent=silent)


class OpenAIStub:
    """Stands in for ``openai.OpenAI``; records ``chat.completions.create`` kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.content = "A"
        self.reasoning: str | None = None

    def __call__(self, **_init):
        stub = self

        class _Completions:
            def create(self, **kwargs):
                stub.calls.append(kwargs)
                msg = types.SimpleNamespace(content=stub.content, reasoning=stub.reasoning, model_extra={})
                usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
                choice = types.SimpleNamespace(message=msg, finish_reason="stop")
                return types.SimpleNamespace(choices=[choice], usage=usage)

        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))

    @property
    def last(self) -> dict:
        return self.calls[-1]


@pytest.fixture
def openai_stub(monkeypatch) -> OpenAIStub:
    stub = OpenAIStub()
    monkeypatch.setattr(client_mod, "OpenAI", stub)
    return stub


def parts(call: dict) -> list[str]:
    return [p["type"] for p in call["messages"][-1]["content"]]


def text_of(call: dict) -> str:
    return [p for p in call["messages"][-1]["content"] if p["type"] == "text"][0]["text"]


def video_url_of(call: dict) -> str:
    return [p for p in call["messages"][-1]["content"] if p["type"] == "video_url"][0]["video_url"]["url"]
