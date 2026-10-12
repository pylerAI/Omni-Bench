"""Video transport (file / base64 + transcode cache), strip_mm_kwargs, the
inference block vs legacy keys, and resume of error rows."""

from __future__ import annotations

import base64
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from conftest import FFMPEG, video_url_of

from omni_bench import video_transport as vt
from omni_bench.asr.schema import TranscriptionSegment
from omni_bench.asr.strategies import SttStrategy, register_strategy
from omni_bench.client import build_chat_client
from omni_bench.config import ModelConfig
from omni_bench.io import append_jsonl, load_resumable_records, read_jsonl_records

seen_audio: list[Path] = []
FLAGS = {"mm_processor_kwargs": {"use_audio_in_video": True}, "media_io_kwargs": {"video": {"fps": 2}}}


@register_strategy("fake_transport")
class FakeStrategy(SttStrategy):
    def _transcribe(self, audio_path, request):
        seen_audio.append(Path(audio_path))
        return [TranscriptionSegment(0, 0.0, 1.0, "hello")], {"language": "en"}


def model(**extra):
    return ModelConfig(name="m", weight_path="/x", base_url="http://x/v1", extra=extra)


@pytest.fixture
def clips(media):
    """A lossless 1080p clip (large enough to exceed a tiny transcode threshold) and a fake webm."""
    mp4 = media.dir / "clip.mp4"
    subprocess.run([*FFMPEG, "-f", "lavfi", "-i", "testsrc=duration=2:size=1920x1080:rate=10", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=2", "-shortest", "-c:v", "libx264", "-crf", "0", str(mp4)],
                   check=True)
    webm = media.dir / "clip.webm"
    webm.write_bytes(b"fake-webm")
    return media.dir, mp4, webm


def test_default_transport_is_file_url(openai_stub, clips):
    _, mp4, _ = clips
    build_chat_client(model(audio_mode="none")).complete("Q", video_path=mp4)
    assert video_url_of(openai_stub.last).startswith("file://")


def test_base64_below_threshold_sends_original_bytes_with_mime(openai_stub, clips):
    tmp, mp4, webm = clips
    c = build_chat_client(model(audio_mode="none", video_transport="base64",
                                transcode={"threshold_mb": 1000, "cache_dir": str(tmp / "tc")}))
    c.complete("Q", video_path=mp4)
    url = video_url_of(openai_stub.last)
    assert url.startswith("data:video/mp4;base64,") and base64.b64decode(url.split(",", 1)[1]) == mp4.read_bytes()
    c.complete("Q", video_path=webm)
    assert video_url_of(openai_stub.last).startswith("data:video/webm;base64,")


def test_strip_mm_kwargs_removes_the_keys_and_reasoning_is_recorded(openai_stub, clips):
    tmp, _, webm = clips
    big = {"threshold_mb": 1000, "cache_dir": str(tmp / "tc")}
    c = build_chat_client(model(audio_mode="none", video_transport="base64", transcode=big))
    c.complete("Q", video_path=webm, extra_body=FLAGS)
    assert openai_stub.last["extra_body"]["mm_processor_kwargs"]["use_audio_in_video"] is False   # none only sets False
    openai_stub.content, openai_stub.reasoning = "B", "thinking..."
    c2 = build_chat_client(model(audio_mode="none", video_transport="base64", transcode=big, strip_mm_kwargs=True,
                                 extra_body={"chat_template_kwargs": {"enable_thinking": True}}))
    res = c2.complete("Q", video_path=webm, extra_body=FLAGS)
    body = openai_stub.last["extra_body"]
    assert "mm_processor_kwargs" not in body and "media_io_kwargs" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert res.reasoning == "thinking..." and res.finish_reason == "stop" and res.meta()["completion_tokens"] == 5


def test_concurrent_callers_share_one_transcode_and_asr_reads_the_original(openai_stub, clips, monkeypatch):
    tmp, mp4, _ = clips
    small = vt.TranscodeSettings.from_dict({"threshold_mb": 0.01, "cache_dir": str(tmp / "tc2"), "threads": 2})
    calls: list[int] = []
    real = vt._run_ffmpeg

    def counting(*a, **k):
        calls.append(threading.get_ident())
        return real(*a, **k)

    monkeypatch.setattr(vt, "_run_ffmpeg", counting)
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda _: vt.ensure_transcoded(mp4, small), range(8)))
    assert len(set(outs)) == 1 and outs[0].exists()
    assert len(set(calls)) == 1, f"transcoded by {len(set(calls))} threads"
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,height",
                            "-of", "csv=p=0", str(outs[0])], capture_output=True, text=True).stdout
    assert "audio" not in probe and int(probe.strip().split(",")[-1]) <= 720

    seen_audio.clear()
    asr = {"strategy": {"name": "fake_transport", "model": "fake/t", "device": "cpu"}, "cache_dir": str(tmp / "asr")}
    c3 = build_chat_client(model(audio_mode="asr_text", asr=asr, video_transport="base64",
                                 transcode={"threshold_mb": 0.01, "cache_dir": str(tmp / "tc2"), "threads": 2}))
    c3.complete("Q", video_path=mp4)
    assert len(seen_audio) == 1                                    # demuxed from the original container
    cached = [f.read_text() for f in (tmp / "asr").rglob("*.json")]
    assert len(cached) == 1 and str(mp4.resolve()) in cached[0]   # ASR cache keyed on the original
    sent = base64.b64decode(video_url_of(openai_stub.last).split(",", 1)[1])
    assert sent == outs[0].read_bytes() and sent != mp4.read_bytes()


def test_inference_block_equals_legacy_keys(openai_stub, clips):
    tmp, _, webm = clips
    big = {"threshold_mb": 1000, "cache_dir": str(tmp / "tc")}
    legacy = build_chat_client(model(audio_mode="none", video_transport="base64", transcode=big, strip_mm_kwargs=True))
    new = build_chat_client(model(inference={"audio": "none", "transport": "base64", "strip_mm_kwargs": True},
                                  transcode=big))
    assert legacy.pipeline.settings == new.pipeline.settings
    legacy.complete("Q", video_path=webm, extra_body=FLAGS)
    new.complete("Q", video_path=webm, extra_body=FLAGS)
    assert openai_stub.calls[-1] == openai_stub.calls[-2]
    with pytest.raises(ValueError, match="conflicts"):
        build_chat_client(model(inference={"transport": "file"}, video_transport="base64"))
    with pytest.raises(ValueError, match="unknown inference key"):
        build_chat_client(model(inference={"transprot": "file"}))


def test_resume_retries_error_rows_and_moves_them_aside(tmp_path):
    rec = tmp_path / "records.jsonl"
    append_jsonl(rec, {"question_id": 1, "response": "A"})
    append_jsonl(rec, {"question_id": 2, "response": "", "error": "HTTP 400"})
    ok, done = load_resumable_records(rec, lambda r: str(r["question_id"]))
    assert done == {"1"} and len(ok) == 1
    assert [r["question_id"] for r in read_jsonl_records(rec)] == [1]
    assert [r["question_id"] for r in read_jsonl_records(tmp_path / "records.errors.jsonl")] == [2]
