"""video_transport / strip_mm_kwargs / reasoning capture / error-retry resume (no server)."""
import base64, subprocess, sys, tempfile, threading, types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

sent = []

class _Completions:
    def create(self, **kwargs):
        sent.append(kwargs)
        msg = types.SimpleNamespace(content="B", reasoning="thinking...", model_extra={})
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        choice = types.SimpleNamespace(message=msg, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice], usage=usage)

class _OpenAI:
    def __init__(self, **kwargs): self.chat = types.SimpleNamespace(completions=_Completions())

openai_stub = types.ModuleType("openai")
openai_stub.OpenAI = _OpenAI
sys.modules["openai"] = openai_stub

from omni_bench import video_transport as vt
from omni_bench.asr.schema import TranscriptionSegment
from omni_bench.asr.strategies import SttStrategy, register_strategy
from omni_bench.client import build_chat_client
from omni_bench.config import ModelConfig
from omni_bench.io import append_jsonl, load_resumable_records, read_jsonl_records

seen_audio = []

@register_strategy("fake_transport")
class FakeStrategy(SttStrategy):
    def _transcribe(self, audio_path, request):
        seen_audio.append(Path(audio_path))
        return [TranscriptionSegment(0, 0.0, 1.0, "hello")], {"language": "en"}

def video_part(call):
    return [c for c in call["messages"][-1]["content"] if c["type"] == "video_url"][0]["video_url"]["url"]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    mp4 = tmp / "clip.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=duration=2:size=1920x1080:rate=10", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=2", "-shortest", "-c:v", "libx264", "-crf", "0",
                    str(mp4)], check=True)
    webm = tmp / "clip.webm"
    webm.write_bytes(b"fake-webm")

    def model(**extra):
        return ModelConfig(name="m", weight_path="/x", base_url="http://x/v1", extra=extra)

    # --- default transport stays file:// ---
    sent.clear()
    build_chat_client(model(audio_mode="none")).complete("Q", video_path=mp4)
    assert video_part(sent[-1]).startswith("file://"), video_part(sent[-1])[:40]

    # --- base64 below threshold: original bytes, mime by extension ---
    big = {"threshold_mb": 1000, "cache_dir": str(tmp / "tc")}
    c = build_chat_client(model(audio_mode="none", video_transport="base64", transcode=big))
    c.complete("Q", video_path=mp4)
    url = video_part(sent[-1])
    assert url.startswith("data:video/mp4;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == mp4.read_bytes()
    c.complete("Q", video_path=webm)
    assert video_part(sent[-1]).startswith("data:video/webm;base64,")
    print("base64 transport OK")

    # --- strip_mm_kwargs removes keys entirely (NoAudio would only set False) ---
    flags = {"mm_processor_kwargs": {"use_audio_in_video": True}, "media_io_kwargs": {"video": {"fps": 2}}}
    c.complete("Q", video_path=webm, extra_body=flags)
    assert sent[-1]["extra_body"]["mm_processor_kwargs"]["use_audio_in_video"] is False
    c2 = build_chat_client(model(audio_mode="none", video_transport="base64", transcode=big,
                                 strip_mm_kwargs=True, extra_body={"chat_template_kwargs": {"enable_thinking": True}}))
    res = c2.complete("Q", video_path=webm, extra_body=flags)
    body = sent[-1]["extra_body"]
    assert "mm_processor_kwargs" not in body and "media_io_kwargs" not in body, body
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert res.reasoning == "thinking..." and res.finish_reason == "stop"
    assert res.meta()["completion_tokens"] == 5
    print("strip_mm_kwargs + reasoning OK")

    # --- above threshold: concurrent callers share one transcode ---
    small = vt.TranscodeSettings.from_dict({"threshold_mb": 0.01, "cache_dir": str(tmp / "tc2"), "threads": 2})
    calls = []
    real = vt._run_ffmpeg
    def counting(*a, **k):
        calls.append(threading.get_ident()); return real(*a, **k)
    vt._run_ffmpeg = counting
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda _: vt.ensure_transcoded(mp4, small), range(8)))
    vt._run_ffmpeg = real
    assert len(set(outs)) == 1 and outs[0].exists()
    assert len({c for c in calls}) == 1, f"transcoded by {len(set(calls))} threads"
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,height",
                            "-of", "csv=p=0", str(outs[0])], capture_output=True, text=True).stdout
    assert "audio" not in probe and int(probe.strip().split(",")[-1]) <= 720, probe
    print(f"transcode OK ({len(calls)} ffmpeg passes, ladder until ≤ threshold):", probe.strip())

    # --- ASR still reads the ORIGINAL path, transport sends the transcode ---
    sent.clear(); seen_audio.clear()
    asr = {"strategy": {"name": "fake_transport", "model": "fake/t", "device": "cpu"},
           "cache_dir": str(tmp / "asr")}
    c3 = build_chat_client(model(audio_mode="asr_text", asr=asr, video_transport="base64",
                                 transcode={"threshold_mb": 0.01, "cache_dir": str(tmp / "tc2"), "threads": 2}))
    c3.complete("Q", video_path=mp4)
    assert len(seen_audio) == 1  # demuxed from the original container
    cached = [f.read_text() for f in (tmp / "asr").rglob("*.json")]
    assert len(cached) == 1 and str(mp4.resolve()) in cached[0], "ASR cache not keyed on original"
    sent_bytes = base64.b64decode(video_part(sent[-1]).split(",", 1)[1])
    assert sent_bytes == outs[0].read_bytes() and sent_bytes != mp4.read_bytes()
    print("ASR uses original path; request carries transcode OK")

    # --- inference: block == legacy flat keys ---
    sent.clear()
    legacy = build_chat_client(model(audio_mode="none", video_transport="base64", transcode=big,
                                     strip_mm_kwargs=True))
    new = build_chat_client(model(inference={"audio": "none", "transport": "base64",
                                             "strip_mm_kwargs": True}, transcode=big))
    assert legacy.pipeline.settings == new.pipeline.settings, (legacy.pipeline.settings, new.pipeline.settings)
    legacy.complete("Q", video_path=webm, extra_body=flags)
    new.complete("Q", video_path=webm, extra_body=flags)
    assert sent[-1] == sent[-2]
    try:
        build_chat_client(model(inference={"transport": "file"}, video_transport="base64"))
        raise SystemExit("conflict must raise")
    except ValueError as exc:
        print("legacy/new conflict rejected:", str(exc)[:70])
    try:
        build_chat_client(model(inference={"transprot": "file"}))
        raise SystemExit("typo must raise")
    except ValueError as exc:
        print("inference typo rejected:", str(exc)[:70])
    print("inference block == legacy keys OK")

    # --- resume: error records are retried, moved aside ---
    rec = tmp / "records.jsonl"
    append_jsonl(rec, {"question_id": 1, "response": "A"})
    append_jsonl(rec, {"question_id": 2, "response": "", "error": "HTTP 400"})
    ok, done = load_resumable_records(rec, lambda r: str(r["question_id"]))
    assert done == {"1"} and len(ok) == 1
    assert [r["question_id"] for r in read_jsonl_records(rec)] == [1]
    assert [r["question_id"] for r in read_jsonl_records(tmp / "records.errors.jsonl")] == [2]
    print("resume retry OK")

print("\nALL TRANSPORT SMOKE TESTS PASSED")
