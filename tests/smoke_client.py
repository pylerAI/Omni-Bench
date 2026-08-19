"""Exercise the audio-mode clients with a stubbed OpenAI SDK (no server needed)."""
import subprocess, sys, tempfile, types
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

# --- stub the openai SDK so client.py imports without the dependency ---------
sent = []

class _Completions:
    def create(self, **kwargs):
        sent.append(kwargs)
        msg = types.SimpleNamespace(content="A")
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=1, total_tokens=11)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)

class _OpenAI:
    def __init__(self, **kwargs): self.chat = types.SimpleNamespace(completions=_Completions())

openai_stub = types.ModuleType("openai")
openai_stub.OpenAI = _OpenAI
sys.modules["openai"] = openai_stub

from omni_bench.asr import AsrSettings, TranscribeCommand, build_cache, build_strategy
from omni_bench.asr.schema import TranscriptionSegment
from omni_bench.asr.strategies import SttStrategy, register_strategy
from omni_bench.asr_client import AsrTextChatClient, NoAudioChatClient, build_chat_client
from omni_bench.client import VllmChatClient
from omni_bench.config import ModelConfig

@register_strategy("fake2")
class FakeStrategy(SttStrategy):
    def _transcribe(self, audio_path, request):
        return [TranscriptionSegment(0, 1.0, 4.0, f"spoken words from {audio_path.suffix}")], {"language": "en"}

def text_of(call):
    return [c for c in call["messages"][-1]["content"] if c["type"] == "text"][0]["text"]

def parts(call):
    return [c["type"] for c in call["messages"][-1]["content"]]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    wav = tmp / "a.wav"
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","lavfi",
                    "-i","sine=frequency=440:duration=2","-ar","16000","-ac","1",str(wav)], check=True)
    mp4 = tmp / "b.mp4"
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","lavfi","-i",
                    "testsrc=duration=2:size=64x64:rate=5","-f","lavfi","-i",
                    "sine=frequency=440:duration=2","-shortest",str(mp4)], check=True)

    asr_raw = {"strategy": {"name": "fake2", "model": "fake/v2", "device": "cpu"},
               "cache_dir": str(tmp / "cache")}

    def model(mode, asr=None):
        extra = {"audio_mode": mode}
        if asr: extra["asr"] = asr
        return ModelConfig(name="m", weight_path="/x", base_url="http://x/v1", extra=extra)

    # --- factory dispatch ---
    assert type(build_chat_client(model("native"))) is VllmChatClient
    assert type(build_chat_client(model("none"))) is NoAudioChatClient
    assert type(build_chat_client(model("asr_text", asr_raw))) is AsrTextChatClient
    assert type(build_chat_client(ModelConfig(name="m", weight_path="/x"))) is VllmChatClient  # default
    try:
        build_chat_client(model("bogus")); raise SystemExit("should have raised")
    except ValueError as exc:
        print("unknown mode rejected:", exc)

    # --- audio_mode: none  (av_speakerbench-style video + omnidcbench flags) ---
    sent.clear()
    c = build_chat_client(model("none"))
    c.complete("QUESTION", video_path=mp4, audio_path=wav,
               extra_body={"mm_processor_kwargs": {"use_audio_in_video": True}})
    call = sent[-1]
    assert parts(call) == ["video_url", "text"], parts(call)          # audio dropped
    assert text_of(call) == "QUESTION"                                # prompt untouched
    assert call["extra_body"]["mm_processor_kwargs"]["use_audio_in_video"] is False
    print("none-mode OK:", parts(call))

    # --- audio_mode: asr_text, explicit audio file (worldsense/omnivideobench) ---
    sent.clear()
    c = build_chat_client(model("asr_text", asr_raw))
    c.complete("QUESTION", image_urls=["data:image/jpeg;base64,xx"], audio_path=wav)
    call = sent[-1]
    assert parts(call) == ["image_url", "text"], parts(call)
    body = text_of(call)
    assert body.endswith("\n\nQUESTION"), body                        # official prompt preserved verbatim
    assert "[00:01] spoken words from .wav" in body, body
    print("asr_text (audio file) OK")

    # --- audio_mode: asr_text, audio inside the video (av_speakerbench/omnidcbench) ---
    sent.clear()
    c.complete("QUESTION", video_path=mp4,
               extra_body={"mm_processor_kwargs": {"use_audio_in_video": True}})
    call = sent[-1]
    assert parts(call) == ["video_url", "text"]
    assert "spoken words from .wav" in text_of(call)                  # demuxed from the mp4
    assert call["extra_body"]["mm_processor_kwargs"]["use_audio_in_video"] is False
    print("asr_text (audio in video) OK")

    # --- Video-MME shape: frames only, no audio anywhere -> untouched ---
    sent.clear()
    c.complete("QUESTION", image_urls=["data:image/jpeg;base64,xx"])
    call = sent[-1]
    assert text_of(call) == "QUESTION", text_of(call)
    assert parts(call) == ["image_url", "text"]
    print("videomme passthrough OK")

    # --- strict_cache refuses to transcribe inline ---
    strict = dict(asr_raw, strict_cache=True, cache_dir=str(tmp / "empty-cache"))
    sc = build_chat_client(model("asr_text", strict))
    try:
        sc.complete("Q", audio_path=wav); raise SystemExit("should have raised")
    except FileNotFoundError as exc:
        print("strict_cache OK:", str(exc)[:60], "...")

print("\nALL CLIENT SMOKE TESTS PASSED")
