"""Exercise the ASR package end-to-end with a fake strategy (no GPU, no Whisper)."""
import subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from omni_bench.asr import (
    AsrSettings, BatchTranscribeCommand, TranscribeCommand, TranscriptCache,
    build_cache, build_requests, build_strategy, format_transcript, iter_media_files,
)
from omni_bench.asr.schema import Transcription, TranscriptionRequest, TranscriptionSegment
from omni_bench.asr.strategies import SttStrategy, StrategySpec, available_strategies, register_strategy

calls = []

@register_strategy("fake")
class FakeStrategy(SttStrategy):
    def load(self): calls.append("load")
    def _transcribe(self, audio_path, request):
        calls.append(("transcribe", audio_path.name))
        return (
            [TranscriptionSegment(0, 0.0, 3.5, "hello there"),
             TranscriptionSegment(1, 3.5, 7.0, "second line")],
            {"language": "en", "language_probability": 0.99, "duration_s": 7.0},
        )

print("registry:", available_strategies())
assert "faster_whisper" in available_strategies()
assert "transformers_whisper" in available_strategies()

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # a real 2s tone wav so ffprobe/audio_source take the real path
    wav = tmp / "clip.wav"
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","lavfi",
                    "-i","sine=frequency=440:duration=2","-ar","16000","-ac","1",str(wav)], check=True)
    mp4 = tmp / "clip_video.mp4"
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","lavfi","-i",
                    "testsrc=duration=2:size=64x64:rate=5","-f","lavfi","-i",
                    "sine=frequency=440:duration=2","-shortest",str(mp4)], check=True)
    silent = tmp / "silent.mp4"
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","lavfi","-i",
                    "testsrc=duration=1:size=64x64:rate=5",str(silent)], check=True)

    settings = AsrSettings.from_dict({
        "strategy": {"name": "fake", "model": "fake/v1", "device": "cpu"},
        "cache_dir": str(tmp / "cache"),
    })
    strategy = build_strategy(settings.strategy)
    cache = build_cache(settings, strategy)
    print("namespace:", cache.namespace)
    assert cache.namespace == "fake__fake__v1"

    cmd = TranscribeCommand(strategy, cache)

    # 1. wav passes through untouched
    t1 = cmd.execute(TranscriptionRequest(media_path=wav))
    assert [c for c in calls if c == "load"], "strategy must load lazily"
    assert t1.segments[0].text == "hello there"
    assert ("transcribe", "clip.wav") in calls

    # 2. cache hit -> no second transcription
    before = len(calls)
    t2 = cmd.execute(TranscriptionRequest(media_path=wav))
    assert len(calls) == before, "second call must be served from cache"
    assert t2.text == t1.text and t2.media.key == t1.media.key

    # 3. video is demuxed to a temp wav
    t3 = cmd.execute(TranscriptionRequest(media_path=mp4))
    assert t3.media.key != t1.media.key
    assert any(c[0] == "transcribe" for c in calls if isinstance(c, tuple))

    # 4. video with no audio track -> valid empty transcript, engine never runs
    before = len(calls)
    t4 = cmd.execute(TranscriptionRequest(media_path=silent))
    assert len(calls) == before, "silent video must not reach the engine"
    assert not t4.has_speech and t4.params["empty_reason"] == "no_audio_stream"

    # 5. JSON round-trip
    restored = Transcription.from_dict(t1.to_dict())
    assert restored.text == t1.text and restored.segments[1].start_s == 3.5

    # 6. prompt formatting
    block = format_transcript(t1)
    print("--- block ---"); print(block); print("-------------")
    assert "[00:00] hello there" in block and "[00:03] second line" in block
    assert format_transcript(t4) == "Audio transcript: (no speech detected)"
    assert "truncated" in format_transcript(t1, max_chars=10)

    # 7. batch command
    report = BatchTranscribeCommand(cmd, workers=4).execute(
        build_requests([wav, mp4, silent, tmp / "missing.wav"])
    )
    print("batch:", report.to_dict())
    assert report.total == 4 and report.cached == 3 and report.failed == 1

    # 8. media discovery
    found = iter_media_files(tmp)
    assert wav in found and mp4 in found

    # 9. settings round-trip keeps unknown engine options
    rt = AsrSettings.from_dict(settings.to_dict())
    assert rt.strategy.name == "fake" and rt.cache_dir == settings.cache_dir

print("\nALL ASR SMOKE TESTS PASSED")
