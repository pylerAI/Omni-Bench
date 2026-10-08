"""ASR package end to end with a fake strategy (no GPU, no Whisper)."""

from __future__ import annotations

from omni_bench.asr import (
    AsrSettings,
    BatchTranscribeCommand,
    TranscribeCommand,
    build_cache,
    build_requests,
    build_strategy,
    format_transcript,
    iter_media_files,
)
from omni_bench.asr.schema import Transcription, TranscriptionRequest, TranscriptionSegment
from omni_bench.asr.strategies import SttStrategy, available_strategies, register_strategy

calls: list = []


@register_strategy("fake")
class FakeStrategy(SttStrategy):
    def load(self):
        calls.append("load")

    def _transcribe(self, audio_path, request):
        calls.append(("transcribe", audio_path.name))
        return (
            [TranscriptionSegment(0, 0.0, 3.5, "hello there"), TranscriptionSegment(1, 3.5, 7.0, "second line")],
            {"language": "en", "language_probability": 0.99, "duration_s": 7.0},
        )


def test_registry_lists_the_builtin_engines():
    names = available_strategies()
    assert {"faster_whisper", "transformers_whisper", "vllm_asr", "nemo_streaming"} <= set(names)


def test_transcribe_cache_format_and_batch(media):
    calls.clear()
    settings = AsrSettings.from_dict({"strategy": {"name": "fake", "model": "fake/v1", "device": "cpu"},
                                      "cache_dir": str(media.dir / "cache")})
    strategy = build_strategy(settings.strategy)
    cache = build_cache(settings, strategy)
    assert cache.namespace == "fake__fake__v1"
    cmd = TranscribeCommand(strategy, cache)

    # wav passes through untouched; the engine loads lazily
    t1 = cmd.execute(TranscriptionRequest(media_path=media.wav))
    assert "load" in calls and ("transcribe", "clip.wav") in calls
    assert t1.segments[0].text == "hello there"

    # cache hit -> no second transcription
    before = len(calls)
    t2 = cmd.execute(TranscriptionRequest(media_path=media.wav))
    assert len(calls) == before
    assert t2.text == t1.text and t2.media.key == t1.media.key

    # video is demuxed to a temp wav
    t3 = cmd.execute(TranscriptionRequest(media_path=media.mp4))
    assert t3.media.key != t1.media.key

    # video without an audio track -> valid empty transcript, engine never runs
    before = len(calls)
    t4 = cmd.execute(TranscriptionRequest(media_path=media.silent))
    assert len(calls) == before
    assert not t4.has_speech and t4.params["empty_reason"] == "no_audio_stream"

    # JSON round trip
    restored = Transcription.from_dict(t1.to_dict())
    assert restored.text == t1.text and restored.segments[1].start_s == 3.5

    # prompt formatting
    block = format_transcript(t1)
    assert "[00:00] hello there" in block and "[00:03] second line" in block
    assert format_transcript(t4) == "Audio transcript: (no speech detected)"
    assert "truncated" in format_transcript(t1, max_chars=10)

    # batch: three cached, one missing file fails without stopping the batch
    report = BatchTranscribeCommand(cmd, workers=4).execute(
        build_requests([media.wav, media.mp4, media.silent, media.dir / "missing.wav"])
    )
    assert report.total == 4 and report.cached == 3 and report.failed == 1

    found = iter_media_files(media.dir)
    assert media.wav in found and media.mp4 in found

    # settings round trip
    rt = AsrSettings.from_dict(settings.to_dict())
    assert rt.strategy.name == "fake" and rt.cache_dir == settings.cache_dir
