"""Video-MME subtitles / audio inputs (defaults off) and their helpers."""

from __future__ import annotations

import shutil
import subprocess
import threading
import wave
from concurrent.futures import ThreadPoolExecutor

import pytest

from omni_bench.asr import audio as audio_mod

from omni_bench.adapters.videomme import VideoMMEAdapter
from omni_bench.asr.audio import ensure_wav, extract_wav, has_audio_stream, media_duration_s
from omni_bench.subtitles import frame_times, parse_srt, resolve_srt, subtitles_for_frames

SRT = """1
00:00:00,000 --> 00:00:04,000
<font>hello there</font>

2
00:00:03,000 --> 00:00:08,500
general
kenobi

3
00:00:20,000 --> 00:00:22,000
much later
"""

ITEM = {"question": "What is shown?", "options": ["A. a", "B. b", "C. c", "D. d"]}


def test_parse_srt_strips_markup_and_joins_lines(tmp_path):
    srt = tmp_path / "vid.srt"
    srt.write_text(SRT)
    cues = parse_srt(srt)
    assert [c.text for c in cues] == ["hello there", "general kenobi", "much later"]
    assert cues[1].start_s == 3.0 and cues[1].end_s == 8.5
    assert resolve_srt(tmp_path, "vid") == srt and resolve_srt(tmp_path, "missing") is None


def test_only_cues_covering_sampled_frames_are_used(tmp_path):
    srt = tmp_path / "v.srt"
    srt.write_text(SRT)
    cues = parse_srt(srt)
    # 30 s video, 300 frames: indices 0, 50, 100 -> 0 s, 5 s, 10 s
    times = frame_times([0, 50, 100], 300, 30.0)
    assert times == [0.0, 5.0, 10.0]
    # 10 s is covered by nothing -> nearest earlier cue (same as the previous frame, collapsed)
    assert subtitles_for_frames(cues, times) == "hello there\ngeneral kenobi"
    assert subtitles_for_frames(cues, times, max_chars=12) == "hello there"
    assert subtitles_for_frames([], times) == "" and frame_times([1], 0, 10.0) == []


def test_prompt_without_subtitles_is_the_frames_only_wording():
    prompt = VideoMMEAdapter._prompt(ITEM)
    assert prompt == (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        "What is shown?\nA. a\nB. b\nC. c\nD. d\nThe best answer is:"
    )
    with_subs = VideoMMEAdapter._prompt(ITEM, subtitles="hello there")
    assert with_subs == "This video's subtitles are listed below:\nhello there\n" + prompt


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_audio_is_demuxed_to_16k_mono_wav(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=duration=2:size=64x64:rate=5", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=2", "-shortest", str(clip)], check=True)
    silent = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=duration=1:size=64x64:rate=5", str(silent)], check=True)
    assert has_audio_stream(clip) and not has_audio_stream(silent)
    wav = extract_wav(clip, tmp_path / "out.wav")
    assert wav.exists() and abs((media_duration_s(wav) or 0) - 2.0) < 0.2


def test_only_consecutive_duplicates_are_collapsed(tmp_path):
    srt = tmp_path / "v.srt"
    srt.write_text(SRT)
    cues = parse_srt(srt)
    # hello (0 s, 1 s) -> kenobi (5 s) -> hello again (2 s, later in the frame list)
    assert subtitles_for_frames(cues, [0.0, 1.0, 5.0, 2.0]) == "hello there\ngeneral kenobi\nhello there"


def _make_clip(path):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=duration=3:size=64x64:rate=5", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=3", "-shortest", str(path)], check=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_concurrent_extraction_of_one_video_yields_one_complete_wav(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip)
    target = tmp_path / "cache" / "clip.wav"
    runs: list[int] = []
    real = audio_mod._run_extract

    def counting(*args, **kwargs):
        runs.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(audio_mod, "_run_extract", counting)
    barrier = threading.Barrier(8)

    def worker(_):
        barrier.wait()
        return ensure_wav(clip, target)

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(worker, range(8)))
    assert set(results) == {target} and len(runs) == 1           # one ffmpeg run for eight callers
    with wave.open(str(target)) as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000
        assert abs(w.getnframes() / w.getframerate() - 3.0) < 0.2
    assert sorted(p.name for p in target.parent.iterdir()) == ["clip.wav"]   # no temp files left


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_extract_wav_writes_atomically_even_when_raced(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip)
    target = tmp_path / "out" / "clip.wav"
    with ThreadPoolExecutor(6) as pool:                           # no lock: every caller runs ffmpeg
        list(pool.map(lambda _: extract_wav(clip, target), range(6)))
    with wave.open(str(target)) as w:
        assert abs(w.getnframes() / w.getframerate() - 3.0) < 0.2
    assert sorted(p.name for p in target.parent.iterdir()) == ["clip.wav"]


def test_failed_extraction_leaves_no_file(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(audio_mod.AudioExtractionError):
        extract_wav(bad, tmp_path / "out" / "bad.wav")
    assert list((tmp_path / "out").iterdir()) == []
