"""Video-MME subtitles / audio inputs (defaults off) and their helpers."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from omni_bench.adapters.videomme import VideoMMEAdapter
from omni_bench.asr.audio import extract_wav, has_audio_stream, media_duration_s
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
    # 10 s is covered by nothing -> nearest earlier cue (duplicate, dropped)
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
