"""Official Video-MME subtitles, matched to the frames actually sampled.

The benchmark ships SRT files separately from the parquet, and its README is
specific about how to use them:

    With respect to the setting of adding subtitles, you should only use the
    subtitles corresponding to the sampled video frames. For example, if you
    extract 10 frames per video for evaluation, take the 10 subtitles that
    correspond to the time of those 10 frames.

So this picks the cue covering each sampled frame's timestamp rather than
dumping the whole track. The cues are auto-generated and roll forward with
heavy overlap — consecutive frames usually land on the same cue — so
duplicates are collapsed while order is preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TAG = re.compile(r"<[^>]+>")
_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


@dataclass(slots=True, frozen=True)
class Cue:
    start_s: float
    end_s: float
    text: str


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(path: str | Path) -> list[Cue]:
    """Cues from an SRT file, with markup stripped and blank cues dropped."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", raw):
        match = _TIME.search(block)
        if not match:
            continue
        start = _seconds(*match.group(1, 2, 3, 4))
        end = _seconds(*match.group(5, 6, 7, 8))
        body = block[match.end():]
        text = " ".join(_TAG.sub("", line).strip() for line in body.splitlines())
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            cues.append(Cue(start, end, text))
    return cues


def subtitles_for_frames(
    cues: list[Cue], frame_times_s: list[float], *, max_chars: int | None = None
) -> str:
    """Cue text covering each sampled frame, de-duplicated, order preserved."""
    if not cues or not frame_times_s:
        return ""
    picked: list[str] = []
    seen: set[str] = set()
    for t in frame_times_s:
        hit = next((c for c in cues if c.start_s <= t <= c.end_s), None)
        if hit is None:
            # Nothing covers this instant; fall back to the nearest earlier cue.
            earlier = [c for c in cues if c.end_s <= t]
            hit = earlier[-1] if earlier else None
        if hit is None or hit.text in seen:
            continue
        seen.add(hit.text)
        picked.append(hit.text)
    text = "\n".join(picked)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0]
    return text


def frame_times(indices: list[int], total_frames: int, duration_s: float) -> list[float]:
    """Frame indices to wall-clock seconds."""
    if total_frames <= 0 or duration_s <= 0:
        return []
    return [min(duration_s, i / total_frames * duration_s) for i in indices]


def resolve_srt(subtitle_dir: str | Path, video_id: str) -> Path | None:
    directory = Path(subtitle_dir)
    for name in (f"{video_id}.srt", f"{video_id}.SRT"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None
