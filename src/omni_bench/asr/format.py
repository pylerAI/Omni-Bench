"""Render a :class:`Transcription` into the text block injected into prompts."""

from __future__ import annotations

from omni_bench.asr.schema import Transcription

DEFAULT_HEADER = "Audio transcript of the media (speech recognised automatically):"
DEFAULT_EMPTY_TEXT = "Audio transcript: (no speech detected)"


def format_timestamp(seconds: float) -> str:
    # Floor, so a segment's stamp never lands after the speech actually starts.
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_transcript(
    transcription: Transcription,
    *,
    header: str = DEFAULT_HEADER,
    empty_text: str = DEFAULT_EMPTY_TEXT,
    with_timestamps: bool = True,
    max_chars: int | None = None,
    max_end_s: float | None = None,
) -> str:
    """Prompt-ready transcript block.

    ``max_chars`` keeps the tail of the transcript, which is where the answer
    usually lives for long clips, and marks the truncation explicitly.

    ``max_end_s`` drops segments starting past that timestamp. Needed when the
    comparison target hears only part of the audio: Qwen3-Omni's feature
    extractor stops at 300s (``n_samples`` 4,800,000 at 16 kHz), so on a
    40-minute video an untruncated transcript would hand the cascade 8x the
    audio its baseline receives.
    """
    if not transcription.has_speech:
        return empty_text

    if max_end_s is not None:
        kept = [s for s in transcription.segments if s.start_s < max_end_s]
        if not kept:
            return empty_text
        transcription = Transcription(
            media=transcription.media,
            backend=transcription.backend,
            model=transcription.model,
            segments=kept,
            language=transcription.language,
            language_probability=transcription.language_probability,
            duration_s=transcription.duration_s,
            elapsed_s=transcription.elapsed_s,
            params=transcription.params,
        )

    if with_timestamps:
        lines = [
            f"[{format_timestamp(segment.start_s)}] {segment.text.strip()}"
            for segment in transcription.segments
            if segment.text.strip()
        ]
        body = "\n".join(lines)
    else:
        body = transcription.text

    if max_chars is not None and len(body) > max_chars:
        body = "...(truncated)...\n" + body[-max_chars:]

    return f"{header}\n{body}"
