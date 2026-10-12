"""Cache-only strategy for NVIDIA Nemotron streaming ASR (NeMo cache-aware streaming).

Transcripts are produced offline in a separate NeMo environment (riva_asr/) and
written into the standard transcript cache under
``nemo_streaming__<model slug>``. This strategy exists only so that namespace
resolves from a config; it never transcribes inline, so run with
``strict_cache: true``.
"""

from __future__ import annotations

from pathlib import Path

from omni_bench.asr.schema import TranscriptionRequest
from omni_bench.asr.strategies import SttStrategy, register_strategy


@register_strategy("nemo_streaming")
class NemoStreamingCacheStrategy(SttStrategy):
    """Read-only: transcripts must already exist in the cache."""

    def _transcribe(self, audio_path: Path, request: TranscriptionRequest):
        raise RuntimeError(
            f"nemo_streaming is cache-only (model {self.spec.model}); no cached transcript "
            f"for {request.media_path} under namespace '{self.cache_namespace()}'. "
            "Generate it with the NeMo pipeline in riva_asr/ and keep asr.strict_cache=true."
        )
