from __future__ import annotations

from omni_bench.adapters.base import BenchmarkAdapter


ADAPTER_NAMES = (
    "av_speakerbench",
    "mlvu",
    "omnivideobench",
    "videomme",
    "worldsense",
)


def get_adapter(name: str) -> BenchmarkAdapter:
    if name == "av_speakerbench":
        from omni_bench.adapters.av_speakerbench import AVSpeakerBenchAdapter

        return AVSpeakerBenchAdapter()
    if name == "mlvu":
        from omni_bench.adapters.mlvu import MLVUAdapter

        return MLVUAdapter()
    if name == "omnivideobench":
        from omni_bench.adapters.omnivideobench import OmniVideoBenchAdapter

        return OmniVideoBenchAdapter()
    if name == "videomme":
        from omni_bench.adapters.videomme import VideoMMEAdapter

        return VideoMMEAdapter()
    if name == "worldsense":
        from omni_bench.adapters.worldsense import WorldSenseAdapter

        return WorldSenseAdapter()
    raise ValueError(f"Unsupported benchmark '{name}'. Known: {sorted(ADAPTER_NAMES)}")


ADAPTERS: dict[str, None] = {
    name: None for name in ADAPTER_NAMES
}
