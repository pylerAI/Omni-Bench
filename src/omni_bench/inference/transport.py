"""Transport axis: how a local ``video_path`` reaches the server.

``file``   — ``file://`` URL; needs ``--allowed-local-media-path`` on the server.
``base64`` — inline data URL; files above ``transcode.threshold_mb`` are sent
             from the transcode cache (see :mod:`omni_bench.video_transport`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from omni_bench.inference.base import BuildContext, Registry, file_url
from omni_bench.video_transport import TranscodeSettings, video_data_url


class Transport(ABC):
    name: str

    @classmethod
    def create(cls, ctx: BuildContext) -> "Transport":
        return cls()

    @abstractmethod
    def video_url(self, path: str | Path) -> str:
        raise NotImplementedError


TRANSPORTS: Registry[Transport] = Registry("transport")


@TRANSPORTS.register("file")
class FileTransport(Transport):
    def video_url(self, path: str | Path) -> str:
        return file_url(path)


@TRANSPORTS.register("base64")
class Base64Transport(Transport):
    def __init__(self, transcode: TranscodeSettings) -> None:
        self.transcode = transcode

    @classmethod
    def create(cls, ctx: BuildContext) -> "Transport":
        # `transcode:` stays a top-level model key; scripts/pretranscode_videos.py reads it too.
        return cls(TranscodeSettings.from_dict(ctx.model.extra.get("transcode")))

    def video_url(self, path: str | Path) -> str:
        return video_data_url(path, self.transcode)
