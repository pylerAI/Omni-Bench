"""Transport axis: how a local ``video_path`` reaches the server.

``file`` — ``file://`` URL; needs ``--allowed-local-media-path`` on the server.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from omni_bench.inference.base import BuildContext, Registry, file_url


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
