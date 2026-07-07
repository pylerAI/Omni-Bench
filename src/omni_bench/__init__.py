"""Unified benchmark runner for omni-modal model evaluation."""

import os

# Frame sampling seeks to arbitrary positions (cap.set(POS_FRAMES)), which makes
# libav's H.264 decoder log recoverable "mmco: unref short failure" warnings. The
# returned frames are still usable, so quiet the ffmpeg log to fatal-only (our code
# checks VideoCapture return values rather than relying on these messages). Set
# before any cv2.VideoCapture is created; overridable via the environment.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

__version__ = "0.1.0"
