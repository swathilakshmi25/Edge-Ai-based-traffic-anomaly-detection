"""Video validation and frame reading (OpenCV).

OpenCV is imported lazily so that the rest of the package (rules, database, tests) works
without it. Video resources are always released, also on errors or early stops.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union

from .config import VideoConfig
from .datatypes import FramePacket


class VideoValidationError(Exception):
    """The file is missing, unsupported, too large or cannot be decoded."""


def _cv2():
    try:
        import cv2  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise VideoValidationError("OpenCV is not installed. Run: pip install -r requirements.txt") from exc
    return cv2


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    fps_assumed: bool
    frame_count: int
    duration_s: Optional[float]
    size_mb: float


def _valid_fps(fps) -> bool:
    return fps is not None and math.isfinite(fps) and 1.0 <= fps <= 240.0


def validate_video_file(path, cfg: VideoConfig) -> VideoInfo:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise VideoValidationError(f"Video file not found: {p}")
    ext = p.suffix.lower().lstrip(".")
    allowed = [e.lower().lstrip(".") for e in cfg.allowed_extensions]
    if ext not in allowed:
        raise VideoValidationError(f"Unsupported file type '.{ext}'. Allowed: {', '.join(allowed)}.")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb <= 0:
        raise VideoValidationError("The video file is empty.")
    if size_mb > cfg.max_upload_mb:
        raise VideoValidationError(f"File is {size_mb:.0f} MB; the configured limit is {cfg.max_upload_mb} MB.")
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoValidationError("OpenCV could not open this video (corrupt file or unsupported codec).")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoValidationError("The video opened but no frame could be decoded.")
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        assumed = not _valid_fps(fps)
        if assumed:
            fps = cfg.default_fps
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        n = n if n > 0 else 0
        return VideoInfo(str(p), w, h, float(fps), assumed, n, (n / fps) if n else None, size_mb)
    finally:
        cap.release()


class VideoSource:
    """Iterable of :class:`FramePacket`. Use ``close()`` (or exhaust it) to release the capture."""

    def __init__(self, source: Union[str, int], cfg: VideoConfig, max_seconds: Optional[float] = None):
        self.source = source
        self.cfg = cfg
        self.is_camera = isinstance(source, int)
        self.max_seconds = max_seconds if max_seconds is not None else (cfg.max_seconds or 0)
        self.warnings: List[str] = []
        self.total_frames = 0
        self.fps = cfg.default_fps
        self._stop = False
        self._cap = None

    def close(self) -> None:
        self._stop = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[FramePacket]:
        cv2 = _cv2()
        if self.is_camera and not self.max_seconds:
            raise VideoValidationError("Camera input needs a maximum duration (video.max_seconds).")
        cap = cv2.VideoCapture(self.source)
        self._cap = cap
        try:
            if not cap.isOpened():
                raise VideoValidationError("Could not open the video source.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not _valid_fps(fps):
                if not self.is_camera:
                    self.warnings.append(f"Video reports no valid FPS; assuming {self.cfg.default_fps:g}. "
                                         "Timestamps (and so durations) may be inaccurate.")
                fps = self.cfg.default_fps
            self.fps = float(fps)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not self.is_camera else 0
            stride = max(1, int(self.cfg.frame_stride))
            started = time.monotonic()
            frame_no = 0
            while not self._stop:
                if (frame_no % stride) != 0:
                    if not cap.grab():  # skip decoding of unused frames
                        break
                    frame_no += 1
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    break  # end of video (or camera lost)
                frame_no += 1
                t = (time.monotonic() - started) if self.is_camera else frame_no / self.fps
                if self.max_seconds and t > self.max_seconds:
                    break
                h, w = frame.shape[:2]
                rw = int(self.cfg.resize_width or 0)
                if rw and w > rw:
                    frame = cv2.resize(frame, (rw, max(1, int(round(h * rw / w)))), interpolation=cv2.INTER_AREA)
                yield FramePacket(index=frame_no, t=t, image=frame)
        finally:
            self.close()
