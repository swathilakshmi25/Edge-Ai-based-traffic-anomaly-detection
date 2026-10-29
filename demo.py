"""SIMULATED demo scenario - *not* AI inference.

A scripted scene (synthetic vehicles on a blank road) is fed through the same tracking, rule,
database and dashboard code as real video. It exists so the full pipeline can be demonstrated and
tested without a traffic video or model. Everything produced here is stored with mode='demo' and
is labelled "SIMULATED" in the interface. It says nothing about detector accuracy.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np

from .datatypes import Detection, FramePacket

FRAME_W, FRAME_H = 960, 540
FPS = 10.0
DURATION_S = 70.0
BOX_W, BOX_H = 100, 50


def _box(cx: float, cy: float):
    return (cx - BOX_W / 2, cy - BOX_H / 2, cx + BOX_W / 2, cy + BOX_H / 2)


def scripted_vehicles(t: float):
    """Return [(track_id, label, cx, cy)] for the scripted scene at video time ``t`` seconds."""
    out = []
    # Normal flow, left -> right (expected direction "right"), two lanes.
    for k in range(0, 18):  # one vehicle every 4 s
        for lane_y, offset in ((300, 0.0), (380, 2.0)):
            t0 = 4.0 * k + offset
            x = -60 + 120.0 * (t - t0)
            if t >= t0 and -60 <= x <= FRAME_W + 60:
                out.append((1000 + 2 * k + (1 if offset else 0), "car", x, lane_y))
    # Wrong-way vehicle: right -> left, t = 5..14 s
    if 5.0 <= t <= 14.5:
        out.append((100, "car", 900 - 100.0 * (t - 5.0), 450))
    # Vehicle that drives in, stops at x=550 from t=8 s to t=40 s, then leaves
    if 2.0 <= t < 8.0:
        out.append((200, "truck", -50 + 100.0 * (t - 2.0), 220))
    elif 8.0 <= t < 40.0:
        out.append((200, "truck", 550.0, 220))
    elif 40.0 <= t <= 52.0:
        out.append((200, "truck", 550 + 100.0 * (t - 40.0), 220))
    # Two vehicles that converge and stay very close, t = 20..30 s
    if 20.0 <= t <= 30.0:
        gap = max(30.0, 100.0 - 25.0 * (t - 20.0))
        x = -40 + 110.0 * (t - 20.0) + 100
        out.append((300, "car", x, 90.0))
        out.append((301, "car", x + 5.0, 90.0 + gap))
    # Congestion: 14 slow vehicles in two rows, t = 50..70 s
    if t >= 50.0:
        for k in range(7):
            for row, y in enumerate((70.0, 500.0)):
                out.append((400 + 2 * k + row, "car", 80 + 120.0 * k + 10.0 * (t - 50.0), y))
    return out


class SimulatedDetector:
    """Looks like a detector (``detect(packet)``) but only replays the scripted scene."""

    is_simulated = True
    model_label = "SIMULATED (scripted scene, not a model)"

    def __init__(self, seed: int = 7):
        self.seed = seed

    def new_session(self) -> None:
        pass

    def detect(self, packet: FramePacket) -> List[Detection]:
        rng = np.random.RandomState(self.seed + packet.index)  # tiny deterministic jitter, like a real detector
        dets = []
        for tid, label, cx, cy in scripted_vehicles(packet.t):
            jx, jy = rng.uniform(-1.5, 1.5, size=2)
            if not (0 <= cx + jx <= FRAME_W):
                continue
            dets.append(Detection(track_id=tid, label=label, confidence=None, bbox=_box(cx + jx, cy + jy)))
        return dets


class SimulatedSource:
    """Yields blank 'road' frames at 10 FPS for 70 s."""

    def __init__(self, duration_s: float = DURATION_S):
        self.duration_s = duration_s
        self.fps = FPS
        self.total_frames = int(duration_s * FPS)
        self.warnings: List[str] = ["Simulated scene: no real video is being analysed."]
        self._stop = False
        self._background = self._make_background()

    @staticmethod
    def _make_background() -> np.ndarray:
        img = np.full((FRAME_H, FRAME_W, 3), 45, dtype=np.uint8)
        for y in (90, 220, 300, 380, 450):
            img[y - 30:y + 30, :, :] = 70
        return img

    def close(self) -> None:
        self._stop = True

    def __iter__(self) -> Iterator[FramePacket]:
        for i in range(1, self.total_frames + 1):
            if self._stop:
                break
            yield FramePacket(index=i, t=i / self.fps, image=self._background)
