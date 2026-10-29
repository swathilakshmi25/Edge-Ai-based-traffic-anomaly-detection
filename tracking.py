"""Per-vehicle track history built from tracker IDs.

Association across frames is done by the tracker (Ultralytics ByteTrack in the real pipeline).
This module only keeps the *history* needed by the anomaly rules: positions over time,
heading, age and (optionally) dwell time inside a configured zone.

Known limitations: occlusion, missed detections and ID switches break a vehicle's history;
pixel movement is not a calibrated physical speed.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .datatypes import Detection
from .utils import point_in_polygon


class TrackState:
    def __init__(self, track_id: int, label: str, t: float, history_len: int):
        self.track_id = track_id
        self.label = label
        self.first_t = t
        self.last_t = t
        self.last_conf: Optional[float] = None
        self.last_bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
        self.points: Deque[Tuple[float, float, float]] = deque(maxlen=history_len)  # (t, cx, cy)
        self.zone_time_s = 0.0  # accumulated time with centre inside the configured zone
        self.hits = 0

    @property
    def age_s(self) -> float:
        return self.last_t - self.first_t

    @property
    def position(self) -> Tuple[float, float]:
        _, x, y = self.points[-1]
        return x, y

    def net_displacement(self, n: Optional[int] = None) -> Tuple[float, float]:
        pts = list(self.points)[-n:] if n else list(self.points)
        if len(pts) < 2:
            return 0.0, 0.0
        return pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]

    def heading_deg(self, min_px: float = 5.0) -> Optional[float]:
        """Approximate direction of travel in image coordinates: 0=right, 90=down, 180=left, 270=up."""
        dx, dy = self.net_displacement()
        if math.hypot(dx, dy) < min_px:
            return None
        return math.degrees(math.atan2(dy, dx)) % 360.0


class TrackStore:
    def __init__(self, history_len: int = 60, max_gap_s: float = 2.0):
        self.history_len = history_len
        self.max_gap_s = max_gap_s
        self.tracks: Dict[int, TrackState] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, t: float, detections: Sequence[Detection],
               zone_px: Optional[Sequence[Tuple[float, float]]] = None) -> Dict[int, TrackState]:
        for det in detections:
            if det.track_id < 0:
                continue  # untracked detections are counted elsewhere but have no history
            cx, cy = det.center
            tr = self.tracks.get(det.track_id)
            if tr is None:
                tr = TrackState(det.track_id, det.label, t, self.history_len)
                self.tracks[det.track_id] = tr
                dt = 0.0
            else:
                dt = max(0.0, t - tr.last_t)
            tr.label = det.label
            tr.last_t = t
            tr.last_conf = det.confidence
            tr.last_bbox = det.bbox
            tr.points.append((t, cx, cy))
            tr.hits += 1
            if zone_px and point_in_polygon(cx, cy, zone_px):
                tr.zone_time_s += min(dt, self.max_gap_s)
        stale: List[int] = [tid for tid, tr in self.tracks.items() if t - tr.last_t > self.max_gap_s]
        for tid in stale:
            del self.tracks[tid]
        return self.tracks
