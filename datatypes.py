"""Shared data structures (no heavy dependencies)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

WRONG_WAY = "wrong_way"
PROLONGED_STOP = "prolonged_stop"
PROXIMITY = "collision_like_proximity"
CONGESTION = "congestion"

EVENT_TYPES = (WRONG_WAY, PROLONGED_STOP, PROXIMITY, CONGESTION)

EVENT_LABELS = {
    WRONG_WAY: "Wrong-way movement",
    PROLONGED_STOP: "Prolonged stop",
    PROXIMITY: "Collision-like proximity",
    CONGESTION: "Unusual congestion",
}

REVIEW_STATUSES = ("unreviewed", "reviewed", "false_positive")


@dataclass(frozen=True)
class Detection:
    """One detected vehicle in one frame. ``track_id`` is -1 when untracked."""

    track_id: int
    label: str
    confidence: Optional[float]  # None for simulated (demo) detections
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in processed-frame pixels

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class AnomalyEvent:
    """A rule-generated *candidate* anomaly that needs human review."""

    event_type: str
    video_time_s: float
    frame_index: int
    track_ids: Tuple[int, ...]
    confidence: Optional[float]  # detector confidence of the involved vehicles, NOT an event probability
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str] = None


@dataclass
class FrameStats:
    frame_index: int
    video_time_s: float
    vehicle_count: int
    tracked_count: int
    occupancy: float  # sum of vehicle box areas / frame area (0..1), a rough density proxy


@dataclass
class FramePacket:
    index: int  # 1-based frame number in the source video
    t: float  # seconds from start of video
    image: Any  # numpy BGR image
