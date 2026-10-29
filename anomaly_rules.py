"""Configurable rule-based anomaly detection.

Every rule produces *candidate* events for human review. Rules use video timestamps, so they
behave consistently when the frame rate or the frame stride changes.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .config import AppConfig, CongestionConfig, ProximityConfig, StoppedConfig, WrongWayConfig, zone_to_pixels
from .datatypes import (CONGESTION, PROLONGED_STOP, PROXIMITY, WRONG_WAY, AnomalyEvent, Detection, FrameStats)
from .tracking import TrackState, TrackStore
from .utils import center_distance, iou, point_in_polygon

_DIR_VECTORS = {"right": (1.0, 0.0), "left": (-1.0, 0.0), "down": (0.0, 1.0), "up": (0.0, -1.0)}


@dataclass
class FrameContext:
    frame_index: int
    t: float
    detections: Sequence[Detection]
    tracks: Dict[int, TrackState]
    frame_w: int
    frame_h: int
    stats: FrameStats


def _cooldown_ok(last: Dict, key, t: float, cooldown_s: float) -> bool:
    prev = last.get(key)
    return prev is None or (t - prev) >= cooldown_s


class WrongWayRule:
    event_type = WRONG_WAY

    def __init__(self, cfg: WrongWayConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._streak: Dict[int, int] = {}
        self._last: Dict[int, float] = {}

    def evaluate(self, ctx: FrameContext) -> List[AnomalyEvent]:
        c = self.cfg
        if not c.enabled:
            return []
        ux, uy = _DIR_VECTORS[c.expected_direction]
        events: List[AnomalyEvent] = []
        for tid in [k for k in self._streak if k not in ctx.tracks]:
            self._streak.pop(tid, None)
        for tid, tr in ctx.tracks.items():
            if tr.last_t != ctx.t:
                continue  # not observed in this frame
            pts = list(tr.points)[-c.window_points:]
            if len(pts) < c.min_observations:
                self._streak[tid] = 0
                continue
            dx, dy = pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]
            against = -(dx * ux + dy * uy)  # positive = moving opposite to the expected direction
            total = math.hypot(dx, dy)
            # Require the movement to be mostly along the expected axis (ignores lane changes / turns).
            if against >= c.min_displacement_px and against >= 0.5 * total:
                self._streak[tid] = self._streak.get(tid, 0) + 1
            else:
                self._streak[tid] = 0
            if self._streak[tid] >= c.persistence_frames and _cooldown_ok(self._last, tid, ctx.t, c.cooldown_s):
                self._last[tid] = ctx.t
                span = pts[-1][0] - pts[0][0]
                events.append(AnomalyEvent(
                    event_type=WRONG_WAY, video_time_s=ctx.t, frame_index=ctx.frame_index,
                    track_ids=(tid,), confidence=tr.last_conf,
                    explanation=(f"Track {tid} ({tr.label}) moved {against:.0f}px against the configured "
                                 f"'{c.expected_direction}' direction over {len(pts)} observations "
                                 f"({span:.1f}s), sustained for {self._streak[tid]} consecutive frames."),
                    details={"expected_direction": c.expected_direction, "against_px": round(against, 1),
                             "observations": len(pts), "window_s": round(span, 2),
                             "heading_deg": tr.heading_deg()}))
        return events


class StoppedVehicleRule:
    event_type = PROLONGED_STOP

    def __init__(self, cfg: StoppedConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._state: Dict[int, dict] = {}
        self._last: Dict[int, float] = {}

    def evaluate(self, ctx: FrameContext) -> List[AnomalyEvent]:
        c = self.cfg
        if not c.enabled:
            return []
        zone = zone_to_pixels(c.zone, ctx.frame_w, ctx.frame_h) if c.zone else None
        events: List[AnomalyEvent] = []
        for tid in [k for k in self._state if k not in ctx.tracks]:
            self._state.pop(tid, None)
        for tid, tr in ctx.tracks.items():
            if tr.last_t != ctx.t:
                continue
            x, y = tr.position
            if zone and not point_in_polygon(x, y, zone):
                self._state.pop(tid, None)
                continue
            s = self._state.get(tid)
            # Anchor-based test: the vehicle counts as stationary while it stays within
            # move_threshold_px of where it first stopped. Small box jitter does not reset the timer.
            if s is None or math.hypot(x - s["anchor"][0], y - s["anchor"][1]) > c.move_threshold_px:
                self._state[tid] = {"anchor": (x, y), "since": ctx.t, "reported": False}
                continue
            duration = ctx.t - s["since"]
            if (duration >= c.stop_seconds and not s["reported"] and tr.age_s >= c.min_track_age_s
                    and _cooldown_ok(self._last, tid, ctx.t, c.cooldown_s)):
                s["reported"] = True  # one alert per stop episode
                self._last[tid] = ctx.t
                events.append(AnomalyEvent(
                    event_type=PROLONGED_STOP, video_time_s=ctx.t, frame_index=ctx.frame_index,
                    track_ids=(tid,), confidence=tr.last_conf,
                    explanation=(f"Track {tid} ({tr.label}) stayed within {c.move_threshold_px:.0f}px of "
                                 f"({s['anchor'][0]:.0f}, {s['anchor'][1]:.0f}) for {duration:.1f}s "
                                 f"(threshold {c.stop_seconds:.1f}s)."),
                    details={"stationary_s": round(duration, 2), "anchor": [round(s["anchor"][0], 1), round(s["anchor"][1], 1)],
                             "zone_time_s": round(tr.zone_time_s, 2), "zone_configured": bool(zone)}))
        return events


class ProximityRule:
    """Collision-LIKE proximity: two tracked vehicles stay very close / overlapping for several frames.

    This is a heuristic (as in the report); it cannot tell a collision from vehicles that are
    simply close in dense traffic or overlap because of camera perspective.
    """

    event_type = PROXIMITY

    def __init__(self, cfg: ProximityConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._streak: Dict[Tuple[int, int], int] = {}
        self._last: Dict[Tuple[int, int], float] = {}

    def evaluate(self, ctx: FrameContext) -> List[AnomalyEvent]:
        c = self.cfg
        if not c.enabled:
            return []
        dets = [d for d in ctx.detections if d.track_id >= 0]
        close_now: Dict[Tuple[int, int], Tuple[float, float, Detection, Detection]] = {}
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, b = dets[i], dets[j]
                if a.track_id == b.track_id:
                    continue
                dist = center_distance(a.center, b.center)
                ov = iou(a.bbox, b.bbox)
                if dist < c.distance_px or (c.min_iou > 0 and ov >= c.min_iou):
                    close_now[tuple(sorted((a.track_id, b.track_id)))] = (dist, ov, a, b)
        for key in [k for k in self._streak if k not in close_now]:
            del self._streak[key]
        events: List[AnomalyEvent] = []
        for key, (dist, ov, a, b) in close_now.items():
            self._streak[key] = self._streak.get(key, 0) + 1
            if self._streak[key] >= c.persistence_frames and _cooldown_ok(self._last, key, ctx.t, c.cooldown_s):
                self._last[key] = ctx.t
                confs = [d.confidence for d in (a, b) if d.confidence is not None]
                events.append(AnomalyEvent(
                    event_type=PROXIMITY, video_time_s=ctx.t, frame_index=ctx.frame_index,
                    track_ids=key, confidence=min(confs) if confs else None,
                    explanation=(f"Vehicles {key[0]} ({a.label}) and {key[1]} ({b.label}) stayed close for "
                                 f"{self._streak[key]} consecutive frames (centre distance {dist:.0f}px, "
                                 f"box overlap IoU {ov:.2f}; thresholds {c.distance_px:.0f}px / {c.min_iou:.2f}). "
                                 f"Proximity alone is not proof of a collision."),
                    details={"center_distance_px": round(dist, 1), "iou": round(ov, 3)}))
        return events


class CongestionRule:
    event_type = CONGESTION

    def __init__(self, cfg: CongestionConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._samples: Deque[Tuple[float, int]] = deque()
        self._above_since: Optional[float] = None
        self._last_t: Optional[float] = None

    def evaluate(self, ctx: FrameContext) -> List[AnomalyEvent]:
        c = self.cfg
        if not c.enabled:
            return []
        t = ctx.t
        self._samples.append((t, ctx.stats.vehicle_count))
        while self._samples and t - self._samples[0][0] > c.window_s:
            self._samples.popleft()
        smoothed = sum(n for _, n in self._samples) / len(self._samples)
        if smoothed >= c.count_threshold:
            if self._above_since is None:
                self._above_since = t
            sustained = t - self._above_since
            if sustained >= c.sustain_s and (self._last_t is None or t - self._last_t >= c.cooldown_s):
                self._last_t = t
                return [AnomalyEvent(
                    event_type=CONGESTION, video_time_s=t, frame_index=ctx.frame_index, track_ids=(),
                    confidence=None,
                    explanation=(f"Smoothed vehicle count {smoothed:.1f} (mean over {c.window_s:.0f}s) stayed at or above "
                                 f"{c.count_threshold:.0f} for {sustained:.1f}s (required {c.sustain_s:.1f}s). "
                                 f"Occupancy proxy {ctx.stats.occupancy:.2f}."),
                    details={"smoothed_count": round(smoothed, 2), "sustained_s": round(sustained, 2),
                             "occupancy": round(ctx.stats.occupancy, 3)})]
        elif smoothed < 0.8 * c.count_threshold:  # hysteresis: avoid flicker around the threshold
            self._above_since = None
        return []


class AnomalyEngine:
    """Owns the track history and all rules; call :meth:`process` once per processed frame."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.store = TrackStore(history_len=max(60, cfg.wrong_way.window_points), max_gap_s=2.0)
        self.rules = [WrongWayRule(cfg.wrong_way), StoppedVehicleRule(cfg.stopped_vehicle),
                      ProximityRule(cfg.proximity), CongestionRule(cfg.congestion)]

    def reset(self) -> None:
        self.store.reset()
        for r in self.rules:
            r.reset()

    def process(self, frame_index: int, t: float, detections: Sequence[Detection],
                frame_size: Tuple[int, int]) -> Tuple[FrameStats, List[AnomalyEvent]]:
        w, h = frame_size
        area = float(max(1, w * h))
        occ = min(1.0, sum(d.area for d in detections) / area)
        stats = FrameStats(frame_index, t, len(detections), sum(1 for d in detections if d.track_id >= 0), occ)
        zone_px = zone_to_pixels(self.cfg.stopped_vehicle.zone, w, h) if self.cfg.stopped_vehicle.zone else None
        tracks = self.store.update(t, detections, zone_px)
        ctx = FrameContext(frame_index, t, detections, tracks, w, h, stats)
        events: List[AnomalyEvent] = []
        for rule in self.rules:
            events.extend(rule.evaluate(ctx))
        return stats, events
