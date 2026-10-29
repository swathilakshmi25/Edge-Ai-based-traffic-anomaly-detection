"""Orchestration: video source -> detector/tracker -> anomaly engine -> database -> UI callback."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .anomaly_rules import AnomalyEngine
from .config import AppConfig
from .datatypes import AnomalyEvent, Detection, FramePacket, FrameStats
from .event_store import EventStore, EventStoreError
from .utils import resolve_path

log = logging.getLogger(__name__)


@dataclass
class UpdateInfo:
    packet: FramePacket
    annotated: Any  # BGR image with overlays, or None when rendering is disabled
    detections: List[Detection]
    stats: FrameStats
    new_events: List[AnomalyEvent]
    progress: Optional[float]
    events_total: int
    running_fps: float


@dataclass
class RunSummary:
    run_id: Optional[int]
    mode: str
    source: str
    status: str  # completed | stopped | failed
    frames_processed: int = 0
    video_seconds: float = 0.0
    wall_seconds: float = 0.0
    measured_fps: Optional[float] = None
    mean_detect_ms: Optional[float] = None
    peak_vehicles: int = 0
    events_count: int = 0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    events: List[AnomalyEvent] = field(default_factory=list)


def run_analysis(source, detector, engine: AnomalyEngine, store: Optional[EventStore], *, cfg: AppConfig,
                 source_id: str, render: bool = True, on_update: Optional[Callable[[UpdateInfo], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None) -> RunSummary:
    """Process every packet from ``source``. Never raises for processing/DB errors: they are returned in the summary."""
    mode = "demo" if getattr(detector, "is_simulated", False) else "real"
    summary = RunSummary(run_id=None, mode=mode, source=source_id, status="completed")
    engine.reset()
    stats_log: List[FrameStats] = []
    detect_time = 0.0
    flagged: Dict[int, float] = {}
    recent_events: List[AnomalyEvent] = []
    save_images = bool(render and cfg.storage.save_event_images)
    images_dir = resolve_path(cfg.storage.outputs_dir) / "event_images"

    if store is not None:
        try:
            summary.run_id = store.start_run(mode, source_id, getattr(detector, "model_label", None),
                                             json.dumps(cfg.to_dict(), default=str))
        except EventStoreError as exc:
            summary.warnings.append(f"Database unavailable - events are kept in memory only: {exc}")
            store = None

    t_start = time.perf_counter()
    last_t = 0.0
    try:
        detector.new_session()
        for packet in source:
            if should_stop is not None and should_stop():
                summary.status = "stopped"
                break
            t0 = time.perf_counter()
            detections = detector.detect(packet)
            detect_time += time.perf_counter() - t0
            h, w = packet.image.shape[:2]
            stats, events = engine.process(packet.index, packet.t, detections, (w, h))
            stats_log.append(stats)
            last_t = packet.t
            summary.frames_processed += 1
            summary.peak_vehicles = max(summary.peak_vehicles, stats.vehicle_count)

            for ev in events:
                for tid in ev.track_ids:
                    flagged[tid] = packet.t
                recent_events.append(ev)
            flagged = {k: v for k, v in flagged.items() if packet.t - v <= 5.0}
            recent_events = [e for e in recent_events if packet.t - e.video_time_s <= 2.5]

            annotated = None
            if render:
                from .viz import draw_overlay  # lazy: needs OpenCV
                annotated = draw_overlay(packet.image, detections, set(flagged), recent_events, stats.vehicle_count,
                                         packet.t, cfg.stopped_vehicle.zone or None, simulated=(mode == "demo"))
            for ev in events:
                if save_images and annotated is not None:
                    from .viz import save_image
                    path = images_dir / f"run{summary.run_id or 0}_f{ev.frame_index}_{ev.event_type}.jpg"
                    if save_image(path, annotated):
                        ev.image_path = str(path)
                summary.events.append(ev)
                if store is not None:
                    try:
                        store.add_event(ev, run_id=summary.run_id, mode=mode, source=source_id)
                    except EventStoreError as exc:
                        summary.warnings.append(f"Event could not be saved: {exc}")
            elapsed = max(1e-9, time.perf_counter() - t_start)
            if on_update is not None:
                total = getattr(source, "total_frames", 0) or 0
                progress = min(1.0, packet.index / total) if total else None
                on_update(UpdateInfo(packet, annotated, detections, stats, events, progress,
                                     len(summary.events), summary.frames_processed / elapsed))
    except Exception as exc:  # processing errors must not crash the application
        log.exception("Analysis failed")
        summary.status = "failed"
        summary.error = f"{type(exc).__name__}: {exc}"
    except BaseException:  # e.g. Streamlit's Stop button / KeyboardInterrupt: record, then let it propagate
        summary.status = "stopped"
        raise
    finally:
        closer = getattr(source, "close", None)
        if callable(closer):
            closer()
        summary.warnings.extend(getattr(source, "warnings", []) or [])
        summary.wall_seconds = time.perf_counter() - t_start
        summary.video_seconds = last_t
        summary.events_count = len(summary.events)
        if summary.frames_processed:
            summary.measured_fps = summary.frames_processed / max(1e-9, summary.wall_seconds)
            summary.mean_detect_ms = 1000.0 * detect_time / summary.frames_processed
        if store is not None and summary.run_id is not None:
            try:
                store.add_counts(summary.run_id, stats_log)
                store.finish_run(summary.run_id, status=summary.status, frames_processed=summary.frames_processed,
                                 video_seconds=summary.video_seconds, wall_seconds=summary.wall_seconds,
                                 measured_fps=summary.measured_fps, mean_detect_ms=summary.mean_detect_ms,
                                 peak_vehicles=summary.peak_vehicles, events_count=summary.events_count,
                                 error=summary.error)
            except EventStoreError as exc:
                summary.warnings.append(f"Run summary could not be saved: {exc}")
    return summary
