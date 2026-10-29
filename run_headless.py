"""Run the same pipeline without the dashboard (useful for benchmarking and quick checks).

    python scripts/run_headless.py --demo                      # simulated scene (NOT AI inference)
    python scripts/run_headless.py --video videos/clip.mp4     # real YOLO detection + tracking
    python scripts/run_headless.py --video videos/clip.mp4 --max-seconds 30 --stride 3

Events and measured speed are stored in the same SQLite database the dashboard reads.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traffic_anomaly.anomaly_rules import AnomalyEngine  # noqa: E402
from traffic_anomaly.config import ConfigError, load_config  # noqa: E402
from traffic_anomaly.datatypes import EVENT_LABELS  # noqa: E402
from traffic_anomaly.demo import SimulatedDetector, SimulatedSource  # noqa: E402
from traffic_anomaly.detector import DetectionError, ModelLoadError, YoloVehicleDetector  # noqa: E402
from traffic_anomaly.event_store import EventStore, EventStoreError  # noqa: E402
from traffic_anomaly.pipeline import run_analysis  # noqa: E402
from traffic_anomaly.utils import format_clock, resolve_path  # noqa: E402
from traffic_anomaly.video_source import VideoSource, VideoValidationError, validate_video_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="path to a recorded traffic video")
    ap.add_argument("--demo", action="store_true", help="use the simulated scene instead of a video")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--stride", type=int, default=None, help="process every Nth frame")
    ap.add_argument("--label", default=None, help="source label stored with the events")
    args = ap.parse_args()
    if not args.demo and not args.video:
        ap.error("give --video PATH or --demo")
    try:
        cfg = load_config()
        if args.stride:
            cfg.video.frame_stride = args.stride
        store = EventStore(resolve_path(cfg.storage.db_path))
        if args.demo:
            source, detector, label = SimulatedSource(), SimulatedDetector(), "DEMO (simulated scene)"
        else:
            info = validate_video_file(args.video, cfg.video)
            print(f"Video: {info.width}x{info.height} @ {info.fps:.2f} FPS, {info.size_mb:.1f} MB")
            detector = YoloVehicleDetector(cfg.model).load()
            source = VideoSource(args.video, cfg.video, max_seconds=args.max_seconds)
            label = args.label or Path(args.video).name
    except (ConfigError, VideoValidationError, ModelLoadError, EventStoreError) as exc:
        print(f"ERROR: {exc}")
        return 1

    def progress(u):
        if u.packet.index % 50 == 0:
            print(f"  t={format_clock(u.packet.t)}  vehicles={u.stats.vehicle_count:3d}  events={u.events_total}  {u.running_fps:.1f} frames/s")

    s = run_analysis(source, detector, AnomalyEngine(cfg), store, cfg=cfg, source_id=label, render=False, on_update=progress)
    print(f"\nStatus: {s.status}" + (f"  ERROR: {s.error}" if s.error else ""))
    print(f"Frames processed: {s.frames_processed}   peak vehicles: {s.peak_vehicles}   "
          f"measured speed on this machine: {s.measured_fps or 0:.1f} frames/s")
    for w in s.warnings:
        print("WARNING:", w)
    for e in s.events:
        print(f"  [{format_clock(e.video_time_s)}] {EVENT_LABELS[e.event_type]}: {e.explanation}")
    print(f"Stored in {store.db_path}")
    return 0 if s.status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
