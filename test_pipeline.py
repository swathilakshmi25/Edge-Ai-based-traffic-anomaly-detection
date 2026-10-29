import tempfile
import unittest
from pathlib import Path

import numpy as np

from traffic_anomaly.anomaly_rules import AnomalyEngine
from traffic_anomaly.config import AppConfig
from traffic_anomaly.datatypes import (CONGESTION, EVENT_TYPES, PROLONGED_STOP, PROXIMITY, WRONG_WAY, FramePacket)
from traffic_anomaly.demo import FRAME_H, FRAME_W, SimulatedDetector, SimulatedSource, scripted_vehicles
from traffic_anomaly.event_store import EventStore, EventStoreError
from traffic_anomaly.pipeline import run_analysis


class FakeSource:
    def __init__(self, n=20, fps=10.0):
        self.n, self.fps, self.total_frames, self.warnings, self.closed = n, fps, n, [], False

    def close(self):
        self.closed = True

    def __iter__(self):
        for i in range(1, self.n + 1):
            yield FramePacket(i, i / self.fps, np.zeros((100, 200, 3), dtype=np.uint8))


class RealLikeDetector:  # not simulated => stored as mode "real"
    is_simulated = False
    model_label = "fake"

    def __init__(self, fail_at=None):
        self.fail_at = fail_at

    def new_session(self):
        pass

    def detect(self, packet):
        if self.fail_at and packet.index == self.fail_at:
            raise RuntimeError("inference exploded")
        return []


class DemoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.tmp.name) / "e.db")
        self.cfg = AppConfig()

    def tearDown(self):
        self.tmp.cleanup()

    def test_demo_scene_triggers_each_category_once_and_persists(self):
        s = run_analysis(SimulatedSource(), SimulatedDetector(), AnomalyEngine(self.cfg), self.store, cfg=self.cfg,
                         source_id="DEMO", render=False)
        self.assertEqual(s.status, "completed")
        self.assertEqual(s.mode, "demo")
        self.assertEqual(s.frames_processed, 700)
        types = sorted(e.event_type for e in s.events)
        self.assertEqual(types, sorted([WRONG_WAY, PROLONGED_STOP, PROXIMITY, CONGESTION]))
        df = self.store.list_events()
        self.assertEqual(len(df), 4)
        self.assertTrue((df["mode"] == "demo").all())
        self.assertEqual(len(self.store.list_events(mode="real")), 0)
        runs = self.store.list_runs()
        self.assertEqual(runs.iloc[0]["status"], "completed")
        self.assertEqual(runs.iloc[0]["events_count"], 4)
        self.assertEqual(len(self.store.get_counts(s.run_id)), 700)
        self.assertGreater(s.measured_fps, 0)

    def test_demo_is_deterministic(self):
        a = run_analysis(SimulatedSource(), SimulatedDetector(), AnomalyEngine(self.cfg), None, cfg=self.cfg, source_id="d", render=False)
        b = run_analysis(SimulatedSource(), SimulatedDetector(), AnomalyEngine(self.cfg), None, cfg=self.cfg, source_id="d", render=False)
        self.assertEqual([(e.event_type, e.frame_index) for e in a.events], [(e.event_type, e.frame_index) for e in b.events])

    def test_demo_detections_stay_in_frame(self):
        d = SimulatedDetector()
        for i in range(1, 701):
            for det in d.detect(FramePacket(i, i / 10, None)):
                self.assertTrue(0 <= det.center[0] <= FRAME_W and 0 <= det.center[1] <= FRAME_H)

    def test_real_mode_label_for_non_simulated_detector(self):
        s = run_analysis(FakeSource(), RealLikeDetector(), AnomalyEngine(self.cfg), self.store, cfg=self.cfg, source_id="v", render=False)
        self.assertEqual(s.mode, "real")
        self.assertEqual(self.store.list_runs().iloc[0]["mode"], "real")

    def test_detector_failure_is_contained_and_run_closed(self):
        src = FakeSource()
        s = run_analysis(src, RealLikeDetector(fail_at=5), AnomalyEngine(self.cfg), self.store, cfg=self.cfg, source_id="v", render=False)
        self.assertEqual(s.status, "failed")
        self.assertIn("inference exploded", s.error)
        self.assertEqual(s.frames_processed, 4)
        self.assertTrue(src.closed)  # video resource released
        row = self.store.list_runs().iloc[0]
        self.assertEqual(row["status"], "failed")
        self.assertIn("inference exploded", row["error"])

    def test_stop_request(self):
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 3
        src = FakeSource()
        s = run_analysis(src, RealLikeDetector(), AnomalyEngine(self.cfg), self.store, cfg=self.cfg, source_id="v",
                         render=False, should_stop=stop)
        self.assertEqual(s.status, "stopped")
        self.assertEqual(s.frames_processed, 3)
        self.assertTrue(src.closed)

    def test_source_error_midway_is_reported_and_closed(self):
        class Boom(FakeSource):
            def __iter__(self):
                yield from list(super().__iter__())[:2]
                raise OSError("disk read error")
        src = Boom()
        s = run_analysis(src, RealLikeDetector(), AnomalyEngine(self.cfg), self.store, cfg=self.cfg, source_id="v", render=False)
        self.assertEqual(s.status, "failed")
        self.assertIn("disk read error", s.error)
        self.assertTrue(src.closed)

    def test_database_failure_does_not_crash_run(self):
        class BrokenStore:
            def start_run(self, *a, **k):
                raise EventStoreError("disk full")
        s = run_analysis(SimulatedSource(duration_s=5), SimulatedDetector(), AnomalyEngine(self.cfg), BrokenStore(),
                         cfg=self.cfg, source_id="d", render=False)
        self.assertEqual(s.status, "completed")
        self.assertTrue(any("Database unavailable" in w for w in s.warnings))

    def test_event_write_failure_is_warned_not_fatal(self):
        class FlakyStore(EventStore):
            def add_event(self, *a, **k):
                raise EventStoreError("locked")
        flaky = FlakyStore(Path(self.tmp.name) / "f.db")
        s = run_analysis(SimulatedSource(), SimulatedDetector(), AnomalyEngine(self.cfg), flaky, cfg=self.cfg, source_id="d", render=False)
        self.assertEqual(s.status, "completed")
        self.assertEqual(len(s.events), 4)
        self.assertTrue(any("could not be saved" in w for w in s.warnings))

    def test_callback_receives_updates(self):
        seen = []
        run_analysis(SimulatedSource(duration_s=3), SimulatedDetector(), AnomalyEngine(self.cfg), None, cfg=self.cfg,
                     source_id="d", render=False, on_update=lambda u: seen.append((u.packet.index, u.progress)))
        self.assertEqual(len(seen), 30)
        self.assertAlmostEqual(seen[-1][1], 1.0)

    def test_scripted_scene_has_no_duplicate_ids_per_frame(self):
        for t in (3.0, 12.0, 25.0, 45.0, 60.0):
            ids = [v[0] for v in scripted_vehicles(t)]
            self.assertEqual(len(ids), len(set(ids)), t)


if __name__ == "__main__":
    unittest.main()


try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class RenderingTests(unittest.TestCase):
    def test_overlay_and_event_images(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = AppConfig()
            cfg.storage.outputs_dir = d
            cfg.storage.save_event_images = True
            store = EventStore(Path(d) / "e.db")
            frames = []
            s = run_analysis(SimulatedSource(duration_s=30), SimulatedDetector(), AnomalyEngine(cfg), store, cfg=cfg,
                             source_id="d", render=True, on_update=lambda u: frames.append(u.annotated))
            self.assertEqual(len(frames), 300)
            self.assertEqual(frames[0].shape, (FRAME_H, FRAME_W, 3))
            self.assertFalse(np.array_equal(frames[0], SimulatedSource()._background))  # something was drawn
            imgs = list((Path(d) / "event_images").glob("*.jpg"))
            self.assertEqual(len(imgs), len(s.events))
            self.assertTrue(all(e.image_path for e in s.events))
            self.assertEqual(store.list_events()["image_path"].notna().sum(), len(s.events))
