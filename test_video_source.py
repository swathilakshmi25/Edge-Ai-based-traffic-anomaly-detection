"""Video validation / reading tests. They need OpenCV and are skipped when it is not installed."""
import tempfile
import unittest
from pathlib import Path

from traffic_anomaly.config import VideoConfig

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None

from traffic_anomaly.video_source import VideoSource, VideoValidationError, validate_video_file


def write_video(path, n=30, fps=10, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        frame = np.full((size[1], size[0], 3), i * 5 % 255, dtype=np.uint8)
        w.write(frame)
    w.release()


class ValidationWithoutOpenCV(unittest.TestCase):
    def test_missing_and_bad_extension(self):
        cfg = VideoConfig()
        with self.assertRaises(VideoValidationError):
            validate_video_file("no_such_file.mp4", cfg)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.txt"
            p.write_text("hi")
            with self.assertRaises(VideoValidationError) as cm:
                validate_video_file(p, cfg)
            self.assertIn("Unsupported", str(cm.exception))
            e = Path(d) / "empty.mp4"
            e.write_bytes(b"")
            with self.assertRaises(VideoValidationError):
                validate_video_file(e, cfg)
            big = VideoConfig(max_upload_mb=1)
            b = Path(d) / "big.mp4"
            b.write_bytes(b"0" * (2 * 1024 * 1024))
            with self.assertRaises(VideoValidationError) as cm:
                validate_video_file(b, big)
            self.assertIn("limit", str(cm.exception))


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class WithOpenCV(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.avi = Path(self.tmp.name) / "t.avi"
        write_video(self.avi)

    def tearDown(self):
        self.tmp.cleanup()

    def test_corrupt_file_rejected(self):
        bad = Path(self.tmp.name) / "bad.mp4"
        bad.write_bytes(b"this is not a video" * 100)
        with self.assertRaises(VideoValidationError):
            validate_video_file(bad, VideoConfig())

    def test_valid_video_info(self):
        info = validate_video_file(self.avi, VideoConfig())
        self.assertEqual((info.width, info.height), (320, 240))
        self.assertAlmostEqual(info.fps, 10.0, places=1)

    def test_stride_and_timestamps(self):
        src = VideoSource(str(self.avi), VideoConfig(frame_stride=3, resize_width=0))
        packets = list(src)
        self.assertEqual([p.index for p in packets], [1, 4, 7, 10, 13, 16, 19, 22, 25, 28])
        self.assertAlmostEqual(packets[1].t, 0.4, places=2)

    def test_resize(self):
        src = VideoSource(str(self.avi), VideoConfig(frame_stride=1, resize_width=160))
        first = next(iter(src))
        self.assertEqual(first.image.shape[:2], (120, 160))
        src.close()

    def test_max_seconds(self):
        src = VideoSource(str(self.avi), VideoConfig(frame_stride=1, resize_width=0), max_seconds=1.0)
        self.assertEqual(len(list(src)), 10)

    def test_capture_released_after_early_stop(self):
        src = VideoSource(str(self.avi), VideoConfig(frame_stride=1))
        it = iter(src)
        next(it)
        src.close()
        self.assertIsNone(src._cap)

    def test_unopenable_source(self):
        src = VideoSource(str(Path(self.tmp.name) / "none.avi"), VideoConfig())
        with self.assertRaises(VideoValidationError):
            list(src)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class EndToEndWithRealFile(unittest.TestCase):
    """Real video file -> VideoSource -> (stand-in detector) -> rules -> SQLite. Not a YOLO test."""

    def test_file_through_pipeline(self):
        from traffic_anomaly.anomaly_rules import AnomalyEngine
        from traffic_anomaly.config import AppConfig
        from traffic_anomaly.datatypes import Detection
        from traffic_anomaly.event_store import EventStore
        from traffic_anomaly.pipeline import run_analysis

        class MovingLeft:  # pretends a vehicle drives right->left (wrong way for expected "right")
            is_simulated = False
            model_label = "stand-in (test)"

            def new_session(self):
                pass

            def detect(self, packet):
                x = 300 - 80 * packet.t
                return [Detection(1, "car", 0.9, (x - 20, 100, x + 20, 130))]

        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "clip.avi"
            write_video(video, n=100, fps=10, size=(320, 240))
            cfg = AppConfig()
            cfg.video.frame_stride = 1
            cfg.video.resize_width = 0
            info = validate_video_file(video, cfg.video)
            store = EventStore(Path(d) / "e.db")
            summary = run_analysis(VideoSource(str(video), cfg.video), MovingLeft(), AnomalyEngine(cfg), store,
                                   cfg=cfg, source_id="clip.avi", render=True)
            self.assertEqual(summary.status, "completed")
            self.assertEqual(summary.frames_processed, 100)
            self.assertEqual(summary.mode, "real")
            ev = store.list_events()
            self.assertEqual(ev["event_type"].tolist(), ["wrong_way"])
            self.assertEqual(ev.iloc[0]["source"], "clip.avi")
            self.assertGreater(info.fps, 0)
