import unittest

from traffic_anomaly.config import AppConfig
from traffic_anomaly.datatypes import CONGESTION, PROLONGED_STOP, PROXIMITY, WRONG_WAY
from traffic_anomaly.tracking import TrackStore
from tests.helpers import det, run, timeline


def only(events, etype):
    return [e for e in events if e.event_type == etype]


class WrongWayTests(unittest.TestCase):
    def test_leftward_vehicle_flagged_when_right_expected(self):
        ev = run(timeline(10, 10, lambda t: [det(1, 800 - 100 * t, 300)]))
        ww = only(ev, WRONG_WAY)
        self.assertEqual(len(ww), 1)  # cooldown (20 s) prevents repeats inside 10 s
        self.assertEqual(ww[0].track_ids, (1,))
        self.assertIn("against", ww[0].explanation)
        self.assertEqual(ww[0].details["expected_direction"], "right")

    def test_rightward_vehicle_not_flagged(self):
        self.assertEqual(only(run(timeline(10, 10, lambda t: [det(1, 100 + 100 * t, 300)])), WRONG_WAY), [])

    def test_single_noisy_detection_not_flagged(self):
        def make(t):
            x = 100 + 100 * t
            if abs(t - 3.0) < 1e-6:
                x -= 80  # one noisy backward jump
            return [det(1, x, 300)]
        self.assertEqual(only(run(timeline(8, 10, make)), WRONG_WAY), [])

    def test_lane_change_is_not_wrong_way(self):
        # drifting left slightly while mostly moving down the frame, expected "right"
        ev = run(timeline(8, 10, lambda t: [det(1, 500 - 6 * t, 100 + 60 * t)]))
        self.assertEqual(only(ev, WRONG_WAY), [])

    def test_direction_down(self):
        cfg = AppConfig()
        cfg.wrong_way.expected_direction = "down"
        up = run(timeline(8, 10, lambda t: [det(1, 400, 450 - 60 * t)]), cfg)
        down = run(timeline(8, 10, lambda t: [det(1, 400, 50 + 60 * t)]), cfg)
        self.assertEqual(len(only(up, WRONG_WAY)), 1)
        self.assertEqual(only(down, WRONG_WAY), [])

    def test_disabled(self):
        cfg = AppConfig()
        cfg.wrong_way.enabled = False
        self.assertEqual(run(timeline(10, 10, lambda t: [det(1, 800 - 100 * t, 300)]), cfg), [])

    def test_repeat_after_cooldown(self):
        ev = only(run(timeline(60, 10, lambda t: [det(1, 900 - 60 * t, 300)])), WRONG_WAY)
        self.assertGreaterEqual(len(ev), 2)
        self.assertGreaterEqual(ev[1].video_time_s - ev[0].video_time_s, 20.0)


class StoppedTests(unittest.TestCase):
    def jitter(self, t):
        return (t * 7919) % 3 - 1.5  # deterministic +-1.5 px jitter

    def test_long_stop_flagged_once(self):
        ev = run(timeline(30, 10, lambda t: [det(5, 400 + self.jitter(t), 250 + self.jitter(t * 3))]))
        st = only(ev, PROLONGED_STOP)
        self.assertEqual(len(st), 1)
        self.assertGreaterEqual(st[0].details["stationary_s"], 10.0)

    def test_short_stop_not_flagged(self):
        def make(t):
            return [det(5, 400 if t < 6 else 400 + 60 * (t - 6), 250)]
        self.assertEqual(only(run(timeline(14, 10, make)), PROLONGED_STOP), [])

    def test_moving_vehicle_not_flagged(self):
        self.assertEqual(only(run(timeline(30, 10, lambda t: [det(5, 50 + 20 * t, 250)])), PROLONGED_STOP), [])

    def test_timing_independent_of_frame_rate(self):
        for fps in (5, 10, 30):
            ev = only(run(timeline(14, fps, lambda t: [det(5, 400, 250)])), PROLONGED_STOP)
            self.assertEqual(len(ev), 1, f"fps={fps}")
            # first sighting + 10 s stop threshold, whatever the frame rate
            self.assertAlmostEqual(ev[0].video_time_s, 10.1, delta=0.15)

    def test_zone_excludes_vehicles_outside(self):
        cfg = AppConfig()
        cfg.stopped_vehicle.zone = [[0, 0], [1, 0], [1, 0.3], [0, 0.3]]  # top 30% of the frame only
        inside = run(timeline(14, 10, lambda t: [det(5, 400, 50)]), cfg)
        outside = run(timeline(14, 10, lambda t: [det(5, 400, 400)]), cfg)
        self.assertEqual(len(only(inside, PROLONGED_STOP)), 1)
        self.assertEqual(only(outside, PROLONGED_STOP), [])

    def test_threshold_configurable(self):
        cfg = AppConfig()
        cfg.stopped_vehicle.stop_seconds = 3
        self.assertEqual(len(only(run(timeline(6, 10, lambda t: [det(5, 400, 250)]), cfg), PROLONGED_STOP)), 1)


class ProximityTests(unittest.TestCase):
    def test_sustained_closeness_flagged_once(self):
        ev = run(timeline(5, 10, lambda t: [det(1, 300, 200), det(2, 330, 230)]))
        px = only(ev, PROXIMITY)
        self.assertEqual(len(px), 1)
        self.assertEqual(px[0].track_ids, (1, 2))
        self.assertIn("not proof", px[0].explanation)

    def test_single_frame_blip_ignored(self):
        def make(t):
            close = abs(t - 2.0) < 1e-6
            return [det(1, 300, 200), det(2, 330 if close else 600, 230)]
        self.assertEqual(only(run(timeline(5, 10, make)), PROXIMITY), [])

    def test_far_apart_not_flagged(self):
        self.assertEqual(only(run(timeline(5, 10, lambda t: [det(1, 100, 100), det(2, 500, 400)])), PROXIMITY), [])

    def test_overlap_triggers_even_if_centres_far_for_big_boxes(self):
        cfg = AppConfig()
        cfg.proximity.distance_px = 10
        ev = run(timeline(3, 10, lambda t: [det(1, 300, 200, w=200, h=100), det(2, 360, 200, w=200, h=100)]), cfg)
        self.assertEqual(len(only(ev, PROXIMITY)), 1)

    def test_untracked_detections_ignored(self):
        self.assertEqual(only(run(timeline(5, 10, lambda t: [det(-1, 300, 200), det(-1, 310, 205)])), PROXIMITY), [])


class CongestionTests(unittest.TestCase):
    def many(self, n):
        return [det(i, 40 + (i % 8) * 110, 60 + (i // 8) * 90) for i in range(n)]

    def test_sustained_high_count_flagged_once(self):
        ev = only(run(timeline(20, 10, lambda t: self.many(15))), CONGESTION)
        self.assertEqual(len(ev), 1)
        self.assertGreaterEqual(ev[0].details["sustained_s"], 5.0)

    def test_brief_spike_ignored(self):
        ev = run(timeline(20, 10, lambda t: self.many(20) if 8 < t < 9.5 else self.many(3)))
        self.assertEqual(only(ev, CONGESTION), [])

    def test_below_threshold_not_flagged(self):
        self.assertEqual(only(run(timeline(30, 10, lambda t: self.many(8))), CONGESTION), [])

    def test_cooldown_then_repeat(self):
        ev = only(run(timeline(150, 5, lambda t: self.many(15))), CONGESTION)
        self.assertGreaterEqual(len(ev), 2)
        self.assertGreaterEqual(ev[1].video_time_s - ev[0].video_time_s, 60.0)


class TrackStoreTests(unittest.TestCase):
    def test_heading_and_history(self):
        s = TrackStore()
        for i in range(10):
            s.update(i * 0.1, [det(1, 100 + 10 * i, 200)])
        tr = s.tracks[1]
        self.assertAlmostEqual(tr.heading_deg(), 0.0, places=3)  # moving right
        self.assertEqual(len(tr.points), 10)
        self.assertAlmostEqual(tr.age_s, 0.9, places=6)

    def test_stale_tracks_pruned(self):
        s = TrackStore(max_gap_s=2.0)
        s.update(0.0, [det(1, 100, 100)])
        s.update(1.0, [det(2, 200, 100)])
        s.update(5.0, [det(2, 210, 100)])
        self.assertNotIn(1, s.tracks)
        self.assertIn(2, s.tracks)

    def test_zone_dwell_time(self):
        s = TrackStore()
        zone = [(0, 0), (500, 0), (500, 500), (0, 500)]
        for i in range(11):
            s.update(i * 0.5, [det(1, 100, 100)], zone)
        self.assertAlmostEqual(s.tracks[1].zone_time_s, 5.0, places=6)

    def test_untracked_ignored(self):
        s = TrackStore()
        s.update(0.0, [det(-1, 100, 100)])
        self.assertEqual(s.tracks, {})


if __name__ == "__main__":
    unittest.main()
