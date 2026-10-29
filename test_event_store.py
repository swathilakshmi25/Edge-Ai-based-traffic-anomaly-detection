import tempfile
import unittest
from pathlib import Path

import pandas as pd

from traffic_anomaly.datatypes import AnomalyEvent, FrameStats, PROLONGED_STOP, WRONG_WAY
from traffic_anomaly.event_store import EventStore, EventStoreError, events_to_csv


def ev(kind=WRONG_WAY, t=1.0, tids=(3,), expl="x", conf=0.8):
    return AnomalyEvent(kind, t, 10, tids, conf, expl, {"k": 1})


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "e.db"
        self.store = EventStore(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_creates_file_and_folder(self):
        self.assertTrue(self.db.exists())

    def test_add_and_list(self):
        rid = self.store.start_run("real", "cam1", "yolo", "{}")
        eid = self.store.add_event(ev(), run_id=rid, mode="real", source="cam1")
        df = self.store.list_events()
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["event_id"], eid)
        self.assertEqual(row["track_ids"], "3")
        self.assertEqual(row["review_status"], "unreviewed")
        self.assertEqual(row["status"], "candidate")

    def test_persistence_across_instances(self):
        self.store.add_event(ev(), run_id=None, mode="real", source="a")
        again = EventStore(self.db)
        self.assertEqual(len(again.list_events()), 1)

    def test_filters(self):
        self.store.add_event(ev(WRONG_WAY), run_id=None, mode="real", source="a")
        self.store.add_event(ev(PROLONGED_STOP), run_id=None, mode="real", source="b")
        self.store.add_event(ev(PROLONGED_STOP), run_id=None, mode="demo", source="DEMO")
        self.assertEqual(len(self.store.list_events(event_types=[PROLONGED_STOP])), 2)
        self.assertEqual(len(self.store.list_events(event_types=[PROLONGED_STOP], mode="real")), 1)
        self.assertEqual(len(self.store.list_events(sources=["a"])), 1)
        self.assertEqual(len(self.store.list_events(event_types=[])), 0)
        self.assertEqual(len(self.store.list_events(date_from="2000-01-01", date_to="2999-01-01")), 3)
        self.assertEqual(len(self.store.list_events(date_from="2999-01-01")), 0)
        self.assertEqual(self.store.count_by_type(mode="real"), {WRONG_WAY: 1, PROLONGED_STOP: 1})
        self.assertEqual(self.store.distinct_sources(), ["DEMO", "a", "b"])

    def test_review_status(self):
        eid = self.store.add_event(ev(), run_id=None, mode="real", source="a")
        self.assertEqual(self.store.set_review_status([eid], "reviewed"), 1)
        e = self.store.get_event(eid)
        self.assertEqual(e["review_status"], "reviewed")
        self.assertIsNotNone(e["reviewed_at"])
        self.store.set_review_status([eid], "unreviewed")
        self.assertIsNone(self.store.get_event(eid)["reviewed_at"])
        with self.assertRaises(EventStoreError):
            self.store.set_review_status([eid], "bogus")
        self.assertIsNone(self.store.get_event(9999))

    def test_sql_injection_attempts_are_inert(self):
        self.store.add_event(ev(), run_id=None, mode="real", source="a'; DROP TABLE events;--")
        df = self.store.list_events(sources=["a'; DROP TABLE events;--", "x') OR ('1'='1"])
        self.assertEqual(len(df), 1)
        self.assertEqual(len(self.store.list_events()), 1)

    def test_counts_and_run_lifecycle(self):
        rid = self.store.start_run("demo", "DEMO", None, None)
        self.store.add_counts(rid, [FrameStats(1, 0.1, 3, 3, 0.05), FrameStats(2, 0.2, 5, 5, 0.08)])
        self.store.finish_run(rid, status="completed", frames_processed=2, video_seconds=0.2, wall_seconds=0.01,
                              measured_fps=200.0, mean_detect_ms=0.1, peak_vehicles=5, events_count=0)
        self.assertEqual(self.store.get_counts(rid)["vehicle_count"].tolist(), [3, 5])
        runs = self.store.list_runs()
        self.assertEqual(runs.iloc[0]["status"], "completed")
        self.assertEqual(self.store.list_runs(mode="real").shape[0], 0)

    def test_delete_runs_cascades_by_mode(self):
        r1 = self.store.start_run("demo", "DEMO", None, None)
        self.store.add_event(ev(), run_id=r1, mode="demo", source="DEMO")
        self.store.add_counts(r1, [FrameStats(1, 0.1, 3, 3, 0.05)])
        r2 = self.store.start_run("real", "cam", None, None)
        self.store.add_event(ev(), run_id=r2, mode="real", source="cam")
        self.store.delete_runs("demo")
        self.assertEqual(len(self.store.list_events()), 1)
        self.assertEqual(self.store.get_counts(r1).shape[0], 0)
        with self.assertRaises(EventStoreError):
            self.store.delete_runs("everything")

    def test_unusable_path_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(EventStoreError):
                EventStore(Path(d))  # a directory, not a database file

    def test_csv_export(self):
        self.store.add_event(ev(expl="=HYPERLINK(\"http://evil\")"), run_id=None, mode="real", source="@src")
        df = self.store.list_events()
        text = events_to_csv(df)
        back = pd.read_csv(__import__("io").StringIO(text))
        self.assertEqual(len(back), 1)
        for col in ("event_id", "event_type", "created_at", "review_status", "video_time_s"):
            self.assertIn(col, back.columns)
        self.assertTrue(back.loc[0, "explanation"].startswith("'="))  # formula neutralised
        self.assertTrue(back.loc[0, "source"].startswith("'@"))


if __name__ == "__main__":
    unittest.main()
