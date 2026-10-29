import tempfile
import unittest
from pathlib import Path

import pandas as pd

from traffic_anomaly import analytics
from traffic_anomaly.datatypes import EVENT_TYPES, AnomalyEvent, PROLONGED_STOP, WRONG_WAY
from traffic_anomaly.event_store import EventStore


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        s = EventStore(Path(self.tmp.name) / "e.db")
        for kind, t in ((WRONG_WAY, 3.0), (WRONG_WAY, 14.0), (PROLONGED_STOP, 15.0)):
            s.add_event(AnomalyEvent(kind, t, 1, (1,), 0.5, "x"), run_id=None, mode="real", source="a")
        self.df = s.list_events()
        self.store = s

    def tearDown(self):
        self.tmp.cleanup()

    def test_distribution_has_every_category(self):
        d = analytics.type_distribution(self.df)
        self.assertEqual(len(d), len(EVENT_TYPES))
        self.assertEqual(int(d.set_index("event_type").loc[WRONG_WAY, "count"]), 2)
        self.assertEqual(int(d["count"].sum()), 3)

    def test_time_bins(self):
        t = analytics.events_by_video_time(self.df, 10.0)
        self.assertEqual(int(t["count"].sum()), 3)
        self.assertEqual(sorted(t["time_bin_s"].unique().tolist()), [0.0, 10.0])

    def test_empty_inputs_do_not_crash(self):
        e = pd.DataFrame()
        self.assertEqual(int(analytics.type_distribution(e)["count"].sum()), 0)
        self.assertTrue(analytics.events_by_video_time(e).empty)
        self.assertTrue(analytics.events_by_day(e).empty)
        self.assertEqual(analytics.review_stats(e)["total"], 0)
        self.assertEqual(analytics.counts_summary(e)["samples"], 0)

    def test_review_stats(self):
        ids = self.df["event_id"].tolist()
        self.store.set_review_status([ids[0]], "reviewed")
        self.store.set_review_status([ids[1]], "false_positive")
        rs = analytics.review_stats(self.store.list_events())
        self.assertEqual((rs["reviewed"], rs["false_positive"], rs["fp_share"]), (1, 1, 0.5))

    def test_by_day(self):
        d = analytics.events_by_day(self.df)
        self.assertEqual(int(d["count"].sum()), 3)


if __name__ == "__main__":
    unittest.main()
