"""Summaries computed from *recorded* events and counts only (no invented numbers)."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .datatypes import EVENT_LABELS, EVENT_TYPES


def with_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "event_type" in out.columns:
        out["event_label"] = out["event_type"].map(EVENT_LABELS).fillna(out["event_type"])
    return out


def type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["event_type"].value_counts() if not df.empty else pd.Series(dtype=int)
    rows = [{"event_type": t, "label": EVENT_LABELS[t], "count": int(counts.get(t, 0))} for t in EVENT_TYPES]
    return pd.DataFrame(rows)


def review_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"review_status": [], "count": []})
    d = df["review_status"].value_counts().rename_axis("review_status").reset_index(name="count")
    return d


def events_by_video_time(df: pd.DataFrame, bin_s: float = 10.0) -> pd.DataFrame:
    """Events per video-time bin and type (useful for offline videos)."""
    if df.empty:
        return pd.DataFrame({"time_bin_s": [], "label": [], "count": []})
    d = with_labels(df)
    d["time_bin_s"] = (d["video_time_s"] // bin_s) * bin_s
    return d.groupby(["time_bin_s", "event_label"]).size().reset_index(name="count").rename(columns={"event_label": "label"})


def events_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Events per recording day (UTC date the event was stored)."""
    if df.empty:
        return pd.DataFrame({"date": [], "label": [], "count": []})
    d = with_labels(df)
    d["date"] = d["created_at"].str.slice(0, 10)
    return d.groupby(["date", "event_label"]).size().reset_index(name="count").rename(columns={"event_label": "label"})


def counts_summary(counts: pd.DataFrame) -> Dict[str, float]:
    if counts.empty:
        return {"samples": 0, "peak": 0, "mean": 0.0}
    return {"samples": int(len(counts)), "peak": int(counts["vehicle_count"].max()),
            "mean": round(float(counts["vehicle_count"].mean()), 2)}


def run_performance(runs: pd.DataFrame) -> pd.DataFrame:
    """Measured processing performance per run (as recorded on THIS machine)."""
    cols = ["run_id", "mode", "source", "status", "frames_processed", "video_seconds", "wall_seconds",
            "measured_fps", "mean_detect_ms", "peak_vehicles", "events_count", "model"]
    if runs.empty:
        return pd.DataFrame(columns=cols)
    return runs[[c for c in cols if c in runs.columns]].round(2)


def review_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Share of *human-reviewed* alerts marked false positive. Not a precision/recall evaluation."""
    if df.empty:
        return {"total": 0, "reviewed": 0, "false_positive": 0, "fp_share": None}
    fp = int((df["review_status"] == "false_positive").sum())
    ok = int((df["review_status"] == "reviewed").sum())
    judged = fp + ok
    return {"total": int(len(df)), "reviewed": ok, "false_positive": fp,
            "fp_share": (fp / judged) if judged else None}
