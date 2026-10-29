"""SQLite persistence for runs, anomaly events and per-frame vehicle counts."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .datatypes import REVIEW_STATUSES, AnomalyEvent, FrameStats
from .utils import csv_safe

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mode            TEXT NOT NULL CHECK (mode IN ('real', 'demo')),
    source          TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    frames_processed INTEGER DEFAULT 0,
    video_seconds   REAL,
    wall_seconds    REAL,
    measured_fps    REAL,
    mean_detect_ms  REAL,
    peak_vehicles   INTEGER,
    events_count    INTEGER DEFAULT 0,
    model           TEXT,
    config_json     TEXT,
    error           TEXT
);
CREATE TABLE IF NOT EXISTS events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER REFERENCES runs(run_id) ON DELETE CASCADE,
    mode          TEXT NOT NULL CHECK (mode IN ('real', 'demo')),
    created_at    TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    video_time_s  REAL,
    frame_index   INTEGER,
    source        TEXT NOT NULL,
    track_ids     TEXT,
    confidence    REAL,
    explanation   TEXT,
    details_json  TEXT,
    status        TEXT NOT NULL DEFAULT 'candidate',
    review_status TEXT NOT NULL DEFAULT 'unreviewed',
    reviewed_at   TEXT,
    image_path    TEXT
);
CREATE TABLE IF NOT EXISTS vehicle_counts (
    run_id        INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    video_time_s  REAL NOT NULL,
    vehicle_count INTEGER NOT NULL,
    occupancy     REAL
);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_review  ON events(review_status);
CREATE INDEX IF NOT EXISTS idx_counts_run     ON vehicle_counts(run_id);
"""


class EventStoreError(Exception):
    """Raised for any database problem (message is safe to show to the user)."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class EventStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.initialise()

    @contextmanager
    def _conn(self):
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as exc:
            raise EventStoreError(f"Cannot open database {self.db_path}: {exc}") from exc
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise EventStoreError(f"Database error: {exc}") from exc
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialise(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EventStoreError(f"Cannot create database folder {self.db_path.parent}: {exc}") from exc
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ------------------------------------------------------------------ runs
    def start_run(self, mode: str, source: str, model: Optional[str], config_json: Optional[str]) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO runs (mode, source, started_at, status, model, config_json) VALUES (?,?,?,?,?,?)",
                (mode, source, utc_now(), "running", model, config_json))
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, *, status: str, frames_processed: int, video_seconds: float,
                   wall_seconds: float, measured_fps: Optional[float], mean_detect_ms: Optional[float],
                   peak_vehicles: int, events_count: int, error: Optional[str] = None) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE runs SET finished_at=?, status=?, frames_processed=?, video_seconds=?, wall_seconds=?,
                   measured_fps=?, mean_detect_ms=?, peak_vehicles=?, events_count=?, error=? WHERE run_id=?""",
                (utc_now(), status, frames_processed, video_seconds, wall_seconds, measured_fps,
                 mean_detect_ms, peak_vehicles, events_count, error, run_id))

    def list_runs(self, mode: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        sql, params = "SELECT * FROM runs", []
        if mode in ("real", "demo"):
            sql += " WHERE mode = ?"
            params.append(mode)
        sql += " ORDER BY run_id DESC LIMIT ?"
        params.append(int(limit))
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    # ---------------------------------------------------------------- events
    def add_event(self, event: AnomalyEvent, *, run_id: Optional[int], mode: str, source: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO events (run_id, mode, created_at, event_type, video_time_s, frame_index, source,
                   track_ids, confidence, explanation, details_json, image_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, mode, utc_now(), event.event_type, float(event.video_time_s), int(event.frame_index),
                 source, ",".join(str(i) for i in event.track_ids), event.confidence, event.explanation,
                 json.dumps(event.details, default=str), event.image_path))
            return int(cur.lastrowid)

    def list_events(self, *, event_types: Optional[Sequence[str]] = None,
                    review_statuses: Optional[Sequence[str]] = None, sources: Optional[Sequence[str]] = None,
                    mode: Optional[str] = None, run_id: Optional[int] = None,
                    date_from: Optional[str] = None, date_to: Optional[str] = None,
                    limit: Optional[int] = None) -> pd.DataFrame:
        where: List[str] = []
        params: List[Any] = []

        def in_clause(column: str, values: Optional[Sequence[str]]):
            if values is not None:
                if len(values) == 0:
                    where.append("1 = 0")
                else:
                    where.append(f"{column} IN ({','.join('?' * len(values))})")
                    params.extend(values)

        in_clause("event_type", event_types)
        in_clause("review_status", review_statuses)
        in_clause("source", sources)
        if mode in ("real", "demo"):
            where.append("mode = ?")
            params.append(mode)
        if run_id is not None:
            where.append("run_id = ?")
            params.append(int(run_id))
        if date_from:
            where.append("substr(created_at, 1, 10) >= ?")
            params.append(str(date_from))
        if date_to:
            where.append("substr(created_at, 1, 10) <= ?")
            params.append(str(date_to))
        sql = "SELECT * FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY event_id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def get_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM events WHERE event_id = ?", (int(event_id),)).fetchone()
        return dict(row) if row else None

    def set_review_status(self, event_ids: Iterable[int], review_status: str) -> int:
        if review_status not in REVIEW_STATUSES:
            raise EventStoreError(f"Invalid review status {review_status!r}; use one of {REVIEW_STATUSES}.")
        ids = [int(i) for i in event_ids]
        if not ids:
            return 0
        reviewed_at = None if review_status == "unreviewed" else utc_now()
        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE events SET review_status=?, reviewed_at=? WHERE event_id IN ({','.join('?' * len(ids))})",
                [review_status, reviewed_at, *ids])
            return cur.rowcount

    def distinct_sources(self, mode: Optional[str] = None) -> List[str]:
        sql, params = "SELECT DISTINCT source FROM events", []
        if mode in ("real", "demo"):
            sql += " WHERE mode = ?"
            params.append(mode)
        with self._conn() as conn:
            return [r[0] for r in conn.execute(sql + " ORDER BY source", params).fetchall()]

    def count_by_type(self, mode: Optional[str] = None) -> Dict[str, int]:
        sql, params = "SELECT event_type, COUNT(*) AS n FROM events", []
        if mode in ("real", "demo"):
            sql += " WHERE mode = ?"
            params.append(mode)
        with self._conn() as conn:
            return {r["event_type"]: r["n"] for r in conn.execute(sql + " GROUP BY event_type", params)}

    # ---------------------------------------------------------------- counts
    def add_counts(self, run_id: int, stats: Sequence[FrameStats]) -> None:
        if not stats:
            return
        rows = [(run_id, float(s.video_time_s), int(s.vehicle_count), float(s.occupancy)) for s in stats]
        with self._conn() as conn:
            conn.executemany("INSERT INTO vehicle_counts (run_id, video_time_s, vehicle_count, occupancy) VALUES (?,?,?,?)", rows)

    def get_counts(self, run_id: int) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT video_time_s, vehicle_count, occupancy FROM vehicle_counts WHERE run_id = ? ORDER BY video_time_s",
                conn, params=[int(run_id)])

    # ----------------------------------------------------------------- admin
    def delete_runs(self, mode: str) -> int:
        """Delete all runs (and their events/counts via cascade) of one mode, e.g. clear demo data."""
        if mode not in ("real", "demo"):
            raise EventStoreError("mode must be 'real' or 'demo'.")
        with self._conn() as conn:
            conn.execute("DELETE FROM events WHERE mode = ?", (mode,))  # also catches events without a run
            cur = conn.execute("DELETE FROM runs WHERE mode = ?", (mode,))
            return cur.rowcount


def events_to_csv(df: pd.DataFrame) -> str:
    """CSV text for export; text cells are neutralised against spreadsheet formula injection."""
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(csv_safe)
    return out.to_csv(index=False)
