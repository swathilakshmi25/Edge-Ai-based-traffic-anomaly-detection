"""The six dashboard pages. Everything shown comes from the SQLite store or the live pipeline."""
from __future__ import annotations

import importlib.metadata as md
import platform
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from .. import analytics
from ..anomaly_rules import AnomalyEngine
from ..config import DIRECTIONS, AppConfig, ConfigError, load_config, save_config, validate
from ..datatypes import EVENT_LABELS, EVENT_TYPES, REVIEW_STATUSES
from ..demo import SimulatedDetector, SimulatedSource
from ..detector import (DetectionError, ModelLoadError, YoloVehicleDetector, download_weights, model_status)
from ..event_store import EventStore, EventStoreError, events_to_csv
from ..pipeline import RunSummary, UpdateInfo, run_analysis
from ..utils import format_clock, resolve_path, sanitize_filename
from ..video_source import VideoSource, VideoValidationError, validate_video_file
from .helpers import show_df, show_image, show_plot
from .style import REVIEW_BANNER, SIM_BANNER

EVENT_COLUMNS = ["event_id", "created_at", "event_type", "video_time_s", "source", "track_ids",
                 "confidence", "review_status", "mode", "explanation"]


def _banners(mode: Optional[str]):
    st.markdown(REVIEW_BANNER, unsafe_allow_html=True)
    if mode == "demo":
        st.markdown(SIM_BANNER, unsafe_allow_html=True)
    elif mode is None:
        st.info("Showing real AND simulated records together. Use the 'Data view' selector to separate them.")


def _safe(fn, default=None):
    try:
        return fn()
    except EventStoreError as exc:
        st.error(f"Database problem: {exc}")
        return default


def _events_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df[[c for c in EVENT_COLUMNS if c in df.columns]].copy()
    d["event_type"] = d["event_type"].map(EVENT_LABELS).fillna(d["event_type"])
    d["video_time_s"] = d["video_time_s"].round(1)
    return d


# ===================================================================== 1. Dashboard
def page_dashboard(cfg: AppConfig, store: EventStore, mode: Optional[str]):
    st.subheader("Dashboard")
    _banners(mode)
    events = _safe(lambda: store.list_events(mode=mode), pd.DataFrame())
    by_type = events["event_type"].value_counts().to_dict() if not events.empty else {}

    last: Optional[RunSummary] = st.session_state.get("last_summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processing status", (last.status.upper() if last else "IDLE"))
    c2.metric("Frames processed (last run)", last.frames_processed if last else 0)
    c3.metric("Peak vehicles in a frame (last run)", last.peak_vehicles if last else 0)
    c4.metric("Total anomaly events stored", len(events))
    if last:
        tag = "SIMULATED" if last.mode == "demo" else "real detector"
        fps = f"{last.measured_fps:.1f} frames/s measured" if last.measured_fps else "n/a"
        st.caption(f"Last run in this session: {last.source} ({tag}) - {fps}.")
        if last.error:
            st.error(f"The last run failed: {last.error}")

    cols = st.columns(len(EVENT_TYPES))
    for col, t in zip(cols, EVENT_TYPES):
        col.metric(EVENT_LABELS[t], int(by_type.get(t, 0)))

    left, right = st.columns(2)
    with left:
        st.markdown("**Events by category**")
        dist = analytics.type_distribution(events)
        fig = px.bar(dist, x="label", y="count", labels={"label": "", "count": "Events"})
        show_plot(fig)
    with right:
        st.markdown("**Events by video time (10 s bins)**")
        trend = analytics.events_by_video_time(events, 10.0)
        if trend.empty:
            st.info("No events recorded yet. Run an analysis first.")
        else:
            show_plot(px.bar(trend, x="time_bin_s", y="count", color="label",
                             labels={"time_bin_s": "Video time (s)", "count": "Events", "label": "Category"}))
    st.markdown("**Recent events**")
    if events.empty:
        st.info("No events stored for this data view yet.")
    else:
        show_df(_events_display(events.head(10)))


# ===================================================================== 2. Video analysis
def _save_upload(uploaded, cfg: AppConfig) -> Path:
    ext = Path(uploaded.name).suffix.lower().lstrip(".")
    if ext not in [e.lower().lstrip(".") for e in cfg.video.allowed_extensions]:
        raise VideoValidationError(f"Unsupported file type '.{ext}'. Allowed: {', '.join(cfg.video.allowed_extensions)}.")
    if uploaded.size > cfg.video.max_upload_mb * 1024 * 1024:
        raise VideoValidationError(f"File is larger than the configured limit of {cfg.video.max_upload_mb} MB.")
    folder = resolve_path(cfg.storage.upload_dir)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{datetime.now():%Y%m%d_%H%M%S}_{sanitize_filename(uploaded.name)}"
    with open(target, "wb") as fh:
        fh.write(uploaded.getbuffer())
    return target


def page_video_analysis(cfg: AppConfig, store: EventStore, mode: Optional[str]):
    st.subheader("Live Monitoring / Video Analysis")
    st.markdown(REVIEW_BANNER, unsafe_allow_html=True)

    kind = st.radio("Input", ["Upload recorded traffic video", "Simulated demo scene (not AI)",
                              "Camera (optional, experimental)"], horizontal=True)
    simulated = kind.startswith("Simulated")
    camera = kind.startswith("Camera")

    # -------- model readiness (real modes only)
    if not simulated:
        ms = model_status(cfg.model)
        if not ms["ultralytics_installed"]:
            st.error("Ultralytics is not installed. Run `pip install -r requirements.txt` in your virtual environment.")
        elif not ms["weights_exist"]:
            st.warning(f"Model weights not found at `{ms['weights_path']}`.")
            if st.button("Download model weights now (needs internet)"):
                with st.spinner("Downloading..."):
                    try:
                        download_weights(cfg.model)
                        st.success("Model downloaded.")
                        st.rerun()
                    except ModelLoadError as exc:
                        st.error(str(exc))
        else:
            st.success(f"Model ready: `{Path(ms['weights_path']).name}` (device: {cfg.model.device or 'auto'}).")
    else:
        st.markdown(SIM_BANNER, unsafe_allow_html=True)

    source_label = cfg.source_id
    video_path: Optional[Path] = None
    cam_index, cam_seconds = 0, 30
    uploaded = None
    if simulated:
        source_label = "DEMO (simulated scene)"
        st.caption("A scripted 70 s scene with one wrong-way vehicle, one long stop, two converging vehicles and a "
                   "congestion build-up. It exercises tracking, rules, database and dashboard - not the AI model.")
    elif camera:
        cam_index = int(st.number_input("Camera index", 0, 10, 0))
        cam_seconds = int(st.number_input("Record/analyse for (seconds)", 5, 600, 30))
        source_label = st.text_input("Camera label (no credentials)", f"Camera {cam_index}")
    else:
        uploaded = st.file_uploader("Upload recorded traffic video", type=cfg.video.allowed_extensions)
        source_label = st.text_input("Source label", uploaded.name if uploaded else cfg.source_id)

    o1, o2, o3 = st.columns(3)
    max_seconds = o1.number_input("Analyse only the first N seconds (0 = all)", 0, 36000, int(cfg.video.max_seconds))
    stride = o2.select_slider("Process every Nth frame", options=[1, 2, 3, 4, 5, 6, 8, 10], value=cfg.video.frame_stride)
    ui_every = o3.slider("Refresh preview every N processed frames", 1, 20, 3)
    st.caption("To stop a running analysis use Streamlit's 'Stop' control (top right). Events found so far stay saved.")

    if not st.button("Start analysis", type="primary"):
        return

    run_cfg = cfg.copy()
    run_cfg.video.frame_stride = int(stride)
    run_cfg.source_id = source_label
    problems = validate(run_cfg)
    if problems:
        st.error("Configuration problem:\n- " + "\n- ".join(problems))
        return

    # -------- build source + detector
    try:
        if simulated:
            source, detector = SimulatedSource(), SimulatedDetector()
        else:
            detector = YoloVehicleDetector(run_cfg.model).load()
            if camera:
                source = VideoSource(cam_index, run_cfg.video, max_seconds=float(cam_seconds))
            else:
                if uploaded is None:
                    st.error("Please upload a video file first.")
                    return
                video_path = _save_upload(uploaded, run_cfg)
                info = validate_video_file(video_path, run_cfg.video)
                st.caption(f"Video: {info.width}x{info.height}, {info.fps:.2f} FPS"
                           f"{' (assumed - file reports none)' if info.fps_assumed else ''}, "
                           f"{('~%.0f s' % info.duration_s) if info.duration_s else 'unknown length'}, {info.size_mb:.1f} MB")
                source = VideoSource(str(video_path), run_cfg.video, max_seconds=float(max_seconds) or None)
    except (VideoValidationError, ModelLoadError, DetectionError, OSError) as exc:
        st.error(str(exc))
        return

    # -------- run
    left, right = st.columns([3, 1])
    frame_slot = left.empty()
    progress = right.progress(0.0)
    m_veh, m_ev, m_fps = right.empty(), right.empty(), right.empty()
    ev_slot = st.empty()
    shown = {"n": 0}

    def on_update(u: UpdateInfo):
        shown["n"] += 1
        m_veh.metric("Vehicles in frame", u.stats.vehicle_count)
        m_ev.metric("Candidate events", u.events_total)
        m_fps.metric("Running FPS (measured)", f"{u.running_fps:.1f}")
        if u.progress is not None:
            progress.progress(u.progress)
        if u.annotated is not None and (shown["n"] % ui_every == 0 or u.new_events):
            show_image(frame_slot, u.annotated[:, :, ::-1], f"Frame {u.packet.index} - {format_clock(u.packet.t)}")

    with st.spinner("Processing locally..."):
        summary = run_analysis(source, detector, AnomalyEngine(run_cfg), store, cfg=run_cfg, source_id=source_label,
                               render=True, on_update=on_update)
    st.session_state["last_summary"] = summary
    progress.empty()
    if summary.status == "failed":
        st.error(f"Analysis failed: {summary.error}")
    elif summary.status == "stopped":
        st.warning("Analysis stopped early.")
    else:
        st.success("Analysis finished.")
    for w in summary.warnings:
        st.warning(w)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Frames processed", summary.frames_processed)
    k2.metric("Peak vehicles", summary.peak_vehicles)
    k3.metric("Candidate events", summary.events_count)
    k4.metric("Measured FPS (this machine)", f"{summary.measured_fps:.1f}" if summary.measured_fps else "n/a")
    if summary.events:
        show_df(pd.DataFrame([{"Video time": format_clock(e.video_time_s), "Category": EVENT_LABELS[e.event_type],
                               "Tracks": ",".join(map(str, e.track_ids)), "Why it was flagged": e.explanation}
                              for e in summary.events]))
    else:
        st.info("No rule-based candidate events were generated for this video.")


# ===================================================================== 3. Anomaly detection
def page_anomaly(cfg: AppConfig, store: EventStore, mode: Optional[str]):
    st.subheader("Anomaly Detection - rules and thresholds")
    _banners(mode)
    st.caption("Changes apply to the next analysis in this session. Use 'Save to config file' to keep them. "
               "Pixel values refer to the processed frame (see video.resize_width).")
    t1, t2, t3, t4 = st.tabs([EVENT_LABELS[t] for t in EVENT_TYPES])
    with t1:
        w = cfg.wrong_way
        st.write("A tracked vehicle is flagged when, over its recent trajectory, it moves against the configured "
                 "direction (mostly along that axis) for several consecutive frames. Single noisy detections are ignored.")
        w.enabled = st.checkbox("Enabled", w.enabled, key="ww_en")
        w.expected_direction = st.selectbox("Expected direction of travel (image coordinates)", DIRECTIONS,
                                            index=DIRECTIONS.index(w.expected_direction), key="ww_dir")
        c1, c2 = st.columns(2)
        w.window_points = c1.slider("Observations in window", 3, 60, int(w.window_points), key="ww_win")
        w.min_observations = c2.slider("Minimum observations", 2, int(w.window_points), min(int(w.min_observations), int(w.window_points)), key="ww_min")
        w.min_displacement_px = c1.slider("Movement against direction (px)", 5, 400, int(w.min_displacement_px), key="ww_px")
        w.persistence_frames = c2.slider("Persistence (consecutive frames)", 1, 20, int(w.persistence_frames), key="ww_per")
        w.cooldown_s = c1.slider("Cooldown per vehicle (s)", 0, 120, int(w.cooldown_s), key="ww_cd")
    with t2:
        s = cfg.stopped_vehicle
        st.write("A vehicle is flagged when it stays within a small radius of where it stopped for the configured "
                 "time. Optionally restrict to a road zone (set `stopped_vehicle.zone` in the config file as fractions 0..1). "
                 "Vehicles waiting in genuine queues or at signals can also trigger this rule.")
        s.enabled = st.checkbox("Enabled", s.enabled, key="st_en")
        c1, c2 = st.columns(2)
        s.stop_seconds = c1.slider("Stop duration (s)", 2, 120, int(s.stop_seconds), key="st_sec")
        s.move_threshold_px = c2.slider("Movement tolerance (px)", 2, 60, int(s.move_threshold_px), key="st_px")
        s.cooldown_s = c1.slider("Cooldown per vehicle (s)", 0, 300, int(s.cooldown_s), key="st_cd")
        st.caption(f"Zone: {'configured (%d points)' % len(s.zone) if s.zone else 'whole frame'}")
    with t3:
        p = cfg.proximity
        st.write("Heuristic from the report: two tracked vehicles whose centres are very close (or whose boxes overlap) "
                 "for several consecutive frames. This is *collision-like* proximity only - it cannot confirm a collision, "
                 "and perspective or dense traffic can cause false candidates.")
        p.enabled = st.checkbox("Enabled", p.enabled, key="px_en")
        c1, c2 = st.columns(2)
        p.distance_px = c1.slider("Centre distance threshold (px)", 10, 200, int(p.distance_px), 5, key="px_d")
        p.min_iou = c2.slider("Box overlap (IoU) threshold", 0.0, 1.0, float(p.min_iou), 0.05, key="px_iou")
        p.persistence_frames = c1.slider("Persistence (consecutive frames)", 1, 20, int(p.persistence_frames), key="px_per")
        p.cooldown_s = c2.slider("Cooldown per vehicle pair (s)", 0, 120, int(p.cooldown_s), key="px_cd")
    with t4:
        g = cfg.congestion
        st.write("Not part of the report's three categories; added from the project brief. The vehicle count is smoothed over a "
                 "time window and must stay above the threshold for a sustained period.")
        g.enabled = st.checkbox("Enabled", g.enabled, key="cg_en")
        c1, c2 = st.columns(2)
        g.count_threshold = c1.slider("Vehicles in frame (smoothed)", 2, 100, int(g.count_threshold), key="cg_n")
        g.window_s = c2.slider("Smoothing window (s)", 1, 60, int(g.window_s), key="cg_w")
        g.sustain_s = c1.slider("Sustain time (s)", 0, 120, int(g.sustain_s), key="cg_s")
        g.cooldown_s = c2.slider("Cooldown (s)", 0, 600, int(g.cooldown_s), key="cg_cd")

    if st.button("Save to config file"):
        try:
            st.success(f"Saved to {save_config(cfg)}")
        except ConfigError as exc:
            st.error(str(exc))

    st.markdown("---")
    st.markdown("**Candidate events with the rule that triggered them**")
    last = st.session_state.get("last_summary")
    df = _safe(lambda: store.list_events(mode=mode, run_id=(last.run_id if last and last.run_id else None), limit=200), pd.DataFrame())
    if df.empty:
        st.info("No events for the latest run / data view yet.")
    else:
        d = df[["event_id", "event_type", "video_time_s", "track_ids", "explanation"]].copy()
        d["event_type"] = d["event_type"].map(EVENT_LABELS)
        d["video_time_s"] = d["video_time_s"].round(1)
        show_df(d)


# ===================================================================== 4. Alert management
def page_alerts(cfg: AppConfig, store: EventStore, mode: Optional[str]):
    st.subheader("Alert Management")
    _banners(mode)
    sources = _safe(lambda: store.distinct_sources(mode), [])
    f1, f2, f3 = st.columns(3)
    types = f1.multiselect("Category", list(EVENT_TYPES), default=list(EVENT_TYPES), format_func=EVENT_LABELS.get)
    reviews = f2.multiselect("Review status", list(REVIEW_STATUSES), default=list(REVIEW_STATUSES))
    srcs = f3.multiselect("Source", sources, default=sources)
    use_dates = st.checkbox("Filter by date stored (UTC)")
    d_from = d_to = None
    if use_dates:
        rng = st.date_input("Date range", value=(date.today(), date.today()))
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            d_from, d_to = rng[0].isoformat(), rng[1].isoformat()
    df = _safe(lambda: store.list_events(event_types=types, review_statuses=reviews, sources=srcs, mode=mode,
                                         date_from=d_from, date_to=d_to), pd.DataFrame())
    st.caption(f"{len(df)} event(s) match the filters.")
    if df.empty:
        st.info("Nothing to show. Run an analysis or relax the filters.")
        return
    show_df(_events_display(df))
    st.download_button("Export filtered events (CSV)", events_to_csv(df), "traffic_events_filtered.csv", "text/csv")

    st.markdown("**Inspect and review an event**")
    ids = df["event_id"].tolist()
    chosen = st.selectbox("Event ID", ids)
    ev = _safe(lambda: store.get_event(int(chosen)))
    if ev:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write(f"**{EVENT_LABELS.get(ev['event_type'], ev['event_type'])}** - video time {format_clock(ev['video_time_s'])}, "
                     f"source *{ev['source']}* ({'SIMULATED' if ev['mode'] == 'demo' else 'real detector'})")
            st.write(ev["explanation"])
            st.caption(f"Track IDs: {ev['track_ids'] or '-'} | detector confidence: "
                       f"{('%.2f' % ev['confidence']) if ev['confidence'] is not None else 'n/a'} | "
                       f"status: {ev['status']} | review: {ev['review_status']}")
            st.code(ev["details_json"] or "{}", language="json")
        with c2:
            img = ev.get("image_path")
            if img and Path(img).exists():
                st.image(img)
            else:
                st.caption("No saved event image (enable storage.save_event_images in Settings).")
        b1, b2, b3 = st.columns(3)
        for col, label, status in ((b1, "Mark reviewed", "reviewed"), (b2, "Mark false positive", "false_positive"),
                                   (b3, "Reset to unreviewed", "unreviewed")):
            if col.button(label, key=f"rv_{status}"):
                _safe(lambda: store.set_review_status([int(chosen)], status))
                st.rerun()
    with st.expander("Bulk action"):
        if st.checkbox(f"I want to mark all {len(df)} filtered events as reviewed"):
            if st.button("Apply to all filtered events"):
                _safe(lambda: store.set_review_status(df["event_id"].tolist(), "reviewed"))
                st.rerun()


# ===================================================================== 5. Analytics
def page_analytics(cfg: AppConfig, store: EventStore, mode: Optional[str]):
    st.subheader("Analytics and Reports")
    _banners(mode)
    events = _safe(lambda: store.list_events(mode=mode), pd.DataFrame())
    runs = _safe(lambda: store.list_runs(mode=mode), pd.DataFrame())
    if events.empty and runs.empty:
        st.info("No data yet. Run an analysis first.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Category distribution**")
        dist = analytics.type_distribution(events)
        dist = dist[dist["count"] > 0]
        if dist.empty:
            st.info("No events.")
        else:
            show_plot(px.pie(dist, names="label", values="count"))
    with c2:
        st.markdown("**Review status**")
        rd = analytics.review_distribution(events)
        if rd.empty:
            st.info("No events.")
        else:
            show_plot(px.bar(rd, x="review_status", y="count"))
    rs = analytics.review_stats(events)
    if rs["fp_share"] is not None:
        st.metric("Alerts judged false positive by a human reviewer",
                  f"{rs['fp_share'] * 100:.0f}%", help="Based only on events you marked reviewed or false-positive. "
                  "This is NOT a precision/recall evaluation.")
        st.caption(f"Based on {rs['reviewed'] + rs['false_positive']} reviewed alert(s).")
    st.markdown("**Events by video time**")
    bin_s = st.select_slider("Bin size (s)", [5, 10, 30, 60], value=10)
    trend = analytics.events_by_video_time(events, float(bin_s))
    if not trend.empty:
        show_plot(px.bar(trend, x="time_bin_s", y="count", color="label", labels={"time_bin_s": "Video time (s)"}))
    st.markdown("**Events per day stored (UTC)**")
    day = analytics.events_by_day(events)
    if not day.empty:
        show_plot(px.bar(day, x="date", y="count", color="label"))

    st.markdown("**Vehicle count over time (per run)**")
    if not runs.empty:
        run_id = st.selectbox("Run", runs["run_id"].tolist(),
                              format_func=lambda r: f"#{r} - {runs.loc[runs.run_id == r, 'source'].iloc[0]} "
                                                    f"({runs.loc[runs.run_id == r, 'mode'].iloc[0]})")
        counts = _safe(lambda: store.get_counts(int(run_id)), pd.DataFrame())
        if not counts.empty:
            cs = analytics.counts_summary(counts)
            m1, m2, m3 = st.columns(3)
            m1.metric("Samples", cs["samples"]); m2.metric("Peak count", cs["peak"]); m3.metric("Mean count", cs["mean"])
            show_plot(px.line(counts, x="video_time_s", y="vehicle_count", labels={"video_time_s": "Video time (s)", "vehicle_count": "Vehicles"}))
            st.download_button("Export vehicle counts of this run (CSV)", counts.to_csv(index=False), f"vehicle_counts_run{run_id}.csv", "text/csv")
        st.markdown("**Measured processing performance (this machine, per run)**")
        st.caption("Values below are measured during your runs; they depend on hardware, model and settings. "
                   "Detection accuracy needs a labelled test set and is not computed here.")
        show_df(analytics.run_performance(runs))
        st.download_button("Export runs (CSV)", events_to_csv(analytics.run_performance(runs)), "runs.csv", "text/csv")
    if not events.empty:
        st.download_button("Export all events in this view (CSV)", events_to_csv(events), "traffic_events_all.csv", "text/csv")


# ===================================================================== 6. Settings
def _pkg(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "not installed"


def page_settings(cfg: AppConfig, store: EventStore, mode: Optional[str]):
    st.subheader("Settings")
    for w in cfg.warnings:
        st.warning(w)
    m, v, sg = cfg.model, cfg.video, cfg.storage
    st.markdown("**Model**")
    c1, c2 = st.columns(2)
    m.weights = c1.text_input("Weights path / name", m.weights)
    m.confidence = c2.slider("Detection confidence", 0.05, 0.95, float(m.confidence), 0.05)
    m.imgsz = c1.select_slider("Inference size", [320, 416, 480, 640, 800, 960, 1280], value=m.imgsz if m.imgsz in (320, 416, 480, 640, 800, 960, 1280) else 640)
    m.device = c2.text_input("Device ('cpu', '0' for first GPU, '' for auto)", m.device)
    m.tracker = c1.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"], index=0 if m.tracker != "botsort.yaml" else 1)
    m.vehicle_classes = c2.multiselect("Vehicle classes", ["car", "motorcycle", "bus", "truck", "bicycle"], default=[c for c in m.vehicle_classes if c in ["car", "motorcycle", "bus", "truck", "bicycle"]])
    st.markdown("**Video**")
    c1, c2, c3 = st.columns(3)
    v.frame_stride = c1.slider("Process every Nth frame", 1, 10, int(v.frame_stride))
    v.resize_width = c2.select_slider("Resize frames wider than (px, 0 = off)", [0, 480, 640, 800, 960, 1280, 1920], value=v.resize_width if v.resize_width in (0, 480, 640, 800, 960, 1280, 1920) else 960)
    v.max_upload_mb = c3.number_input("Max upload (MB)", 1, 20000, int(v.max_upload_mb))
    st.markdown("**Storage and privacy**")
    sg.save_event_images = st.checkbox("Save a snapshot image for each event (may show plates or faces)", sg.save_event_images)
    st.caption(f"Database: `{resolve_path(sg.db_path)}`")

    b1, b2, b3 = st.columns(3)
    if b1.button("Save settings to config file", type="primary"):
        try:
            st.success(f"Saved to {save_config(cfg)}")
        except ConfigError as exc:
            st.error(str(exc))
    if b2.button("Reload from config file"):
        try:
            st.session_state["cfg"] = load_config()
            st.rerun()
        except ConfigError as exc:
            st.error(str(exc))
    if b3.button("Check model"):
        st.json(model_status(cfg.model))

    st.markdown("**Data management**")
    d1, d2 = st.columns(2)
    sure = st.checkbox("I understand deletion is permanent")
    if d1.button("Delete SIMULATED (demo) data") and sure:
        n = _safe(lambda: store.delete_runs("demo"), 0)
        st.success(f"Removed {n} demo run(s).")
    if d2.button("Delete REAL-video data") and sure:
        n = _safe(lambda: store.delete_runs("real"), 0)
        st.success(f"Removed {n} real run(s).")
    up = resolve_path(sg.upload_dir)
    n_up = len(list(up.glob("*"))) if up.exists() else 0
    if st.button(f"Delete {n_up} uploaded video file(s) from disk") and sure:
        for f in up.glob("*"):
            try:
                f.unlink()
            except OSError as exc:
                st.error(f"Could not delete {f.name}: {exc}")
        st.success("Uploaded videos removed.")

    with st.expander("Environment"):
        st.write({"python": sys.version.split()[0], "platform": platform.platform(),
                  **{p: _pkg(p) for p in ("streamlit", "ultralytics", "opencv-python", "numpy", "pandas", "plotly", "PyYAML", "torch")}})
