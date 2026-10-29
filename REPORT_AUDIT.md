# Phase 1 - Audit of the report's source code (Appendix A) and how it was completed

## What the report's code is
A single-file Streamlit script (~4 pages in the report): sidebar settings -> video upload -> `model.track(...)` with
Ultralytics (`yolo11n.pt`, built-in ByteTrack by default) -> three heuristic rules evaluated inside the frame loop ->
annotated frame preview, vehicle-count line chart, event table, two CSV downloads.

| Report requirement (chapter) | Where it was in the report code | Where it is now |
|---|---|---|
| Video input, preprocessing (4.3.1-4.3.2) | `cv2.VideoCapture`, frame stride | `video_source.py` (validation, stride, optional resize, FPS fallback, guaranteed release) |
| Vehicle detection (3.4, 6.4) | `model.track`, class filter | `detector.py` (`YoloVehicleDetector`, explicit errors, no fake fallback) |
| Vehicle tracking (3.5, 6.5) | Ultralytics ByteTrack through `model.track` | same tracker; history in `tracking.py` |
| Wrong-way movement (3.6.1) | `dx` over last 12 points vs +/-25 px | `anomaly_rules.WrongWayRule` |
| Prolonged stopping (3.6.2) | first-vs-last movement < 8 px for N s | `anomaly_rules.StoppedVehicleRule` |
| Collision-like proximity (3.6.3) | centre distance < px, any pair | `anomaly_rules.ProximityRule` |
| Event logging (3.7, 6.7) | in-memory list + CSV download | `event_store.py` (SQLite) + CSV export |
| Dashboard (3.8, 6.8) | one page | `dashboard/pages.py` (6 pages) |
| Test plan T01-T08 (Appendix C) | none | `tests/` + manual checklist in README |
| Congestion (from the project brief) | **not in the report** | `CongestionRule` (clearly marked as an addition) |

## Gaps and defects found in the supplied code, and what was changed
1. **Re-processing on every interaction.** Streamlit re-runs the whole script on each widget change, so the video was
   re-analysed whenever a slider moved. -> Analysis now starts only from a *Start analysis* button.
2. **Wrong-way with no persistence.** An alert could fire from two points and a 25 px move on any axis. -> Needs a minimum
   number of observations, movement mostly along the configured axis, and several consecutive frames; per-vehicle cooldown.
3. **Stop timer not tied to track lifetime.** `stationary_since` was never cleared when a track disappeared, so a
   returning ID could inherit an old timer; "movement" was first-vs-last of a short window. -> Anchor-based stationary test
   (robust to box jitter), state dropped with the track, one alert per stop episode, optional road zone.
4. **Proximity fired on single frames and on untracked boxes (ID -1).** -> Only tracked vehicles, N consecutive frames,
   IoU option, per-pair cooldown, explanation states that proximity is not proof of a collision.
5. **Deduplication scanned the whole event list by message text** and re-alerted every 3-5 s for a persistent condition.
   -> Per-type cooldowns keyed by track / pair.
6. **Model cached across uploads with `persist=True`**, so tracker IDs could carry over from one video into the next. ->
   A fresh model/tracker session is created for every run.
7. **Pixel thresholds depended on video resolution.** -> Frames are resized (default 960 px wide) before detection; all
   pixel thresholds refer to that processed frame.
8. **Fixed temp file name, no validation beyond extension, no resource release on error.** -> Sanitised unique upload names,
   extension / size / decodability checks, `finally`-based release.
9. **`use_container_width`** is deprecated in recent Streamlit releases -> wrapped in a compatibility helper.
10. **No persistence, review workflow or reproducible tests.** -> SQLite, review statuses, 75 automated tests.

## Conflicts / ambiguities
* The report's anomaly categories are wrong-way movement, prolonged stopping and collision-like proximity. The project brief
  also asks for *unusual congestion*; it is implemented as an **addition** and is not described in the report text.
* The report says "ByteTrack or SORT, depending on implementation"; the supplied code uses Ultralytics' built-in
  ByteTrack, which is what is used here. SORT is not implemented.
* The report's sample outputs are explicitly synthetic illustrations; no measured accuracy or speed exists in the report and
  none has been added.
