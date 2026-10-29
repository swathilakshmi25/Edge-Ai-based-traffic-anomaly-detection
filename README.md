# EDGE AI BASED TRAFFIC ANOMALY DETECTION

Local, laptop-friendly prototype that analyses a recorded traffic video, detects and tracks vehicles with YOLO
(Ultralytics, built-in ByteTrack), applies configurable rules, stores **candidate** anomalies in SQLite and shows
everything in a Streamlit dashboard.

> **Alerts are potential anomalies requiring human review.** They are rule-based heuristics, not proof of a violation or
> collision, and this is not a certified road-safety system. No accuracy, precision, recall or speed figure is claimed:
> the software only *measures* processing speed on your machine, and accuracy needs your own labelled test set.

Built from the source code in Appendix A of the project report; see `docs/REPORT_AUDIT.md` for what was fixed/extended.

## 1. Honest status - what has and has not been verified

| Area | Status |
|---|---|
| Config, tracking history, 4 anomaly rules, cooldowns, SQLite, CSV export, analytics, pipeline, error handling | **Automated tests pass** (75 tests; Python 3.12, Linux, OpenCV 4.13, numpy 2.4, pandas 3.0) |
| Video validation / reading / resizing / stride with real files | Automated tests pass (synthetic AVI files created by the tests) |
| YOLO output parsing | Tested with a **fake** model object only |
| **Real YOLO inference + ByteTrack on a real traffic video** | **Not run** - Ultralytics/PyTorch and a traffic video were not available where this was built. You must verify this on your laptop (section 6). |
| Streamlit dashboard | Page code was exercised against a *stub* Streamlit (no exceptions). **Never run in a real browser by the author** - verify on your laptop. |
| Windows | Not tested on Windows. Paths use `pathlib`; commands below are standard. |
| Dependency versions | Given as ranges in `requirements.txt`; Streamlit/Ultralytics/Plotly/PyTorch ranges are unverified. |

## 2. Setup on Windows (VS Code)

Install Python **3.11** (3.10-3.12 should work) from python.org and tick "Add python.exe to PATH". Open the project folder
in VS Code (`File > Open Folder`), then open the terminal (Terminal > New Terminal).

PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked ("running scripts is disabled"): `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use
Command Prompt: `.venv\Scripts\activate.bat`.

Select the interpreter in VS Code: `Ctrl+Shift+P` -> *Python: Select Interpreter* -> `.venv`.

The first install downloads PyTorch (hundreds of MB to ~2 GB). CPU is enough; a GPU is optional (`model.device: "0"`).

## 3. Get the model (not bundled)

```powershell
python scripts/download_model.py
```
Downloads `yolo11n.pt` into `models/` through Ultralytics (internet needed once). Ultralytics models are AGPL-3.0 (or
commercial licence) - check this fits your use. You can also click **Download model** in the app, or place your own weights
at the path in `config/settings.yaml` (`model.weights`).

## 4. Run

```powershell
python -m streamlit run app.py
```
(or double-click `run_app.bat`). The dashboard opens at http://localhost:8501.

Headless / benchmarking:
```powershell
python scripts/run_headless.py --video videos\clip.mp4 --max-seconds 60
python scripts/run_headless.py --demo
```

## 5. Demonstrating each feature

Videos are **not** bundled. Use your own recording (with permission), or a clip whose licence allows your use (e.g. check
terms on Pexels, Wikimedia Commons, or research datasets such as UA-DETRAC). Put it anywhere and upload it in the app.

1. **Dashboard** - status, totals, per-category counts, recent events, trend charts. Use *Data view* in the sidebar:
   real and simulated records are kept apart.
2. **Simulated demo scene** (Live Monitoring page -> *Simulated demo scene*): works without a model or video. It is a
   **scripted scene, not AI inference** (labelled on every frame and in the database as `demo`). It deterministically shows
   one wrong-way vehicle, one 30 s stop, two converging vehicles and a congestion build-up, i.e. all four rules,
   database writes and review workflow. Remove it any time: Settings -> *Delete SIMULATED data*.
3. **Real video** (Live Monitoring -> *Upload recorded traffic video* -> *Start analysis*): boxes show class, track ID and
   detector confidence; red boxes belong to a flagged event. Use *Analyse only first N seconds* for quick CPU runs.
4. **Anomaly Detection** page: rule descriptions and live-editable thresholds, plus the "why it was flagged" text of each event.
5. **Alert Management**: filter by category / review status / source / date, inspect, mark *reviewed* / *false positive*,
   export the filtered table to CSV.
6. **Analytics and Reports**: category distribution, review status, trends, vehicle counts per run, **measured** speed per run, CSV exports.
7. **Settings**: model/video/storage options, save to `config/settings.yaml`, model check, data deletion, environment versions.

Set the **expected direction** (Anomaly Detection -> Wrong-way) to match the camera view: `right` means vehicles should move
towards the right edge of the image, `down` towards the bottom.

## 6. Verification checklist to run on your laptop (acceptance criteria)

```powershell
python -m unittest discover -s tests -t . -v        # automated tests (or run_tests.bat)
python scripts/download_model.py                    # model present
python scripts/run_headless.py --video videos\your_clip.mp4 --max-seconds 30
python -m streamlit run app.py
```
- [ ] Dashboard opens without a startup error.
- [ ] Upload a short daytime clip: boxes with IDs appear; vehicle counts change over time (report test T01, T03).
- [ ] Upload a `.txt` renamed or an unsupported file: clear error (T02). Delete the model file: clear "weights not found" error.
- [ ] Re-run with a known stopped vehicle / opposite-direction vehicle and check that candidates match what you see (T04, T05). Record false alarms (T07, T08).
- [ ] Close and restart the app: events are still listed (SQLite persistence).
- [ ] Export CSV from *Alert Management* and open it.
- [ ] Note the **measured** frames/s from the run summary / Analytics page for your report. Do not quote simulated-demo speeds.

For accuracy figures (precision/recall/F1 from the report's Appendix C) label a set of events in your own test videos
and compute them yourself; the app only provides the human "false positive" share of alerts you reviewed.

## 7. Configuration (`config/settings.yaml`)
Model, thresholds for every rule, video options, storage. Pixel values refer to the *processed* frame (`video.resize_width`,
default 960 px). Optional road zone for the stop rule: `stopped_vehicle.zone` as fractions, e.g. `[[0,0.4],[1,0.4],[1,1],[0,1]]`.
Invalid values stop the app with a message that names the setting. No credentials are stored anywhere; camera input uses a
local device index only.

## 8. Project layout
```
app.py                       Streamlit entry point
config/settings.yaml         all settings
traffic_anomaly/
  config.py                  load / validate / save YAML
  video_source.py            validation + frame reader (OpenCV)
  detector.py                YOLO detection + ByteTrack via Ultralytics (real inference only)
  tracking.py                per-vehicle history, heading, zone dwell time
  anomaly_rules.py           wrong-way, prolonged stop, collision-like proximity, congestion + engine
  event_store.py             SQLite (runs, events, vehicle_counts) + CSV export
  pipeline.py                source -> detector -> rules -> database -> UI callback
  analytics.py               summaries from stored records
  demo.py                    SIMULATED scene (clearly separated from AI mode)
  viz.py                     overlays
  dashboard/                 Streamlit pages
scripts/                     download_model.py, run_headless.py
tests/                       75 unittest tests (pytest also works)
docs/REPORT_AUDIT.md         audit of the report's code
```

## 9. Known limitations
* **Pixel motion is not physical speed.** No camera calibration; thresholds are in pixels of the processed frame and depend on camera angle and zoom.
* Occlusion, missed detections and **ID switches** break a vehicle's history (stops may be missed or restarted; wrong-way streaks reset).
* Stopped-vehicle rule also flags vehicles legitimately queued or waiting at signals unless you configure a zone / longer time.
* Proximity is a heuristic: perspective and dense traffic produce close boxes without any collision.
* Congestion uses vehicles currently in frame, which depends on field of view; tune the threshold per camera.
* Streamlit cannot cancel a running loop from a button; use its *Stop* control. Events found before stopping remain saved.
* The camera option is experimental and untested; recorded video is the supported workflow.
* Event snapshots (off by default) may show plates or faces; keep them off unless needed and delete data when finished (Settings).
* SORT is not implemented (the report allows "ByteTrack or SORT"; ByteTrack is used). MQTT/ONNX are not part of this build.

## 10. Troubleshooting
* `ModuleNotFoundError` -> the virtual environment is not active / interpreter not selected.
* ByteTrack error about `lap` -> `pip install lapx`.
* Very slow on CPU -> raise *Process every Nth frame*, lower `model.imgsz` (e.g. 480), use *first N seconds*.
* Port busy -> `python -m streamlit run app.py --server.port 8502`.
