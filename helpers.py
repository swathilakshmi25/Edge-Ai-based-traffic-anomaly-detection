from traffic_anomaly.anomaly_rules import AnomalyEngine
from traffic_anomaly.config import AppConfig
from traffic_anomaly.datatypes import Detection

W, H = 960, 540


def det(tid, cx, cy, label="car", conf=0.9, w=100, h=50):
    return Detection(tid, label, conf, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def run(frames, cfg=None):
    """frames: iterable of (t, [Detection]). Returns all events produced."""
    engine = AnomalyEngine(cfg or AppConfig())
    events = []
    for i, (t, dets) in enumerate(frames, start=1):
        _, evs = engine.process(i, t, dets, (W, H))
        events.extend(evs)
    return events


def timeline(duration_s, fps, make):
    """make(t) -> list of detections; yields (t, dets) at the given fps."""
    n = int(duration_s * fps)
    return [((i + 1) / fps, make((i + 1) / fps)) for i in range(n)]
