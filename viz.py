"""Frame overlays (OpenCV, imported lazily)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Set

from .config import zone_to_pixels
from .datatypes import EVENT_LABELS, AnomalyEvent, Detection

GREEN = (20, 210, 120)
RED = (40, 40, 235)
YELLOW = (0, 200, 255)


def draw_overlay(image, detections: Sequence[Detection], flagged_ids: Set[int],
                 banner_events: Sequence[AnomalyEvent], vehicle_count: int, t: float,
                 zone: Optional[Sequence[Sequence[float]]] = None, simulated: bool = False):
    """Return an annotated copy (BGR). Boxes show label, track ID and detector confidence."""
    import cv2
    import numpy as np

    img = image.copy()
    h, w = img.shape[:2]
    if zone:
        pts = np.array(zone_to_pixels(list(zone), w, h), dtype=np.int32)
        cv2.polylines(img, [pts], True, YELLOW, 1)
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        color = RED if d.track_id in flagged_ids else GREEN
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        conf = f" {d.confidence:.2f}" if d.confidence is not None else ""
        tid = f" ID:{d.track_id}" if d.track_id >= 0 else ""
        cv2.putText(img, f"{d.label}{tid}{conf}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(img, f"t={t:5.1f}s  vehicles={vehicle_count}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if simulated:
        cv2.putText(img, "SIMULATED DATA (not AI inference)", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2, cv2.LINE_AA)
    for k, ev in enumerate(banner_events[-2:]):
        cv2.putText(img, f"POTENTIAL ANOMALY (review): {EVENT_LABELS.get(ev.event_type, ev.event_type)}",
                    (10, 48 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
    return img


def save_image(path: Path, image) -> bool:
    try:
        import cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(path), image))
    except Exception:
        return False
