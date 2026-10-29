"""Real YOLO vehicle detection + tracking through Ultralytics.

This module never fabricates detections: if the model cannot be loaded or inference fails,
a :class:`ModelLoadError` / :class:`DetectionError` is raised with a readable message.
(Simulated detections for the demo mode live in ``demo.py`` and are clearly flagged.)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .config import ModelConfig
from .datatypes import Detection, FramePacket
from .utils import resolve_path


class ModelLoadError(Exception):
    pass


class DetectionError(Exception):
    pass


def model_status(cfg: ModelConfig) -> dict:
    """Cheap check (no model load) used by the UI."""
    path = resolve_path(cfg.weights)
    try:
        import ultralytics  # noqa: F401
        has_pkg = True
    except Exception:
        has_pkg = False
    return {"weights_path": str(path), "weights_exist": path.exists(), "ultralytics_installed": has_pkg}


def download_weights(cfg: ModelConfig) -> Path:
    """Let Ultralytics download the weights named in ``cfg.weights`` into that path.

    Ultralytics' YOLO weights are distributed under AGPL-3.0 (see https://github.com/ultralytics/ultralytics);
    check that licence suits your use. Needs internet access.
    """
    target = resolve_path(cfg.weights)
    if target.exists():
        return target
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise ModelLoadError(f"Ultralytics is not installed ({exc}). Run: pip install -r requirements.txt") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(target.parent)  # Ultralytics downloads bare file names into the working directory
        YOLO(target.name)
    except Exception as exc:
        raise ModelLoadError(f"Model download failed: {exc}. Check your internet connection, or download "
                             f"'{target.name}' manually into {target.parent}.") from exc
    finally:
        os.chdir(cwd)
    if not target.exists():
        raise ModelLoadError(f"Download finished but {target} was not created. Place the weights there manually.")
    return target


class YoloVehicleDetector:
    is_simulated = False

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model = None
        self.class_ids: List[int] = []
        self.model_label = Path(cfg.weights).name

    def load(self) -> "YoloVehicleDetector":
        path = resolve_path(self.cfg.weights)
        if not path.exists():
            raise ModelLoadError(f"Model weights not found at {path}. Run 'python scripts/download_model.py' "
                                 "or use the Download button in Settings.")
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise ModelLoadError(f"Ultralytics could not be imported ({exc}). Run: pip install -r requirements.txt") from exc
        try:
            self.model = YOLO(str(path))
        except Exception as exc:
            raise ModelLoadError(f"The model file could not be loaded ({exc}). It may be corrupt - delete it and download again.") from exc
        names = getattr(self.model, "names", None) or {}
        wanted = {c.lower() for c in self.cfg.vehicle_classes}
        self.class_ids = sorted(int(i) for i, n in names.items() if str(n).lower() in wanted)
        if not self.class_ids:
            raise ModelLoadError(f"This model has none of the configured vehicle classes {sorted(wanted)}. "
                                 f"Model classes: {list(names.values())[:20]}")
        return self

    def new_session(self) -> None:
        """Start a fresh tracker for a new video (otherwise IDs continue from the previous video)."""
        self.load()

    def detect(self, packet: FramePacket) -> List[Detection]:
        if self.model is None:
            raise DetectionError("The model has not been loaded.")
        c = self.cfg
        try:
            kwargs = dict(persist=True, conf=c.confidence, imgsz=c.imgsz, tracker=c.tracker,
                          classes=self.class_ids, verbose=False)
            if c.device != "":
                kwargs["device"] = c.device
            result = self.model.track(packet.image, **kwargs)[0]
        except Exception as exc:
            raise DetectionError(f"Inference failed on frame {packet.index}: {exc}") from exc
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)
        names = result.names
        out: List[Detection] = []
        for box, cid, conf, tid in zip(xyxy, cls_ids, confs, ids):
            label = str(names.get(int(cid), cid))
            out.append(Detection(track_id=int(tid), label=label, confidence=float(conf),
                                 bbox=tuple(float(v) for v in box)))
        return out
