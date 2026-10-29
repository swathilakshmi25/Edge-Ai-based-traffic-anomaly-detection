"""Configuration loading, validation and saving (YAML)."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .utils import project_root, resolve_path

DIRECTIONS = ("right", "left", "up", "down")
DEFAULT_CONFIG_PATH = project_root() / "config" / "settings.yaml"


class ConfigError(Exception):
    """Raised when the configuration file is unreadable or invalid."""


@dataclass
class ModelConfig:
    weights: str = "models/yolo11n.pt"
    confidence: float = 0.35
    imgsz: int = 640
    device: str = "cpu"
    tracker: str = "bytetrack.yaml"
    vehicle_classes: List[str] = field(default_factory=lambda: ["car", "motorcycle", "bus", "truck"])


@dataclass
class VideoConfig:
    frame_stride: int = 2
    resize_width: int = 960
    max_seconds: float = 0
    default_fps: float = 25.0
    max_upload_mb: int = 500
    allowed_extensions: List[str] = field(default_factory=lambda: ["mp4", "avi", "mov", "mkv"])


@dataclass
class WrongWayConfig:
    enabled: bool = True
    expected_direction: str = "right"
    window_points: int = 10
    min_observations: int = 6
    min_displacement_px: float = 40.0
    persistence_frames: int = 3
    cooldown_s: float = 20.0


@dataclass
class StoppedConfig:
    enabled: bool = True
    stop_seconds: float = 10.0
    move_threshold_px: float = 8.0
    min_track_age_s: float = 2.0
    cooldown_s: float = 30.0
    zone: List[List[float]] = field(default_factory=list)


@dataclass
class ProximityConfig:
    enabled: bool = True
    distance_px: float = 55.0
    min_iou: float = 0.10
    persistence_frames: int = 3
    cooldown_s: float = 10.0


@dataclass
class CongestionConfig:
    enabled: bool = True
    count_threshold: float = 12.0
    window_s: float = 5.0
    sustain_s: float = 5.0
    cooldown_s: float = 60.0


@dataclass
class StorageConfig:
    db_path: str = "data/events.db"
    outputs_dir: str = "outputs"
    upload_dir: str = "data/uploads"
    save_event_images: bool = False


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    wrong_way: WrongWayConfig = field(default_factory=WrongWayConfig)
    stopped_vehicle: StoppedConfig = field(default_factory=StoppedConfig)
    proximity: ProximityConfig = field(default_factory=ProximityConfig)
    congestion: CongestionConfig = field(default_factory=CongestionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    source_id: str = "Recorded video"
    warnings: List[str] = field(default_factory=list, repr=False)

    def copy(self) -> "AppConfig":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("warnings", None)
        return d


_SECTIONS = {
    "model": ModelConfig,
    "video": VideoConfig,
    "wrong_way": WrongWayConfig,
    "stopped_vehicle": StoppedConfig,
    "proximity": ProximityConfig,
    "congestion": CongestionConfig,
    "storage": StorageConfig,
}


def _build_section(cls, data: Any, name: str, warnings: List[str]):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"Section '{name}' must be a mapping, got {type(data).__name__}.")
    known = {f.name for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key in known:
            kwargs[key] = value
        else:
            warnings.append(f"Unknown setting '{name}.{key}' was ignored.")
    return cls(**kwargs)


def from_dict(data: Optional[Dict[str, Any]]) -> AppConfig:
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError("The configuration file must contain a YAML mapping.")
    warnings: List[str] = []
    kwargs: Dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        kwargs[name] = _build_section(cls, data.get(name), name, warnings)
    for key in data:
        if key not in _SECTIONS and key != "source_id":
            warnings.append(f"Unknown top-level setting '{key}' was ignored.")
    if "source_id" in data:
        kwargs["source_id"] = str(data["source_id"])
    cfg = AppConfig(**kwargs, warnings=warnings)
    problems = validate(cfg)
    if problems:
        raise ConfigError("Invalid configuration:\n- " + "\n- ".join(problems))
    return cfg


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(cfg: AppConfig) -> List[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errs: List[str] = []

    def rng(name, value, lo, hi=None):
        if not _is_num(value):
            errs.append(f"{name} must be a number (got {value!r}).")
        elif value < lo or (hi is not None and value > hi):
            errs.append(f"{name} must be between {lo} and {hi if hi is not None else 'infinity'} (got {value}).")

    m, v = cfg.model, cfg.video
    rng("model.confidence", m.confidence, 0.01, 0.99)
    rng("model.imgsz", m.imgsz, 32, 4096)
    if not isinstance(m.weights, str) or not m.weights.strip():
        errs.append("model.weights must be a non-empty path or model name.")
    if not isinstance(m.vehicle_classes, list) or not m.vehicle_classes:
        errs.append("model.vehicle_classes must be a non-empty list of class names.")
    rng("video.frame_stride", v.frame_stride, 1, 30)
    rng("video.resize_width", v.resize_width, 0, 8000)
    rng("video.max_seconds", v.max_seconds, 0)
    rng("video.default_fps", v.default_fps, 1, 240)
    rng("video.max_upload_mb", v.max_upload_mb, 1, 20000)
    if not v.allowed_extensions:
        errs.append("video.allowed_extensions must not be empty.")

    w = cfg.wrong_way
    if w.expected_direction not in DIRECTIONS:
        errs.append(f"wrong_way.expected_direction must be one of {DIRECTIONS} (got {w.expected_direction!r}).")
    rng("wrong_way.window_points", w.window_points, 2, 200)
    rng("wrong_way.min_observations", w.min_observations, 2, 200)
    if _is_num(w.window_points) and _is_num(w.min_observations) and w.min_observations > w.window_points:
        errs.append("wrong_way.min_observations cannot exceed wrong_way.window_points.")
    rng("wrong_way.min_displacement_px", w.min_displacement_px, 1)
    rng("wrong_way.persistence_frames", w.persistence_frames, 1, 100)
    rng("wrong_way.cooldown_s", w.cooldown_s, 0)

    s = cfg.stopped_vehicle
    rng("stopped_vehicle.stop_seconds", s.stop_seconds, 0.5)
    rng("stopped_vehicle.move_threshold_px", s.move_threshold_px, 0.5)
    rng("stopped_vehicle.min_track_age_s", s.min_track_age_s, 0)
    rng("stopped_vehicle.cooldown_s", s.cooldown_s, 0)
    zone_ok = isinstance(s.zone, list) and (len(s.zone) == 0 or len(s.zone) >= 3)
    if zone_ok:
        for pt in s.zone:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2 and all(_is_num(c) and 0 <= c <= 1 for c in pt)):
                zone_ok = False
    if not zone_ok:
        errs.append("stopped_vehicle.zone must be empty or a list of at least 3 [x, y] fractions between 0 and 1.")

    p = cfg.proximity
    rng("proximity.distance_px", p.distance_px, 1)
    rng("proximity.min_iou", p.min_iou, 0, 1)
    rng("proximity.persistence_frames", p.persistence_frames, 1, 100)
    rng("proximity.cooldown_s", p.cooldown_s, 0)

    c = cfg.congestion
    rng("congestion.count_threshold", c.count_threshold, 1)
    rng("congestion.window_s", c.window_s, 0.5)
    rng("congestion.sustain_s", c.sustain_s, 0)
    rng("congestion.cooldown_s", c.cooldown_s, 0)
    return errs


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load configuration. A missing file yields defaults (with a warning)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        cfg = AppConfig()
        cfg.warnings.append(f"Config file {path} not found; built-in defaults are in use.")
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc
    return from_dict(data)


def save_config(cfg: AppConfig, path: Optional[Path] = None) -> Path:
    """Write the configuration to YAML (comments in the original file are not preserved)."""
    problems = validate(cfg)
    if problems:
        raise ConfigError("Refusing to save invalid configuration:\n- " + "\n- ".join(problems))
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, default_flow_style=None)
    except OSError as exc:
        raise ConfigError(f"Could not write {path}: {exc}") from exc
    return path


def zone_to_pixels(zone: List[List[float]], frame_w: int, frame_h: int) -> List[Tuple[float, float]]:
    return [(x * frame_w, y * frame_h) for x, y in zone]


def resolved(cfg_path: str) -> Path:
    return resolve_path(cfg_path)
