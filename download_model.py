"""Download the YOLO weights named in config/settings.yaml (model.weights) via Ultralytics.

Usage (from the project folder, virtual environment active):
    python scripts/download_model.py

Needs internet access once. Ultralytics models are AGPL-3.0 licensed (or available under an
Ultralytics enterprise licence) - check that this suits your use. The weights are NOT bundled.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from traffic_anomaly.config import ConfigError, load_config  # noqa: E402
from traffic_anomaly.detector import ModelLoadError, download_weights  # noqa: E402


def main() -> int:
    try:
        cfg = load_config()
        path = download_weights(cfg.model)
    except (ConfigError, ModelLoadError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Model ready: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
