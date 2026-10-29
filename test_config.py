import tempfile
import unittest
from pathlib import Path

from traffic_anomaly.config import (AppConfig, ConfigError, from_dict, load_config, save_config, validate,
                                    DEFAULT_CONFIG_PATH)


class ConfigTests(unittest.TestCase):
    def test_shipped_config_loads_and_is_valid(self):
        cfg = load_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(cfg.model.vehicle_classes, ["car", "motorcycle", "bus", "truck"])
        self.assertEqual(validate(cfg), [])
        self.assertEqual(cfg.warnings, [])

    def test_missing_file_gives_defaults_with_warning(self):
        cfg = load_config(Path(tempfile.gettempdir()) / "definitely_missing_cfg.yaml")
        self.assertTrue(cfg.warnings)
        self.assertEqual(cfg.video.frame_stride, 2)

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = AppConfig()
            cfg.wrong_way.expected_direction = "left"
            cfg.stopped_vehicle.zone = [[0, 0.4], [1, 0.4], [1, 1], [0, 1]]
            p = save_config(cfg, Path(d) / "c.yaml")
            back = load_config(p)
            self.assertEqual(back.wrong_way.expected_direction, "left")
            self.assertEqual(back.stopped_vehicle.zone[0], [0, 0.4])

    def test_invalid_values_rejected(self):
        with self.assertRaises(ConfigError):
            from_dict({"model": {"confidence": 5}})
        with self.assertRaises(ConfigError):
            from_dict({"wrong_way": {"expected_direction": "sideways"}})
        with self.assertRaises(ConfigError):
            from_dict({"wrong_way": {"window_points": 4, "min_observations": 9}})
        with self.assertRaises(ConfigError):
            from_dict({"stopped_vehicle": {"zone": [[0, 0], [1, 1]]}})
        with self.assertRaises(ConfigError):
            from_dict({"video": []})

    def test_unknown_keys_warn(self):
        cfg = from_dict({"model": {"nonsense": 1}, "extra": 2})
        self.assertEqual(len(cfg.warnings), 2)

    def test_malformed_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.yaml"
            p.write_text("model: [unclosed", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(p)

    def test_save_refuses_invalid(self):
        cfg = AppConfig()
        cfg.model.confidence = 0
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ConfigError):
                save_config(cfg, Path(d) / "x.yaml")


if __name__ == "__main__":
    unittest.main()
