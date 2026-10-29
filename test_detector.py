import unittest

import numpy as np

from traffic_anomaly.config import ModelConfig
from traffic_anomaly.datatypes import FramePacket
from traffic_anomaly.detector import DetectionError, ModelLoadError, YoloVehicleDetector, model_status


class T:  # minimal stand-in for a torch tensor
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeResult:
    names = {2: "car", 5: "bus", 7: "truck"}

    def __init__(self, with_ids=True, empty=False):
        if empty:
            self.boxes = None
            return

        class B:
            pass
        b = B()
        b.xyxy = T([[10, 20, 110, 70], [200, 50, 320, 140]])
        b.cls = T([2, 5])
        b.conf = T([0.9, 0.5])
        b.id = T([7, 8]) if with_ids else None
        b.__class__.__len__ = lambda self: 2
        self.boxes = b


class FakeModel:
    names = {0: "person", 2: "car", 5: "bus", 7: "truck"}

    def __init__(self, result=None, fail=False):
        self.result, self.fail, self.kwargs = result, fail, None

    def track(self, image, **kwargs):
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        self.kwargs = kwargs
        return [self.result]


def packet():
    return FramePacket(1, 0.1, np.zeros((100, 100, 3), dtype=np.uint8))


class DetectorTests(unittest.TestCase):
    def test_missing_weights_gives_clear_error(self):
        det = YoloVehicleDetector(ModelConfig(weights="models/definitely_not_here.pt"))
        with self.assertRaises(ModelLoadError) as cm:
            det.load()
        self.assertIn("not found", str(cm.exception))
        self.assertFalse(model_status(ModelConfig(weights="models/definitely_not_here.pt"))["weights_exist"])

    def test_detect_before_load(self):
        with self.assertRaises(DetectionError):
            YoloVehicleDetector(ModelConfig()).detect(packet())

    def test_output_structure(self):
        det = YoloVehicleDetector(ModelConfig())
        det.model, det.class_ids = FakeModel(FakeResult()), [2, 5, 7]
        out = det.detect(packet())
        self.assertEqual([(d.track_id, d.label) for d in out], [(7, "car"), (8, "bus")])
        self.assertAlmostEqual(out[0].confidence, 0.9, places=5)
        self.assertEqual(out[0].bbox, (10.0, 20.0, 110.0, 70.0))
        self.assertEqual(det.model.kwargs["classes"], [2, 5, 7])
        self.assertEqual(det.model.kwargs["conf"], 0.35)
        self.assertTrue(det.model.kwargs["persist"])
        self.assertFalse(det.is_simulated)

    def test_missing_ids_become_minus_one(self):
        det = YoloVehicleDetector(ModelConfig())
        det.model, det.class_ids = FakeModel(FakeResult(with_ids=False)), [2]
        self.assertEqual([d.track_id for d in det.detect(packet())], [-1, -1])

    def test_no_detections(self):
        det = YoloVehicleDetector(ModelConfig())
        det.model = FakeModel(FakeResult(empty=True))
        self.assertEqual(det.detect(packet()), [])

    def test_inference_failure_wrapped(self):
        det = YoloVehicleDetector(ModelConfig())
        det.model = FakeModel(fail=True)
        with self.assertRaises(DetectionError) as cm:
            det.detect(packet())
        self.assertIn("CUDA out of memory", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
