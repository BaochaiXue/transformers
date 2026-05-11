from __future__ import annotations

import unittest

from edgetam_batched.state_drift_curve import first_iou_below, first_tensor_drift, first_threshold_crossing


class StateDriftCurveTests(unittest.TestCase):
    def test_first_iou_below_returns_frame_and_camera(self):
        rows = [
            {"frame_idx": 0, "camera": "cam0", "mask_iou": 0.99},
            {"frame_idx": 3, "camera": "cam1", "mask_iou": 0.7},
        ]

        self.assertEqual(first_iou_below(rows, 0.9)["frame_idx"], 3)

    def test_first_threshold_crossing_uses_requested_key(self):
        rows = [
            {"frame_idx": 0, "camera": "cam0", "x": 0.0},
            {"frame_idx": 1, "camera": "cam2", "x": 1.5},
        ]

        result = first_threshold_crossing(rows, "x", 1.0)

        self.assertEqual(result["camera"], "cam2")
        self.assertEqual(result["value"], 1.5)

    def test_first_tensor_drift_reports_field(self):
        rows = [
            {"frame_idx": 0, "camera": "cam0", "pred_masks_p95_abs_diff": 0.0, "maskmem_features_p95_abs_diff": 0.0},
            {"frame_idx": 2, "camera": "cam1", "pred_masks_p95_abs_diff": 0.0, "maskmem_features_p95_abs_diff": 0.2},
        ]

        result = first_tensor_drift(rows, threshold=0.1)

        self.assertEqual(result["frame_idx"], 2)
        self.assertEqual(result["field"], "maskmem_features")


if __name__ == "__main__":
    unittest.main()
