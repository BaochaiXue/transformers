from __future__ import annotations

import unittest

import numpy as np

from edgetam_batched.find_first_bad_frame import classify_first_divergence, tensor_diff_summary


class FindFirstBadFrameTests(unittest.TestCase):
    def test_tensor_diff_summary_reports_abs_stats(self):
        ref = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        cand = np.array([0.0, 2.0, 4.0], dtype=np.float32)

        summary = tensor_diff_summary(ref, cand)

        self.assertTrue(summary["shape_match"])
        self.assertEqual(summary["max_abs_diff"], 2.0)
        self.assertAlmostEqual(summary["mean_abs_diff"], 1.0)

    def test_tensor_diff_summary_reports_shape_mismatch(self):
        summary = tensor_diff_summary(np.zeros((1, 2)), np.zeros((2, 1)))

        self.assertFalse(summary["shape_match"])
        self.assertEqual(summary["reference_shape"], [1, 2])
        self.assertEqual(summary["candidate_shape"], [2, 1])

    def test_classify_first_divergence_prioritizes_masks(self):
        label = classify_first_divergence(
            {
                "maskmem_features": {"p95_abs_diff": 10.0, "shape_match": True},
                "pred_masks": {"p95_abs_diff": 0.5, "shape_match": True},
            }
        )

        self.assertEqual(label, "mask_decoder_or_accumulated_state")

    def test_classify_first_divergence_detects_memory_encoder(self):
        label = classify_first_divergence(
            {
                "pred_masks": {"p95_abs_diff": 0.0, "shape_match": True},
                "maskmem_features": {"p95_abs_diff": 0.25, "shape_match": True},
            }
        )

        self.assertEqual(label, "memory_encoder_or_memory_state_recurrence")


if __name__ == "__main__":
    unittest.main()
