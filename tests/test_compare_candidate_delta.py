from __future__ import annotations

import unittest

from edgetam_batched.compare_candidate_delta import build_delta_report


class CompareCandidateDeltaTests(unittest.TestCase):
    def test_delta_uses_evaluated_samples_and_marks_reference_uncertain(self):
        baseline = {
            "backend": "hf_ref_seq_public",
            "metrics": {
                "empty_reference_policy": "ignore-candidate",
                "per_camera_object": {
                    "cam0_obj0": {
                        "evaluated_sample_count": 0,
                        "ignored_reference_empty_count": 10,
                        "candidate_nonempty_when_reference_empty_count": 0,
                        "mask_iou": {"avg": None},
                        "reference_uncertain": True,
                    },
                    "cam0_obj1": {
                        "evaluated_sample_count": 10,
                        "ignored_reference_empty_count": 0,
                        "candidate_nonempty_when_reference_empty_count": 0,
                        "mask_iou": {"avg": 0.96},
                    },
                },
            },
        }
        candidate = {
            "backend": "hf_batch_vision_seq_session",
            "compile_mode": "reduce-overhead",
            "metrics": {
                "empty_reference_policy": "ignore-candidate",
                "per_camera_object": {
                    "cam0_obj0": {
                        "evaluated_sample_count": 0,
                        "ignored_reference_empty_count": 10,
                        "candidate_nonempty_when_reference_empty_count": 2,
                        "mask_iou": {"avg": None},
                        "reference_uncertain": True,
                    },
                    "cam0_obj1": {
                        "evaluated_sample_count": 10,
                        "ignored_reference_empty_count": 0,
                        "candidate_nonempty_when_reference_empty_count": 0,
                        "mask_iou": {"avg": 0.955},
                    },
                },
            },
        }

        delta = build_delta_report(baseline, candidate)

        self.assertFalse(delta["evaluated_subset_mismatch"])
        self.assertFalse(delta["per_object"]["obj0"]["delta_valid"])
        self.assertEqual(delta["per_object"]["obj0"]["reason"], "no evaluated SAM3.1 reference samples")
        self.assertTrue(delta["per_object"]["obj0"]["reference_uncertain"])
        self.assertTrue(delta["per_object"]["obj1"]["not_worse"])
        self.assertAlmostEqual(delta["per_object"]["obj1"]["delta"], -0.005)

    def test_evaluated_subset_mismatch_is_reported(self):
        baseline = {
            "metrics": {"per_camera_object": {"cam0_obj1": {"evaluated_sample_count": 3, "mask_iou": {"avg": 0.9}}}}
        }
        candidate = {
            "metrics": {"per_camera_object": {"cam0_obj1": {"evaluated_sample_count": 2, "mask_iou": {"avg": 0.9}}}}
        }

        delta = build_delta_report(baseline, candidate)

        self.assertTrue(delta["evaluated_subset_mismatch"])


if __name__ == "__main__":
    unittest.main()
