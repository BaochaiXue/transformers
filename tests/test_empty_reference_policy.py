from __future__ import annotations

import unittest

import numpy as np

from edgetam_batched.compare_multisession import compare_outputs


def _outputs(ref_mask, cand_mask=None):
    if cand_mask is None:
        cand_mask = ref_mask
    reference = type("Outputs", (), {})()
    reference.masks = [[np.asarray(ref_mask, dtype=bool)]]
    reference.logits = [[np.asarray(ref_mask, dtype=np.float32)]]
    reference.object_scores = [[np.ones((1,), dtype=np.float32)]]
    candidate = type("Outputs", (), {})()
    candidate.masks = [[np.asarray(cand_mask, dtype=bool)]]
    candidate.logits = [[np.asarray(cand_mask, dtype=np.float32)]]
    candidate.object_scores = [[np.ones((1,), dtype=np.float32)]]
    candidate.partial = False
    candidate.fallback_backend = None
    candidate.blockers = []
    return reference, candidate


class EmptyReferencePolicyTests(unittest.TestCase):
    def test_reference_empty_candidate_empty_is_ignored(self):
        empty = np.zeros((1, 2, 2), dtype=bool)
        reference, candidate = _outputs(empty)

        metrics = compare_outputs(reference, candidate, empty_reference_policy="ignore-candidate")

        row = metrics["per_camera_object"]["cam0_obj0"]
        self.assertEqual(row["evaluated_sample_count"], 0)
        self.assertEqual(row["ignored_reference_empty_count"], 1)
        self.assertEqual(row["candidate_nonempty_when_reference_empty_count"], 0)
        self.assertEqual(metrics["empty_mismatch_count"], 0)
        self.assertIsNone(row["mask_iou"]["avg"])

    def test_reference_empty_candidate_nonempty_is_ignored_not_mismatch(self):
        empty = np.zeros((1, 2, 2), dtype=bool)
        full = np.ones((1, 2, 2), dtype=bool)
        reference, candidate = _outputs(empty, full)

        metrics = compare_outputs(reference, candidate, empty_reference_policy="ignore-candidate")

        row = metrics["per_camera_object"]["cam0_obj0"]
        self.assertEqual(row["evaluated_sample_count"], 0)
        self.assertEqual(row["ignored_reference_empty_count"], 1)
        self.assertEqual(row["candidate_nonempty_when_reference_empty_count"], 1)
        self.assertEqual(metrics["empty_mismatch_count"], 0)
        self.assertEqual(metrics["candidate_nonempty_when_reference_empty_count"], 1)

    def test_reference_nonempty_candidate_empty_is_evaluated_mismatch(self):
        empty = np.zeros((1, 2, 2), dtype=bool)
        full = np.ones((1, 2, 2), dtype=bool)
        reference, candidate = _outputs(full, empty)

        metrics = compare_outputs(reference, candidate, empty_reference_policy="ignore-candidate")

        row = metrics["per_camera_object"]["cam0_obj0"]
        self.assertEqual(row["evaluated_sample_count"], 1)
        self.assertEqual(row["empty_mismatch_count"], 1)
        self.assertEqual(metrics["empty_mismatch_count"], 1)
        self.assertEqual(row["mask_iou"]["avg"], 0.0)

    def test_reference_nonempty_candidate_nonempty_computes_iou(self):
        ref = np.zeros((1, 2, 2), dtype=bool)
        ref[:, 0, :] = True
        cand = np.zeros((1, 2, 2), dtype=bool)
        cand[:, :, 0] = True
        reference, candidate = _outputs(ref, cand)

        metrics = compare_outputs(reference, candidate, empty_reference_policy="ignore-candidate")

        row = metrics["per_camera_object"]["cam0_obj0"]
        self.assertEqual(row["evaluated_sample_count"], 1)
        self.assertAlmostEqual(row["mask_iou"]["avg"], 1.0 / 3.0)

    def test_object_without_evaluated_samples_is_reference_uncertain(self):
        empty = np.zeros((1, 2, 2), dtype=bool)
        reference, candidate = _outputs(empty)

        metrics = compare_outputs(reference, candidate, empty_reference_policy="ignore-candidate")

        obj = metrics["object_summaries"]["obj0"]
        self.assertFalse(obj["object_validated"])
        self.assertTrue(obj["object_reference_absent"])
        self.assertTrue(obj["reference_uncertain"])


if __name__ == "__main__":
    unittest.main()
