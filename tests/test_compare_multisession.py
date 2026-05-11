from __future__ import annotations

import unittest

import numpy as np

from edgetam_batched.compare_multisession import compare_outputs, select_object_outputs


class CompareMultisessionTests(unittest.TestCase):
    def test_identical_outputs_pass(self):
        mask = np.ones((1, 2, 2), dtype=bool)
        outputs = type("Outputs", (), {})()
        outputs.masks = [[mask, mask, mask]]
        outputs.logits = [[mask.astype(np.float32), mask.astype(np.float32), mask.astype(np.float32)]]
        outputs.object_scores = [[np.ones((1,), dtype=np.float32) for _ in range(3)]]
        outputs.partial = False
        outputs.fallback_backend = None
        outputs.blockers = []
        metrics = compare_outputs(outputs, outputs)
        self.assertTrue(metrics["correctness_pass"])

    def test_candidate_mask_is_resized_to_reference_shape(self):
        ref_mask = np.zeros((1, 4, 4), dtype=bool)
        ref_mask[:, :2, :2] = True
        cand_mask = np.zeros((1, 2, 2), dtype=bool)
        cand_mask[:, 0, 0] = True
        reference = type("Outputs", (), {})()
        reference.masks = [[ref_mask, ref_mask, ref_mask]]
        reference.logits = [[ref_mask.astype(np.float32), ref_mask.astype(np.float32), ref_mask.astype(np.float32)]]
        reference.object_scores = [[np.ones((1,), dtype=np.float32) for _ in range(3)]]
        candidate = type("Outputs", (), {})()
        candidate.masks = [[cand_mask, cand_mask, cand_mask]]
        candidate.logits = [[cand_mask.astype(np.float32), cand_mask.astype(np.float32), cand_mask.astype(np.float32)]]
        candidate.object_scores = [[np.ones((1,), dtype=np.float32) for _ in range(3)]]
        candidate.partial = False
        candidate.fallback_backend = None
        candidate.blockers = []

        metrics = compare_outputs(reference, candidate)

        self.assertTrue(metrics["correctness_pass"])

    def test_speed_first_gate_can_relax_strict_thresholds(self):
        full = np.ones((1, 2, 2), dtype=bool)
        empty = np.zeros((1, 2, 2), dtype=bool)
        reference = type("Outputs", (), {})()
        reference.masks = [[full, full, full], [empty, full, full]]
        reference.logits = [
            [full.astype(np.float32), full.astype(np.float32), full.astype(np.float32)],
            [empty.astype(np.float32), full.astype(np.float32), full.astype(np.float32)],
        ]
        reference.object_scores = [[np.ones((1,), dtype=np.float32) for _ in range(3)] for _ in range(2)]
        candidate = type("Outputs", (), {})()
        candidate.masks = [[full, full, full], [full, full, full]]
        candidate.logits = [[full.astype(np.float32), full.astype(np.float32), full.astype(np.float32)] for _ in range(2)]
        candidate.object_scores = [[np.ones((1,), dtype=np.float32) for _ in range(3)] for _ in range(2)]
        candidate.partial = False
        candidate.fallback_backend = None
        candidate.blockers = []

        strict = compare_outputs(reference, candidate)
        speed_first = compare_outputs(
            reference,
            candidate,
            correctness_gate="speed-first",
            min_global_iou_avg=0.6,
            min_global_iou_p50=0.5,
            max_empty_mismatch_count=1,
        )

        self.assertFalse(strict["correctness_pass"])
        self.assertTrue(speed_first["correctness_pass"])
        self.assertTrue(speed_first["correctness_gate"]["speed_first"])

    def test_select_single_object_keeps_last_object_plane(self):
        controller = np.zeros((2, 2), dtype=bool)
        obj = np.ones((2, 2), dtype=bool)
        outputs = type("Outputs", (), {})()
        outputs.masks = [[np.stack([controller, obj], axis=0)]]
        outputs.logits = [[np.stack([controller, obj], axis=0).astype(np.float32)]]
        outputs.object_scores = [[np.array([0.1, 0.9], dtype=np.float32)]]
        outputs.partial = False
        outputs.fallback_backend = None
        outputs.blockers = []

        selected = select_object_outputs(outputs, object_count=1)

        self.assertEqual(selected.masks[0][0].shape, (1, 2, 2))
        self.assertEqual(int(selected.masks[0][0].sum()), 4)
        self.assertAlmostEqual(float(selected.object_scores[0][0][0]), 0.9, places=6)


if __name__ == "__main__":
    unittest.main()
