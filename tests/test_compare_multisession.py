from __future__ import annotations

import unittest

import numpy as np

from edgetam_batched.compare_multisession import compare_outputs


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


if __name__ == "__main__":
    unittest.main()
