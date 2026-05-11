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


if __name__ == "__main__":
    unittest.main()
