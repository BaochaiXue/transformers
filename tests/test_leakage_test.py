from __future__ import annotations

import unittest

import torch

from edgetam_batched.leakage_test import compare_cam0_stability


class LeakageTests(unittest.TestCase):
    def test_unchanged_cam0_passes(self):
        mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
        result = compare_cam0_stability(mask, mask.clone())
        self.assertTrue(result["pass"])

    def test_changed_cam0_fails(self):
        a = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
        b = torch.tensor([[0, 1], [1, 0]], dtype=torch.bool)
        result = compare_cam0_stability(a, b)
        self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main()
