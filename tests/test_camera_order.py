from __future__ import annotations

import unittest

import torch

from edgetam_batched.camera_order import diagonal_best, iou_matrix


class CameraOrderTests(unittest.TestCase):
    def test_diagonal_best_passes(self):
        masks = [
            torch.tensor([1, 0, 0], dtype=torch.bool),
            torch.tensor([0, 1, 0], dtype=torch.bool),
            torch.tensor([0, 0, 1], dtype=torch.bool),
        ]
        matrix = iou_matrix(masks, masks)
        result = diagonal_best(matrix)
        self.assertTrue(result["pass"])

    def test_swapped_order_fails(self):
        refs = [torch.tensor([1, 0]), torch.tensor([0, 1]), torch.tensor([1, 1])]
        candidates = [refs[1], refs[0], refs[2]]
        result = diagonal_best(iou_matrix(candidates, refs))
        self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main()
