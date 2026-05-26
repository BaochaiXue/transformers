import unittest

import numpy as np

from edgetam_batched.compiled_vs_eager_debug import find_first_compiled_regression, summarize_three_way


class _Outputs:
    def __init__(self, masks):
        self.masks = masks


def _mask(x0, x1):
    arr = np.zeros((1, 4, 4), dtype=bool)
    arr[0, 1:3, x0:x1] = True
    return arr


class CompiledVsEagerDebugTests(unittest.TestCase):
    def test_find_first_compiled_regression(self):
        ref = _Outputs([[_mask(1, 3)], [_mask(1, 3)]])
        eager = _Outputs([[_mask(1, 3)], [_mask(1, 3)]])
        compiled = _Outputs([[_mask(1, 3)], [_mask(0, 2)]])
        first = find_first_compiled_regression(ref, eager, compiled, threshold=0.98)
        self.assertEqual(first["frame_idx"], 1)
        self.assertEqual(first["camera"], "cam0")
        self.assertGreaterEqual(first["iou_hf_public_eager"], 0.98)
        self.assertLess(first["iou_hf_public_compiled"], 0.98)

    def test_summarize_three_way(self):
        ref = _Outputs([[_mask(1, 3)]])
        eager = _Outputs([[_mask(1, 3)]])
        compiled = _Outputs([[_mask(1, 3)]])
        summary = summarize_three_way(ref, eager, compiled)
        self.assertEqual(summary["hf_public_vs_eager"]["avg"], 1.0)
        self.assertEqual(summary["eager_vs_compiled"]["avg"], 1.0)


if __name__ == "__main__":
    unittest.main()
