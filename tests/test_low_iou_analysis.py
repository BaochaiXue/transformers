import unittest

import numpy as np

from edgetam_batched.analyze_low_iou_frames import (
    bbox,
    center_distance,
    possible_reasons,
)


class LowIouAnalysisTests(unittest.TestCase):
    def test_bbox_uses_exclusive_max_corner(self):
        mask = np.zeros((6, 7), dtype=bool)
        mask[2:5, 1:4] = True
        self.assertEqual(bbox(mask), [1, 2, 4, 5])

    def test_center_distance(self):
        self.assertAlmostEqual(center_distance([0, 0, 10, 10], [10, 0, 20, 10]), 10.0)

    def test_possible_reasons_marks_small_and_shifted(self):
        reasons = possible_reasons(
            ref_area=900,
            candidate_area=800,
            center_distance_px=25.0,
            bbox_area_ratio=2.5,
        )
        self.assertIn("small mask IoU sensitivity", reasons)
        self.assertIn("tracking drift / location shift", reasons)
        self.assertIn("scale mismatch", reasons)


if __name__ == "__main__":
    unittest.main()
