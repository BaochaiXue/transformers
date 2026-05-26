from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from edgetam_batched.sam31_video_text_box_reference import (
    _write_role_video_segments,
    mask_to_normalized_xywh,
)


class Sam31VideoTextBoxReferenceTests(unittest.TestCase):
    def test_mask_to_normalized_xywh_uses_tight_exclusive_box(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 3:13] = True

        box = mask_to_normalized_xywh(mask)

        self.assertEqual(box, [3 / 20, 2 / 10, 10 / 20, 6 / 10])

    def test_mask_to_normalized_xywh_rejects_empty_mask(self):
        with self.assertRaises(ValueError):
            mask_to_normalized_xywh(np.zeros((4, 5), dtype=bool))

    def test_role_writer_fills_missing_frames_with_empty_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.zeros((4, 5), dtype=bool)
            mask[1:3, 2:5] = True

            stats = _write_role_video_segments(
                output_root=root,
                camera_idx=1,
                role_idx=0,
                frame_token_by_index={0: "frame_000000", 1: "frame_000001"},
                video_segments={0: {0: mask}},
            )

            self.assertEqual(stats["saved_frame_count"], 2)
            self.assertEqual(stats["nonempty_frame_count"], 1)
            self.assertEqual(stats["frame0_area"], 6)
            frame0 = np.asarray(Image.open(root / "mask" / "1" / "0" / "frame_000000.png").convert("L")) > 0
            frame1 = np.asarray(Image.open(root / "mask" / "1" / "0" / "frame_000001.png").convert("L")) > 0
            self.assertEqual(int(frame0.sum()), 6)
            self.assertEqual(int(frame1.sum()), 0)


if __name__ == "__main__":
    unittest.main()
