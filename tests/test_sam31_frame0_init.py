from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from edgetam_batched.sam31_frame0_init import (
    mask_stats,
    select_and_union_masks,
    write_frame0_init_mask_root,
)
from edgetam_batched.sam31_replay_reference import initial_masks_by_camera_from_sam31


class Sam31Frame0InitTests(unittest.TestCase):
    def test_green_score_prefers_green_controller_candidate(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:, :] = [120, 110, 105]
        image[0:5, 0:5] = [40, 220, 40]
        image[5:10, 5:10] = [200, 80, 80]
        green_mask = np.zeros((10, 10), dtype=bool)
        green_mask[0:5, 0:5] = True
        red_mask = np.zeros((10, 10), dtype=bool)
        red_mask[5:10, 5:10] = True

        selected = select_and_union_masks(
            [red_mask, green_mask],
            Image.fromarray(image),
            mode="green-score",
            min_area=1,
            max_instances=1,
        )

        self.assertEqual(selected["indices"], [1])
        self.assertEqual(int(selected["mask"].sum()), 25)
        stats = mask_stats(selected["mask"], image)
        self.assertGreater(stats["green_fraction"], 0.9)

    def test_mask_root_schema_loads_controller_then_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay"
            for cam_idx in range(3):
                (replay / f"cam{cam_idx}").mkdir(parents=True)
                Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(
                    replay / f"cam{cam_idx}" / "frame_000000.png"
                )
            (replay / "manifest.json").write_text(
                json.dumps(
                    {
                        "camera_count": 3,
                        "frame_count": 1,
                        "width": 5,
                        "height": 4,
                        "fps": 15,
                        "source": "test",
                    }
                ),
                encoding="utf-8",
            )
            selected = {}
            for cam_idx in range(3):
                controller = np.zeros((4, 5), dtype=bool)
                controller[0:2, 0:2] = True
                obj = np.zeros((4, 5), dtype=bool)
                obj[2:4, 3:5] = True
                selected[cam_idx] = {"controller": controller, "object": obj}

            write_frame0_init_mask_root(
                rgb_replay=replay,
                output_dir=replay / "sam31_image_frame0_init_masks",
                selected_masks=selected,
                controller_label="green towel",
                object_label="stuffed animal",
            )
            initial = initial_masks_by_camera_from_sam31(
                rgb_replay=replay,
                mask_root=replay / "sam31_image_frame0_init_masks",
                controller_prompt="green towel",
                object_prompt="stuffed animal",
            )

            self.assertEqual(len(initial), 3)
            self.assertEqual(int(initial[0][0].sum()), 4)
            self.assertEqual(int(initial[0][1].sum()), 4)


if __name__ == "__main__":
    unittest.main()
