from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from edgetam_batched.sam31_replay_reference import (
    build_sam31_replay_case,
    initial_masks_by_camera_from_sam31,
    load_sam31_reference_outputs,
)


class Sam31ReplayReferenceTests(unittest.TestCase):
    def test_case_builder_and_loader_preserve_controller_object_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "replay"
            for cam_idx in range(3):
                (replay / f"cam{cam_idx}").mkdir(parents=True)
                image = Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8))
                image.save(replay / f"cam{cam_idx}" / "frame_000000.png")
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

            case_root = build_sam31_replay_case(rgb_replay=replay)
            self.assertTrue((case_root / "color" / "0" / "frame_000000.png").exists())

            mask_root = replay / "sam31_video_reference_masks"
            (mask_root / "mask").mkdir(parents=True)
            for cam_idx in range(3):
                (mask_root / "mask" / str(cam_idx) / "11").mkdir(parents=True)
                (mask_root / "mask" / str(cam_idx) / "22").mkdir(parents=True)
                (mask_root / "mask" / f"mask_info_{cam_idx}.json").write_text(
                    json.dumps({"11": "towel", "22": "stuffed animal"}),
                    encoding="utf-8",
                )
                controller = np.zeros((4, 5), dtype=np.uint8)
                controller[0:2, 0:2] = 255
                obj = np.zeros((4, 5), dtype=np.uint8)
                obj[2:4, 3:5] = 255
                Image.fromarray(controller).save(mask_root / "mask" / str(cam_idx) / "11" / "frame_000000.png")
                Image.fromarray(obj).save(mask_root / "mask" / str(cam_idx) / "22" / "frame_000000.png")

            outputs = load_sam31_reference_outputs(rgb_replay=replay, mask_root=mask_root, frames=1)
            self.assertEqual(outputs.backend, "sam31_video_replay_ref")
            self.assertEqual(outputs.masks[0][0].shape, (2, 4, 5))
            self.assertEqual(int(outputs.masks[0][0][0].sum()), 4)
            self.assertEqual(int(outputs.masks[0][0][1].sum()), 4)

            initial = initial_masks_by_camera_from_sam31(rgb_replay=replay, mask_root=mask_root)
            self.assertEqual(len(initial), 3)
            self.assertEqual(int(initial[0][0].sum()), 4)
            self.assertEqual(int(initial[0][1].sum()), 4)


if __name__ == "__main__":
    unittest.main()
