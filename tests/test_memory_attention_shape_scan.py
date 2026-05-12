import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from edgetam_batched.onnx_trt.scan_memory_attention_shapes import scan_fixtures


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class MemoryAttentionShapeScanTests(unittest.TestCase):
    def test_scan_groups_by_shape_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fixtures"
            frame = root / "frame_001"
            frame.mkdir(parents=True)
            for cam in range(3):
                torch.save(
                    {
                        "component": "memory_attention",
                        "camera_idx": cam,
                        "frame_idx": 1,
                        "args": (),
                        "kwargs": {
                            "current_vision_features": torch.zeros(4096, 1, 256),
                            "current_vision_position_embeddings": torch.zeros(4096, 1, 256),
                            "memory": torch.zeros(516, 1, 64),
                            "memory_posision_embeddings": torch.zeros(516, 1, 64),
                            "num_object_pointer_tokens": 4,
                            "num_spatial_memory_tokens": 1,
                        },
                        "output": torch.zeros(1, 1, 4096, 256),
                    },
                    frame / f"cam{cam}_memory_attention.pt",
                )

            payload = scan_fixtures(root)

        self.assertEqual(len(payload["shape_buckets"]), 1)
        bucket = next(iter(payload["shape_buckets"].values()))
        self.assertEqual(bucket["records"], 3)
        self.assertEqual(bucket["num_object_pointer_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
