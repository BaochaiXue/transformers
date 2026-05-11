from __future__ import annotations

import unittest

from edgetam_batched.batched_multisession_runtime import split_hf_vision_features_for_session


class DummyOutputs:
    def __init__(self):
        import torch

        self.fpn_hidden_states = [torch.zeros((4, 3, 5)), torch.zeros((6, 3, 7))]
        self.fpn_position_encoding = [torch.zeros((4, 3, 5)), torch.zeros((6, 3, 7))]


class BatchedRuntimeShapeTests(unittest.TestCase):
    def test_split_features_preserves_single_camera_batch(self):
        out = split_hf_vision_features_for_session(DummyOutputs(), 1)
        self.assertEqual(tuple(out["vision_feats"][0].shape), (4, 1, 5))
        self.assertEqual(tuple(out["vision_pos_embeds"][1].shape), (6, 1, 7))


if __name__ == "__main__":
    unittest.main()
