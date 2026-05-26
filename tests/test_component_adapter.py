from __future__ import annotations

import unittest

import torch

from edgetam_batched.component_adapter import EdgeTamComponentAdapter


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = torch.nn.Identity()
        self.memory_attention = torch.nn.Identity()
        self.mask_decoder = torch.nn.Identity()
        self.memory_encoder = torch.nn.Identity()
        self.prompt_encoder = torch.nn.Identity()

    def get_image_features(self, pixel_values, return_dict=True):
        return pixel_values


class ComponentAdapterTests(unittest.TestCase):
    def test_direct_component_access(self):
        records = EdgeTamComponentAdapter(DummyModel()).report()
        by_name = {record.name: record for record in records}
        self.assertTrue(by_name["vision_encoder"].found)
        self.assertTrue(by_name["vision_encoder"].can_batch)
        self.assertTrue(by_name["memory_attention"].found)


if __name__ == "__main__":
    unittest.main()
