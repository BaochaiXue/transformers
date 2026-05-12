import unittest

import torch

from edgetam_batched.compiled_component_equivalence import _apply_component_precision


class _Model:
    def __init__(self):
        self.memory_attention = torch.nn.Linear(2, 2)
        self.mask_decoder = torch.nn.Linear(2, 2)
        self.prompt_encoder = torch.nn.Linear(2, 2)
        self.memory_encoder = torch.nn.Linear(2, 2)
        self.spatial_perceiver = torch.nn.Linear(2, 2)


class CompiledComponentEquivalenceTests(unittest.TestCase):
    def test_apply_component_precision_converts_related_modules(self):
        model = _Model()
        _apply_component_precision(model, "mask_decoder", torch.float32)
        self.assertEqual(next(model.mask_decoder.parameters()).dtype, torch.float32)
        self.assertEqual(next(model.prompt_encoder.parameters()).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
