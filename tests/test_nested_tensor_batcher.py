from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from edgetam_batched.nested_tensor_batcher import (
    UnsupportedBatchField,
    compare_nested,
    split_nested,
    stack_nested,
)


class NestedTensorBatcherTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_stack_and_split_nested_tensor_dict(self):
        values = [
            {"x": torch.full((2, 3), 1.0), "meta": "same"},
            {"x": torch.full((2, 3), 2.0), "meta": "same"},
            {"x": torch.full((2, 3), 3.0), "meta": "same"},
        ]

        stacked = stack_nested(values)
        self.assertEqual(tuple(stacked["x"].shape), (3, 2, 3))

        split = split_nested(stacked, batch_size=3)
        self.assertTrue(torch.equal(split[1]["x"], values[1]["x"]))
        self.assertEqual(split[2]["meta"], "same")

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_different_metadata_is_rejected(self):
        with self.assertRaises(UnsupportedBatchField):
            stack_nested([{"mode": "a"}, {"mode": "b"}])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_compare_nested_reports_tensor_shape_mismatch(self):
        result = compare_nested(torch.zeros((1, 2)), torch.zeros((2, 2)))
        self.assertFalse(result["pass"])
        self.assertIn("shape mismatch", result["reason"])


if __name__ == "__main__":
    unittest.main()
