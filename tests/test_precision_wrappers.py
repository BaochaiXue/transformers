import unittest

import torch

from edgetam_batched.precision_wrappers import cast_nested_floating, detach_clone_to_dtype


class PrecisionWrapperTests(unittest.TestCase):
    def test_cast_nested_floating(self):
        value = {"x": torch.ones(2, dtype=torch.bfloat16), "n": torch.ones(2, dtype=torch.int64)}
        out = cast_nested_floating(value, torch.float32)
        self.assertEqual(out["x"].dtype, torch.float32)
        self.assertEqual(out["n"].dtype, torch.int64)

    def test_detach_clone_to_dtype_breaks_alias(self):
        value = torch.ones(2, dtype=torch.bfloat16)
        out = detach_clone_to_dtype(value, torch.float32)
        self.assertEqual(out.dtype, torch.float32)
        self.assertNotEqual(out.data_ptr(), value.data_ptr())
        self.assertFalse(out.requires_grad)


if __name__ == "__main__":
    unittest.main()
