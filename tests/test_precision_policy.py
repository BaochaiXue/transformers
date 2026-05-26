import unittest

from edgetam_batched.precision_policy import resolve_precision_policy


class PrecisionPolicyTests(unittest.TestCase):
    def test_memory_path_fp32_sets_recurrent_path(self):
        policy = resolve_precision_policy("memory_path_fp32")
        self.assertEqual(policy.memory_attention_dtype, "fp32")
        self.assertEqual(policy.mask_decoder_dtype, "fp32")
        self.assertEqual(policy.memory_encoder_dtype, "fp32")
        self.assertEqual(policy.store_object_pointer_dtype, "fp32")
        self.assertEqual(policy.store_maskmem_features_dtype, "fp32")

    def test_all_bf16_leaves_everything_bf16(self):
        policy = resolve_precision_policy("all_bf16")
        self.assertEqual(policy.vision_dtype, "bf16")
        self.assertEqual(policy.memory_attention_dtype, "bf16")
        self.assertFalse(policy.uses_fp32)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            resolve_precision_policy("unknown")


if __name__ == "__main__":
    unittest.main()
