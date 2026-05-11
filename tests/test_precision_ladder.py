import unittest

from edgetam_batched.precision_ladder import precision_mode_status


class PrecisionLadderTests(unittest.TestCase):
    def test_implemented_modes(self):
        self.assertTrue(precision_mode_status("all_bf16")["implemented"])
        self.assertTrue(precision_mode_status("all_fp32")["implemented"])

    def test_selective_modes_are_implemented_hooks(self):
        status = precision_mode_status("memory_path_fp32")
        self.assertTrue(status["implemented"])
        self.assertIsNone(status["reason"])


if __name__ == "__main__":
    unittest.main()
