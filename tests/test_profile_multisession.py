from __future__ import annotations

import unittest

from edgetam_batched.profile_multisession import run_scaffold_profile


class ProfileMultisessionTests(unittest.TestCase):
    def test_scaffold_profile_has_stage_summary(self):
        profile = run_scaffold_profile(frames=3, warmup=1)
        self.assertEqual(profile["stage_wall_ms"]["count"], 3)


if __name__ == "__main__":
    unittest.main()
