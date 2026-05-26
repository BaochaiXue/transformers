from __future__ import annotations

import unittest

from edgetam_batched.stats import percentile, summarize


class ProfileStatsTests(unittest.TestCase):
    def test_percentiles(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(percentile(values, 0), 1)
        self.assertEqual(percentile(values, 50), 3)
        self.assertEqual(percentile(values, 100), 5)
        self.assertAlmostEqual(percentile(values, 90), 4.6)

    def test_summary_empty(self):
        summary = summarize([])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["p50"])


if __name__ == "__main__":
    unittest.main()
