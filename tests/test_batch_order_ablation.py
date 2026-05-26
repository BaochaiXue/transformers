import unittest

from edgetam_batched.batch_order_ablation import parse_order


class BatchOrderAblationTests(unittest.TestCase):
    def test_parse_order(self):
        self.assertEqual(parse_order("0,1,2"), (0, 1, 2))
        self.assertEqual(parse_order("1, 1, 1"), (1, 1, 1))

    def test_parse_order_rejects_empty(self):
        with self.assertRaises(ValueError):
            parse_order("")


if __name__ == "__main__":
    unittest.main()
