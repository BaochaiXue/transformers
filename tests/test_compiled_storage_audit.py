import unittest

import torch

from edgetam_batched.storage_alias_audit import tensor_storage_ids


class CompiledStorageAuditTests(unittest.TestCase):
    def test_tensor_storage_ids_marks_views(self):
        base = torch.zeros(4, 4)
        view = base[:2]
        rows = tensor_storage_ids({"view": view})
        self.assertEqual(rows[0]["path"], "view")
        self.assertTrue(rows[0]["is_view"])


if __name__ == "__main__":
    unittest.main()
