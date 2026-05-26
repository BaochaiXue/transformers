import unittest

import torch

from edgetam_batched.storage_alias_audit import tensor_storage_ids


class StorageAliasAuditTests(unittest.TestCase):
    def test_tensor_storage_ids_nested(self):
        tensor = torch.zeros(2, 3)
        payload = {"a": tensor, "b": [tensor.clone()]}
        ids = tensor_storage_ids(payload)
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(item["storage_ptr"] for item in ids))
        self.assertIn("a", {item["path"] for item in ids})
        self.assertIn("b[0]", {item["path"] for item in ids})


if __name__ == "__main__":
    unittest.main()
