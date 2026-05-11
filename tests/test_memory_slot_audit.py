from __future__ import annotations

import unittest

from edgetam_batched.memory_slot_audit import _session_obj_id_to_idx, _session_obj_ids, compare_slot_digests


class FakeSession:
    def obj_ids(self):
        return [10, 20]

    def obj_id_to_idx(self, obj_id):
        return {10: 0, 20: 1}[obj_id]


class MemorySlotAuditTests(unittest.TestCase):
    def test_matching_digests_pass(self):
        digest = {
            "cond_keys": [0],
            "noncond_keys": [1],
            "tracked_keys": [1],
            "latest_key": 1,
            "latest_bucket": "non_cond_frame_outputs",
            "latest_fields": ["pred_masks"],
            "obj_ids": [1],
        }

        self.assertTrue(compare_slot_digests(digest, dict(digest))["pass"])

    def test_key_mismatch_reports_field(self):
        reference = {"cond_keys": [0], "noncond_keys": [1], "tracked_keys": [], "latest_key": 1, "latest_bucket": "x", "latest_fields": [], "obj_ids": [1]}
        candidate = dict(reference)
        candidate["noncond_keys"] = [2]

        result = compare_slot_digests(reference, candidate)

        self.assertFalse(result["pass"])
        self.assertEqual(result["mismatches"][0]["field"], "noncond_keys")

    def test_method_obj_id_mapping_is_supported(self):
        session = FakeSession()
        obj_ids = _session_obj_ids(session)

        self.assertEqual(obj_ids, [10, 20])
        self.assertEqual(_session_obj_id_to_idx(session, obj_ids), {"10": 0, "20": 1})


if __name__ == "__main__":
    unittest.main()
