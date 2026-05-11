import unittest

from edgetam_batched.decoder_diff_probe import extract_decoder_probe, first_divergent_tensor


class DecoderDiffProbeTests(unittest.TestCase):
    def test_first_divergent_tensor(self):
        rows = [
            {"field": "pred_masks", "shape_match": True, "p95_abs_diff": 0.0, "max_abs_diff": 0.0},
            {"field": "object_pointer", "shape_match": True, "p95_abs_diff": 0.2, "max_abs_diff": 0.3},
        ]
        self.assertEqual(first_divergent_tensor(rows)["field"], "object_pointer")

    def test_extract_decoder_probe(self):
        payload = {
            "backend": "hf_batched_multisession",
            "frame_idx": 47,
            "camera": "cam1",
            "normal_variant": {
                "raw_iou_vs_hf_public": 0.7,
                "geometry": {"threshold_flip_count": 4, "reference_area": 10, "candidate_area": 12},
                "field_diffs": {
                    "pred_masks": {
                        "shape_match": True,
                        "max_abs_diff": 1.0,
                        "mean_abs_diff": 0.1,
                        "p95_abs_diff": 0.5,
                    }
                },
            },
        }
        result = extract_decoder_probe(payload)
        self.assertEqual(result["first_divergent_tensor"]["field"], "pred_masks")
        self.assertEqual(result["logit_diff_causes_threshold_flip_count"], 4)


if __name__ == "__main__":
    unittest.main()
