import unittest

import numpy as np

from edgetam_batched.current_frame_isolation import (
    VARIANT_BLACK_NEIGHBORS,
    VARIANT_NORMAL,
    VARIANT_REPEATED_TARGET,
    build_variant_specs,
    dtype_for_precision,
    infer_issue,
    mask_geometry,
    parse_camera_index,
    replacement_status,
)


class CurrentFrameIsolationTests(unittest.TestCase):
    def test_parse_camera_index(self):
        self.assertEqual(parse_camera_index("cam1"), 1)
        self.assertEqual(parse_camera_index("2"), 2)

    def test_variant_specs_include_target_variants(self):
        specs = {spec.name: spec for spec in build_variant_specs(1, 3)}
        self.assertEqual(specs[VARIANT_NORMAL].image_indices, (0, 1, 2))
        self.assertEqual(specs[VARIANT_REPEATED_TARGET].image_indices, (1, 1, 1))
        self.assertEqual(specs[VARIANT_BLACK_NEIGHBORS].image_indices, (None, 1, None))

    def test_replacement_status_marks_unsupported_hooks(self):
        self.assertTrue(replacement_status("none")["supported"])
        self.assertFalse(replacement_status("decoder_inputs_with_reference")["supported"])
        self.assertTrue(replacement_status("postprocess_output_with_reference")["posthoc"])

    def test_precision_dtype(self):
        self.assertEqual(dtype_for_precision("bfloat16", "all_fp32"), "float32")
        self.assertEqual(dtype_for_precision("bfloat16", "memory_path_fp32"), "bfloat16")

    def test_mask_geometry(self):
        ref = np.zeros((5, 5), dtype=bool)
        cand = np.zeros((5, 5), dtype=bool)
        ref[1:3, 1:3] = True
        cand[2:4, 2:4] = True
        geom = mask_geometry(ref, cand)
        self.assertEqual(geom["reference_area"], 4)
        self.assertEqual(geom["candidate_area"], 4)
        self.assertEqual(geom["intersection_area"], 1)
        self.assertEqual(geom["union_area"], 7)
        self.assertGreater(geom["threshold_flip_count"], 0)

    def test_infer_issue_neighbor_dependency(self):
        rows = [
            {"variant": "normal_batch", "status": "ok", "raw_iou_vs_hf_public": 0.5},
            {"variant": "repeated_target", "status": "ok", "raw_iou_vs_hf_public": 0.99},
        ]
        self.assertIn("neighbor", infer_issue(rows, "none", "all_bf16"))


if __name__ == "__main__":
    unittest.main()
