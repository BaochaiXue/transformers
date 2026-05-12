from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from edgetam_batched.final_report import main


class FinalReportTests(unittest.TestCase):
    def test_final_report_accepts_sam31_iou_summary_and_caveat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            correctness = root / "correctness.json"
            correctness.write_text(
                json.dumps(
                    {
                        "backend": "hf_batched_multisession",
                        "compile_mode": "none",
                        "metrics": {
                            "correctness_pass": False,
                            "mask_correctness_pass": True,
                            "candidate_partial": True,
                            "fallback_backend": "hf_batch_vision_seq_session",
                            "empty_reference_policy": "ignore-candidate",
                            "object_summaries": {
                                "obj0": {
                                    "evaluated_sample_count": 0,
                                    "ignored_reference_empty_count": 10,
                                    "object_reference_absent": True,
                                    "reference_uncertain": True,
                                },
                                "obj1": {
                                    "evaluated_sample_count": 10,
                                    "ignored_reference_empty_count": 0,
                                    "object_reference_absent": False,
                                    "reference_uncertain": False,
                                },
                            },
                        },
                        "object_count": 2,
                        "controller_prompt": "hand",
                        "object_prompt": "stuffed animal",
                    }
                ),
                encoding="utf-8",
            )
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "status": "replay_profile",
                        "backend": "hf_batch_vision_seq_session",
                        "compile_mode": "reduce-overhead",
                        "profile": {
                            "partial": False,
                            "timings_ms": {"stage_wall_ms": {"p50": 58.4, "p90": 65.8}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary = root / "iou_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "reference_source": "sam31-replay",
                        "sam31_mask_root": "/tmp/sam31",
                        "backend": "hf_batch_vision_seq_session",
                        "results": {
                            "reduce-overhead": {
                                "correctness_pass": True,
                                "empty_mismatch_count": 0,
                                "global_mask_iou": {"avg": 0.98138, "min": 0.93835, "p50": 0.99073},
                                "per_camera_object": [
                                    {"key": "cam0_obj0", "avg": 1.0, "reference_nonempty_count": 0},
                                    {"key": "cam1_obj0", "avg": 1.0, "reference_nonempty_count": 0},
                                    {"key": "cam2_obj0", "avg": 1.0, "reference_nonempty_count": 0},
                                    {"key": "cam0_obj1", "avg": 0.97472, "reference_nonempty_count": 100},
                                    {"key": "cam1_obj1", "avg": 0.96269, "reference_nonempty_count": 100},
                                    {"key": "cam2_obj1", "avg": 0.94373, "reference_nonempty_count": 100},
                                ],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            different_types = root / "different_types.json"
            different_types.write_text(
                json.dumps(
                    {
                        "replay": "/tmp/replay",
                        "single_object": {
                            "backend": "hf_batch_vision_seq_session",
                            "compile_mode": "reduce-overhead",
                            "stage_wall_ms": {"p50": 31.31, "p90": 32.99, "p95": 33.75},
                            "complete_group_fps_from_p50": 31.94,
                            "p50_30fps_gate": True,
                            "p90_30fps_gate": True,
                            "p95_30fps_gate": False,
                        },
                        "decision": {
                            "single_object_stuffed_animal_validated": True,
                            "controller_hand_validated": False,
                            "controller_hand_reason": "low IoU outliers on cam0/cam2",
                        },
                    }
                ),
                encoding="utf-8",
            )
            delta = root / "delta.json"
            delta.write_text(
                json.dumps(
                    {
                        "baseline_backend": "hf_ref_seq_public",
                        "candidate_backend": "hf_batch_vision_seq_session",
                        "candidate_compile_mode": "reduce-overhead",
                        "empty_reference_policy": "ignore-candidate",
                        "evaluated_subset_mismatch": False,
                        "per_object": {
                            "obj0": {
                                "baseline_iou_avg_on_evaluated": None,
                                "candidate_iou_avg_on_evaluated": None,
                                "delta": None,
                                "not_worse": None,
                                "evaluated_sample_count": 0,
                                "ignored_reference_empty_count": 10,
                                "candidate_nonempty_when_reference_empty_count": 2,
                                "reference_uncertain": True,
                                "reason": "no evaluated SAM3.1 reference samples",
                            },
                            "obj1": {
                                "baseline_iou_avg_on_evaluated": 0.96,
                                "candidate_iou_avg_on_evaluated": 0.955,
                                "delta": -0.005,
                                "not_worse": True,
                                "evaluated_sample_count": 10,
                                "ignored_reference_empty_count": 0,
                                "candidate_nonempty_when_reference_empty_count": 0,
                                "reference_uncertain": False,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            out_md = root / "final.md"
            out_json = root / "final.json"

            argv = [
                "final_report",
                "--correctness-json",
                str(correctness),
                "--profile-json",
                str(profile),
                "--iou-ref-summary",
                str(summary),
                "--different-types-summary",
                str(different_types),
                "--candidate-delta",
                str(delta),
                "--output-md",
                str(out_md),
                "--output-json",
                str(out_json),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(main(), 0)

            md = out_md.read_text(encoding="utf-8")
            payload = json.loads(out_json.read_text(encoding="utf-8"))

            self.assertGreater(len(md.splitlines()), 20)
            self.assertIn("SAM3.1 replay reference correctness", md)
            self.assertIn("Non-empty object quality: stuffed animal only", md)
            self.assertIn("Controller/towel caveat", md)
            self.assertIn("Different-types sloth_set_2 result", md)
            self.assertIn("Empty SAM3.1 reference policy", md)
            self.assertIn("Original vs compiled delta on evaluated samples", md)
            self.assertIn("empty-vs-empty", md)
            self.assertTrue(payload["decision"]["hf_batch_vision_seq_session_usable"])
            self.assertTrue(payload["decision"]["single_object_stuffed_animal_validated"])
            self.assertTrue(payload["decision"]["single_object_30fps_p50_gate"])
            self.assertTrue(payload["decision"]["single_object_30fps_p90_gate"])
            self.assertEqual(payload["decision"]["single_object_30fps_p95_gate"], "borderline/fail")
            self.assertFalse(payload["decision"]["controller_hand_validated"])
            self.assertFalse(payload["decision"]["hf_batched_multisession_usable"])
            self.assertEqual(payload["decision"]["recommended_backend"], "hf_batch_vision_seq_session")
            self.assertEqual(payload["decision"]["recommended_compile_mode"], "reduce-overhead")
            self.assertEqual(payload["decision"]["empty_reference_policy"], "ignore-candidate")
            self.assertIn("stuffed animal", payload["decision"]["strict_validated_objects"])
            self.assertIn("hand", payload["decision"]["reference_uncertain_objects"])
            self.assertTrue(payload["decision"]["demo22_final_fps_pending"])
            self.assertGreater(out_json.stat().st_size, 100)

    def test_full_compiled_memory_path_can_be_recommended(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            correctness_files = []
            for mode in ("none", "default", "max-autotune-no-cudagraphs", "reduce-overhead"):
                path = root / f"full_{mode}.json"
                path.write_text(
                    json.dumps(
                        {
                            "backend": "hf_batched_multisession",
                            "compile_mode": mode,
                            "precision_mode": "memory_path_fp32",
                            "reference_source": "hf-public-seq",
                            "strict_full_batched": True,
                            "dtype": "bfloat16",
                            "metrics": {
                                "strict_correctness_pass": True,
                                "correctness_pass": True,
                                "global_iou_on_evaluated": {"avg": 0.987, "p50": 0.99},
                                "backend_contract": {
                                    "backend": "hf_batched_multisession",
                                    "batch_vision": True,
                                    "batch_memory_attention": True,
                                    "batch_mask_decoder": True,
                                    "batch_memory_encoder": True,
                                    "batched_state_scatter": True,
                                    "used_public_session_step_in_hot_path": False,
                                    "partial_fallback_used": False,
                                    "contract_pass": True,
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                correctness_files.append(str(path))
            out_md = root / "final.md"
            out_json = root / "final.json"
            argv = [
                "final_report",
                "--correctness-json",
                *correctness_files,
                "--output-md",
                str(out_md),
                "--output-json",
                str(out_json),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(main(), 0)

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["decision"]["hf_batched_multisession_usable"])
            self.assertTrue(payload["decision"]["full_batched_memory_path_fp32_strict_pass"])
            self.assertTrue(payload["decision"]["full_batched_compile_default_pass"])
            self.assertTrue(payload["decision"]["full_batched_compile_max_autotune_no_cudagraphs_pass"])
            self.assertTrue(payload["decision"]["full_batched_compile_reduce_overhead_pass"])
            self.assertEqual(payload["decision"]["recommended_backend"], "hf_batched_multisession")
            self.assertEqual(payload["decision"]["recommended_compile_mode"], "reduce-overhead")


if __name__ == "__main__":
    unittest.main()
