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
                        },
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
            self.assertIn("empty-vs-empty", md)
            self.assertTrue(payload["decision"]["hf_batch_vision_seq_session_usable"])
            self.assertFalse(payload["decision"]["hf_batched_multisession_usable"])
            self.assertEqual(payload["decision"]["recommended_backend"], "hf_batch_vision_seq_session")
            self.assertEqual(payload["decision"]["recommended_compile_mode"], "reduce-overhead")
            self.assertGreater(out_json.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
