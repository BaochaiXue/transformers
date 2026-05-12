import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from edgetam_batched.onnx_trt.report_trt import main, merge_component_reports


class TrtReportTests(unittest.TestCase):
    def test_merge_component_reports_marks_unpassed_stage(self):
        reports = [
            {
                "component": "memory_attention",
                "onnx_export_pass": True,
                "onnx_validation_pass": True,
                "trt_build_pass": True,
                "trt_validation_pass": True,
            },
            {
                "component": "mask_decoder",
                "onnx_export_pass": True,
                "onnx_validation_pass": False,
            },
        ]

        merged = merge_component_reports(reports)

        self.assertTrue(merged["memory_attention"]["trt_validation_pass"])
        self.assertEqual(merged["mask_decoder"]["failure_stage"], "onnx_validation")
        self.assertEqual(merged["memory_encoder"]["failure_stage"], "onnx_export")

    def test_bucketed_memory_attention_report_exposes_demo22_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            component = root / "component_validation_mask_decoder.json"
            component.write_text(
                json.dumps(
                    {
                        "component": "mask_decoder",
                        "trt_validation_pass": True,
                    }
                ),
                encoding="utf-8",
            )
            memory_encoder = root / "component_validation_memory_encoder.json"
            memory_encoder.write_text(
                json.dumps(
                    {
                        "component": "memory_encoder",
                        "trt_validation_pass": True,
                    }
                ),
                encoding="utf-8",
            )
            memory_attention = root / "component_validation_memory_attention.json"
            memory_attention.write_text(
                json.dumps(
                    {
                        "component": "memory_attention",
                        "trt_validation_pass": True,
                    }
                ),
                encoding="utf-8",
            )
            shape = root / "shape.json"
            shape.write_text(
                json.dumps(
                    {
                        "observed_shape_keys": ["objptr4_spmem1_cur4096_mem516", "objptr8_spmem2_cur4096_mem1032"]
                    }
                ),
                encoding="utf-8",
            )
            bucket_report = {
                "buckets": {
                    "shape_000": {
                        "onnx_export_pass": True,
                        "trt_build_pass": True,
                        "trt_validation_pass": True,
                    },
                    "shape_001": {
                        "onnx_export_pass": True,
                        "trt_build_pass": True,
                        "trt_validation_pass": True,
                    },
                }
            }
            bucket_export = root / "bucket_export.json"
            bucket_build = root / "bucket_build.json"
            bucket_validation = root / "bucket_validation.json"
            for path in (bucket_export, bucket_build, bucket_validation):
                path.write_text(json.dumps(bucket_report), encoding="utf-8")
            closed_loop_reports = []
            for scope, flags in (
                ("memory_attention", {"trt_memory_attention": True}),
                ("mask_decoder", {"trt_mask_decoder": True}),
                ("memory_encoder", {"trt_memory_encoder": True}),
                (
                    "memory_path_all",
                    {"trt_memory_attention": True, "trt_mask_decoder": True, "trt_memory_encoder": True},
                ),
            ):
                path = root / f"closed_loop_{scope}.json"
                payload = {
                    "trt_scope": scope,
                    "trt_memory_attention_bucket_dir": "engines/memory_attention_buckets"
                    if scope == "memory_attention"
                    else None,
                    "metrics": {
                        "strict_correctness_pass": True,
                        "backend_contract": flags,
                    },
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                closed_loop_reports.append(str(path))
            output_json = root / "report.json"
            output_md = root / "report.md"

            argv = [
                "report_trt",
                "--component-report",
                str(component),
                str(memory_attention),
                str(memory_encoder),
                "--closed-loop-report",
                *closed_loop_reports,
                "--memory-attention-shape-json",
                str(shape),
                "--memory-attention-bucket-export-json",
                str(bucket_export),
                "--memory-attention-bucket-build-json",
                str(bucket_build),
                "--memory-attention-bucket-validation-json",
                str(bucket_validation),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(main(), 0)

            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(report["batchtam_component_engines_usable"])
            self.assertTrue(report["batchtam_closed_loop_usable"])
            self.assertTrue(report["demo22_trt_integration_allowed"])
            self.assertEqual(report["recommended_trt_scope"], "memory_path_all")
            self.assertEqual(report["memory_attention_shape_strategy"], "bucketed_static_engines")
            self.assertEqual(report["memory_attention_bucket_count"], 2)
            self.assertEqual(report["memory_attention_buckets_exported"], 2)
            self.assertEqual(report["memory_attention_buckets_built"], 2)
            self.assertEqual(report["memory_attention_buckets_validated"], 2)
            self.assertTrue(report["memory_attention_bucketed_closed_loop_pass"])
            self.assertTrue(report["mask_decoder_trt_closed_loop_pass"])
            self.assertTrue(report["memory_encoder_trt_closed_loop_pass"])
            self.assertTrue(report["memory_path_all_trt_closed_loop_pass"])


if __name__ == "__main__":
    unittest.main()
