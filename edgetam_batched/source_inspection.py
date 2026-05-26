"""Inspect the sparse HF EdgeTAM source fork and write a source report."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from .report_utils import markdown_table, write_json, write_markdown


def inspect_source(repo_root: str | Path = ".") -> dict:
    root = Path(repo_root)
    files = {
        "edgetam_modular": root / "src/transformers/models/edgetam_video/modular_edgetam_video.py",
        "edgetam_modeling": root / "src/transformers/models/edgetam_video/modeling_edgetam_video.py",
        "sam2_video_modeling": root / "src/transformers/models/sam2_video/modeling_sam2_video.py",
        "sam2_video_processing": root / "src/transformers/models/sam2_video/processing_sam2_video.py",
    }
    contents = {name: _read(path) for name, path in files.items()}
    payload = {
        "repo": {
            "branch": _git(root, "branch", "--show-current"),
            "commit": _git(root, "rev-parse", "HEAD"),
            "status_short": _git(root, "status", "--short"),
        },
        "files": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "generated_file": "generated from" in contents.get(name, "")[:800].lower()
                or "do not edit" in contents.get(name, "")[:800].lower(),
            }
            for name, path in files.items()
        },
        "findings": {
            "edgetam_subclasses_sam2_video": "Sam2VideoModel" in contents["edgetam_modular"],
            "public_forward_has_inference_session": "inference_session"
            in _extract_method_block(contents["edgetam_modular"], "EdgeTamVideoModel", "forward"),
            "get_image_features_defined": "def get_image_features" in contents["sam2_video_modeling"]
            or "def get_image_features" in contents["edgetam_modular"],
            "session_class_defined": "class EdgeTamVideoInferenceSession" in contents["edgetam_modular"],
            "processor_init_video_session": "def init_video_session" in contents["sam2_video_processing"],
            "processor_add_inputs": "def add_inputs_to_inference_session" in contents["sam2_video_processing"],
        },
        "components": {
            name: _find_tokens("\n".join(contents.values()), token)
            for name, token in {
                "vision_encoder": "vision_encoder",
                "memory_attention": "memory_attention",
                "memory_encoder": "memory_encoder",
                "mask_decoder": "mask_decoder",
                "prompt_encoder": "prompt_encoder",
                "object_pointer": "object_pointer",
            }.items()
        },
        "signatures": {
            "EdgeTamVideoModel.forward": _extract_method_signature(
                contents["edgetam_modular"], "EdgeTamVideoModel", "forward"
            ),
            "Sam2VideoModel.get_image_features": _extract_method_signature(
                contents["sam2_video_modeling"], "Sam2VideoModel", "get_image_features"
            ),
            "EdgeTamVideoInferenceSession": _extract_class_line(contents["edgetam_modular"], "EdgeTamVideoInferenceSession"),
        },
    }
    return payload


def render_source_report(payload: dict) -> str:
    file_rows = [
        [name, item["path"], item["exists"], item["generated_file"], item["size_bytes"]]
        for name, item in payload["files"].items()
    ]
    finding_rows = list(payload["findings"].items())
    component_rows = [[name, count] for name, count in payload["components"].items()]
    signature_rows = list(payload["signatures"].items())
    return "\n".join(
        [
            "# EdgeTAM Batched Source Inspection",
            "",
            "## Repo",
            "",
            markdown_table(["field", "value"], payload["repo"].items()),
            "",
            "## Files",
            "",
            markdown_table(["name", "path", "exists", "generated", "size"], file_rows),
            "",
            "## Findings",
            "",
            markdown_table(["finding", "value"], finding_rows),
            "",
            "## Component Token Counts",
            "",
            markdown_table(["component", "count"], component_rows),
            "",
            "## Signatures",
            "",
            markdown_table(["symbol", "signature"], signature_rows),
            "",
            "## Boundary",
            "",
            "- Do not edit `modeling_edgetam_video.py` directly; it is generated.",
            "- The research runtime should use wrapper modules first and only patch modular source for accessors if blocked.",
        ]
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _find_tokens(text: str, token: str) -> int:
    return len(re.findall(re.escape(token), text))


def _extract_signature(text: str, name: str) -> str:
    match = re.search(rf"def {re.escape(name)}\((.*?)\):", text, re.DOTALL)
    if not match:
        return ""
    sig = " ".join(match.group(1).split())
    return f"def {name}({sig})"


def _extract_method_signature(text: str, class_name: str, method_name: str) -> str:
    block = _extract_class_block(text, class_name)
    return _extract_signature(block, method_name)


def _extract_method_block(text: str, class_name: str, method_name: str) -> str:
    block = _extract_class_block(text, class_name)
    match = re.search(rf"\n    def {re.escape(method_name)}\(.*?(?=\n    def |\nclass |\Z)", block, re.DOTALL)
    return match.group(0) if match else ""


def _extract_class_block(text: str, class_name: str) -> str:
    match = re.search(rf"^class {re.escape(class_name)}\(.*?(?=^class |\Z)", text, re.DOTALL | re.MULTILINE)
    return match.group(0) if match else ""


def _extract_class_line(text: str, name: str) -> str:
    match = re.search(rf"class {re.escape(name)}\(.*?\):", text)
    return match.group(0) if match else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    payload = inspect_source(args.repo_root)
    write_markdown(args.output_md, render_source_report(payload))
    if args.output_json:
        write_json(args.output_json, payload)
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
