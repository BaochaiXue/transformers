"""Scan observed BatchTam memory-attention recurrent shapes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from edgetam_batched.report_utils import markdown_table, write_json, write_markdown

from .memory_attention_shape_key import MemoryAttentionShapeKey


def scan_fixtures(fixtures_dir: str | Path, *, frames: int | None = None) -> dict[str, Any]:
    import torch

    records = []
    root = Path(fixtures_dir)
    frame_dirs = sorted(root.glob("frame_*"))
    if frames is not None:
        frame_dirs = frame_dirs[: int(frames)]
    for frame_dir in frame_dirs:
        for path in sorted(frame_dir.glob("cam*_memory_attention.pt")):
            fixture = torch.load(path, map_location="cpu", weights_only=False)
            kwargs = fixture["kwargs"]
            key = MemoryAttentionShapeKey.from_inputs(
                current_vision_features=kwargs["current_vision_features"],
                current_vision_position_embeddings=kwargs["current_vision_position_embeddings"],
                memory=kwargs["memory"],
                memory_posision_embeddings=kwargs["memory_posision_embeddings"],
                num_object_pointer_tokens=kwargs["num_object_pointer_tokens"],
                num_spatial_memory_tokens=kwargs["num_spatial_memory_tokens"],
            )
            records.append(
                {
                    "frame_idx": int(fixture["frame_idx"]),
                    "camera_idx": int(fixture["camera_idx"]),
                    "batch_size": 3,
                    "num_object_pointer_tokens": key.num_object_pointer_tokens,
                    "num_spatial_memory_tokens": key.num_spatial_memory_tokens,
                    "memory_seq_len": key.memory_seq_len,
                    "current_vision_features_shape": list(kwargs["current_vision_features"].shape),
                    "memory_features_shape": list(kwargs["memory"].shape),
                    "memory_pos_enc_shape": list(kwargs["memory_posision_embeddings"].shape),
                    "shape_key": key.slug,
                    "input_shape_signature": key.to_json()["input_shape_signature"],
                    "fixture_path": str(path),
                }
            )
    buckets: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["shape_key"]].append(record)
    for idx, (shape_key, rows) in enumerate(sorted(grouped.items(), key=lambda item: item[1][0]["memory_seq_len"])):
        first = rows[0]
        buckets[shape_key] = {
            "bucket_index": idx,
            "bucket_dir_name": f"shape_{idx:03d}_{shape_key.split('_cur', 1)[0]}",
            "shape_key": shape_key,
            "records": len(rows),
            "frames": sorted({row["frame_idx"] for row in rows}),
            "first_frame": min(row["frame_idx"] for row in rows),
            "last_frame": max(row["frame_idx"] for row in rows),
            "num_object_pointer_tokens": first["num_object_pointer_tokens"],
            "num_spatial_memory_tokens": first["num_spatial_memory_tokens"],
            "memory_seq_len": first["memory_seq_len"],
            "input_shape_signature": first["input_shape_signature"],
            "representative_fixture_paths": [
                next(row["fixture_path"] for row in rows if row["camera_idx"] == cam_idx)
                for cam_idx in range(3)
                if any(row["camera_idx"] == cam_idx for row in rows)
            ],
        }
    return {
        "component": "memory_attention",
        "fixtures_dir": str(root),
        "records": records,
        "shape_buckets": buckets,
        "observed_shape_keys": list(buckets),
        "max_num_object_pointer_tokens": max((row["num_object_pointer_tokens"] for row in records), default=None),
        "max_num_spatial_memory_tokens": max((row["num_spatial_memory_tokens"] for row in records), default=None),
    }


def render(payload: dict[str, Any]) -> str:
    rows = [
        [
            item["shape_key"],
            len(item["frames"]),
            item["first_frame"],
            item["last_frame"],
            item["num_object_pointer_tokens"],
            item["num_spatial_memory_tokens"],
            item["memory_seq_len"],
        ]
        for item in payload["shape_buckets"].values()
    ]
    return "\n".join(
        [
            "# BatchTam Memory-Attention Shape Sequence",
            "",
            f"- fixtures_dir: `{payload['fixtures_dir']}`",
            f"- observed_shape_count: `{len(payload['shape_buckets'])}`",
            f"- max_num_object_pointer_tokens: `{payload['max_num_object_pointer_tokens']}`",
            f"- max_num_spatial_memory_tokens: `{payload['max_num_spatial_memory_tokens']}`",
            "",
            markdown_table(
                [
                    "shape_key",
                    "frames",
                    "first_frame",
                    "last_frame",
                    "object_pointer_tokens",
                    "spatial_memory_tokens",
                    "memory_seq_len",
                ],
                rows,
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", default=None)
    parser.add_argument("--backend", default="hf_batched_multisession")
    parser.add_argument("--reference-source", default="hf-public-seq")
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--precision-mode", default="memory_path_fp32")
    parser.add_argument("--fixtures-dir", default="docs/generated/fixtures/single_object")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    payload = scan_fixtures(args.fixtures_dir, frames=args.frames)
    payload.update(
        {
            "rgb_replay": args.rgb_replay,
            "backend": args.backend,
            "reference_source": args.reference_source,
            "object_count": args.object_count,
            "object_prompt": args.object_prompt,
            "dtype": args.dtype,
            "precision_mode": args.precision_mode,
        }
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload["shape_buckets"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
