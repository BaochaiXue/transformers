#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangxinjie/EdgeTAM-HF-batched
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate demo_2_max

RGB_REPLAY="/home/zhangxinjie/proj-QQTT-v2/result/demo22_rgb_triplet_100frames_towel_stuffed_animal"

if [ ! -d "$RGB_REPLAY" ]; then
  python -m edgetam_batched.rgb_replay \
    --mode record-realsense-triplet \
    --out-dir "$RGB_REPLAY" \
    --frames 100 \
    --camera-count 3 \
    --width 848 \
    --height 480 \
    --fps 15 \
    --object-prompt "stuffed animal" \
    --controller-prompt "towel" \
    --debug
fi

python -m edgetam_batched.rgb_replay --mode inspect --rgb-replay "$RGB_REPLAY"

python -m edgetam_batched.compare_multisession \
  --rgb-replay "$RGB_REPLAY" \
  --backend hf_batch_vision_seq_session \
  --frames 100 \
  --object-count 2 \
  --object-prompt "stuffed animal" \
  --controller-prompt "towel" \
  --dtype bfloat16 \
  --compile-mode none \
  --graph-output-policy ring_buffer \
  --output-md docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session.md \
  --output-json docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session.json

python -m edgetam_batched.profile_multisession \
  --rgb-replay "$RGB_REPLAY" \
  --backend hf_batch_vision_seq_session \
  --frames 100 \
  --warmup 20 \
  --object-count 2 \
  --dtype bfloat16 \
  --compile-mode none \
  --graph-output-policy ring_buffer \
  --output-json docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_none.json \
  --output-md docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_none.md

python -m edgetam_batched.final_report \
  --correctness-json 'docs/generated/edgetam_batched_correctness_*.json' \
  --profile-json 'docs/generated/edgetam_batched_profile_*.json' \
  --output-md docs/generated/edgetam_batched_multisession_final_report.md \
  --output-json docs/generated/edgetam_batched_multisession_final_report.json

python -m py_compile edgetam_batched/*.py
python -m unittest -v \
  tests.test_state_map \
  tests.test_camera_order \
  tests.test_leakage_test \
  tests.test_profile_stats \
  tests.test_ring_buffer \
  tests.test_compile_config \
  tests.test_batched_runtime_shapes \
  tests.test_component_adapter \
  tests.test_compare_multisession \
  tests.test_profile_multisession

git diff --check
