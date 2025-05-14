#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_DIR="${OUTPUT_DIR:-./kde_output}"
export DATA_ROOT="${DATA_ROOT:-./data}"

accelerate launch \
  --num_processes 1 \
  --mixed_precision no \
  --num_machines 1 \
  rectified_flow/pipelines/train_kde_cifar.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --pca_components 100 \
    --bandwidth 0.1 \
    --n_samples 50000 \
    --seed 0
