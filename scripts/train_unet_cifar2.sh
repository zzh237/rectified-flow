#!/bin/bash
set -e

# --- Configuration ---
: "${OUTPUT_DIR:="./output_unet_cifar_bf16_test"}"
: "${DATA_ROOT:="./data"}"

mkdir -p "$OUTPUT_DIR"

echo "--- Configuration ---"
echo "Output Directory: $OUTPUT_DIR"
echo "Data Root:      $DATA_ROOT"
echo "---------------------"
echo ""

echo "Starting UNet CIFAR-10 training (BF16, short test run) and evaluation..."

MAX_STEPS=8000 # 先用一个较小的步数测试 BF16 速度和参数传递
CHECKPOINTING_STEPS=$((MAX_STEPS / 3))
LR_WARMUP_STEPS=$((MAX_STEPS / 10))
TRAIN_BATCH_SIZE=128
LEARNING_RATE=2e-4

accelerate launch -m rectified_flow.pipelines.train_unet_cifar \
  --output_dir="$OUTPUT_DIR" \
  --data_root="$DATA_ROOT" \
  --seed=0 \
  --resolution=32 \
  --train_batch_size="$TRAIN_BATCH_SIZE" \
  --max_train_steps="$MAX_STEPS" \
  --checkpointing_steps="$CHECKPOINTING_STEPS" \
  --learning_rate="$LEARNING_RATE" \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps="$LR_WARMUP_STEPS" \
  --random_flip \
  --allow_tf32 \
  --mixed_precision="bf16" \       
  --interp="straight" \
  --source_distribution="normal" \
  --is_independent_coupling=True \
  --train_time_distribution="lognormal" \
  --train_time_weight="uniform" \
  --criterion="mse" \
  --use_ema

echo ""
echo "--- Script Finished ---"
# ...

