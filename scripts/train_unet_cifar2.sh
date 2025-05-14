#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# !!! IMPORTANT: SET THESE PATHS BEFORE RUNNING THE SCRIPT !!!
# You can either set them directly here, or pass them as environment variables
# when you run the script (e.g., `OUTPUT_DIR=/path/to/output DATA_ROOT=/path/to/data ./run_train_eval_unet_cifar.sh`)

# Default values (modify as needed or ensure they are set in your environment)
: "${OUTPUT_DIR:="./output_unet_cifar_from_script"}" # Directory to save model checkpoints, logs, and generated samples/metrics
: "${DATA_ROOT:="./data"}"                         # Root directory of your CIFAR-10 dataset

# --- Create directories if they don't exist ---
mkdir -p "$OUTPUT_DIR"
# mkdir -p "$DATA_ROOT" # Usually, data root should already exist if it contains CIFAR-10

echo "--- Configuration ---"
echo "Output Directory: $OUTPUT_DIR"
echo "Data Root:      $DATA_ROOT"
echo "---------------------"
echo ""

echo "Starting UNet CIFAR-10 training and evaluation with Rectified Flow..."
echo "This script assumes 'rectified_flow/pipelines/train_unet_cifar.py' includes evaluation logic."
echo ""

# --- Run Training and Evaluation ---
# The `accelerate launch` command will execute your Python script.
# Ensure your Python script handles the generation of samples and calculation of FID/IS
# metrics after training is complete, using the trained model (preferably EMA).

accelerate launch -m rectified_flow.pipelines.train_unet_cifar \
  --output_dir="$OUTPUT_DIR" \
  --resume_from_checkpoint="latest" \
  --data_root="$DATA_ROOT" \
  --seed=0 \
  --resolution=32 \
  --train_batch_size=128 \
  --max_train_steps=1000000 \
  --checkpointing_steps=20000 \
  --learning_rate=2e-4 \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=10000 \
  --random_flip \
  --allow_tf32 \
  --interp="straight" \
  --source_distribution="normal" \
  --is_independent_coupling=True \
  --train_time_distribution="lognormal" \
  --train_time_weight="uniform" \
  --criterion="mse" \
  --use_ema \
  # --sample_batch_size=64 \ # If you added this arg to your python script for generation
  # --nfe_sampling=255   \ # IMPORTANT: If you added an argument for NFE in your Python script's evaluation part, uncomment and set this.
                           # This controls the quality of generated samples.

echo ""
echo "--- Script Finished ---"
echo "Training checkpoints, logs, and evaluation results (if implemented in the Python script)"
echo "should be in: $OUTPUT_DIR"
echo "If evaluation was run, generated samples might be in: $OUTPUT_DIR/unet_samples"
echo "And metrics in a file like: $OUTPUT_DIR/unet_metrics.txt"
echo "-----------------------"