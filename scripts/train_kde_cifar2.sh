#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# !!! IMPORTANT: SET THESE PATHS BEFORE RUNNING THE SCRIPT !!!
# You can either set them directly here, or pass them as environment variables
# when you run the script (e.g., `OUTPUT_DIR=/path/to/output DATA_ROOT=/path/to/data ./run_kde_cifar.sh`)

# Default values (modify as needed or ensure they are set in your environment)
: "${OUTPUT_DIR:="./output_kde_cifar_from_script"}" # Directory to save generated samples and metrics
: "${DATA_ROOT:="./data"}"                         # Root directory of your CIFAR-10 dataset

# --- Create directories if they don't exist ---
mkdir -p "$OUTPUT_DIR"
# mkdir -p "$DATA_ROOT" # Usually, data root should already exist if it contains CIFAR-10

echo "--- Configuration ---"
echo "Output Directory: $OUTPUT_DIR"
echo "Data Root:      $DATA_ROOT"
echo "---------------------"
echo ""

echo "Starting PCA+KDE baseline for CIFAR-10..."
echo ""

# --- Run KDE Fitting, Sampling, and Evaluation ---
# The `train_kde_cifar.py` script handles fitting PCA+KDE,
# sampling images, and calculating FID/IS metrics.

# Adjust parameters as needed:
PCA_COMPONENTS=100
BANDWIDTH=0.1
N_SAMPLES=50000 # Number of images to sample from KDE for evaluation
SEED=0
# DEVICE="cuda" # Or "mps", "cpu", or leave as None for accelerator to decide (as in your Python script)
                # If you want to specify, uncomment and set. Otherwise, Python script default will be used.

# Note: The `train_kde_cifar.py` uses `accelerator = Accelerator()` but mostly for device selection.
# It's run as a standard Python script, not typically with `accelerate launch`.

python rectified_flow/pipelines/train_kde_cifar.py \
  --data_root="$DATA_ROOT" \
  --output_dir="$OUTPUT_DIR" \
  --pca_components="$PCA_COMPONENTS" \
  --bandwidth="$BANDWIDTH" \
  --n_samples="$N_SAMPLES" \
  --seed="$SEED" \
  # --device="$DEVICE" # Uncomment if you want to override the device selection in the Python script

echo ""
echo "--- Script Finished ---"
echo "KDE generated samples and metrics should be in: $OUTPUT_DIR"
echo "Generated samples are typically in: $OUTPUT_DIR/kde_samples"
echo "Metrics are typically in: $OUTPUT_DIR/metrics.txt"
echo "-----------------------"