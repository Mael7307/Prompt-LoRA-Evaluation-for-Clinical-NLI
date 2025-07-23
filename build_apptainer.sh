#!/usr/bin/env bash
#SBATCH --job-name=build_sacnli_apptainer
#SBATCH --output=logs/build_sacnli_apptainer_%j.out
#SBATCH --error=logs/build_sacnli_apptainer_%j.err
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=1:30:00
#SBATCH --gpus=rtx3090:1
#SBATCH --account=rational

set -euo pipefail
IFS=$'\n\t'

# --- Configuration ---
OUTPUT_FILE="SACNLI.sif"
LOG_PREFIX="[Apptainer Build]"
SRC_DIR="$(pwd)"
TMP_DIR="/tmp"

echo "$LOG_PREFIX Syncing source to $TMP_DIR..."
rsync -a "$SRC_DIR" "$TMP_DIR"

cd "$TMP_DIR" || exit 1

echo "$LOG_PREFIX Starting build in $(pwd)"
echo "$LOG_PREFIX Apptainer version: $(apptainer --version)"

apptainer build --fakeroot --bind /dev/shm:/tmp "$OUTPUT_FILE" Apptainer

echo "$LOG_PREFIX Build complete: $OUTPUT_FILE"
echo "$LOG_PREFIX Copying container back to $SRC_DIR"
cp "$OUTPUT_FILE" "$SRC_DIR"

echo "$LOG_PREFIX Done."

