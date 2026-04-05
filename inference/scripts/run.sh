#!/bin/bash
# Run inference on a directory of damaged images.
# Edit CHECKPOINT, INPUT_DIR, OUTPUT_DIR, and DAMAGE below before submitting.
#
# Usage:
#   sbatch inference/scripts/run.sh
#
# DAMAGE: space-separated damage types present in the input images.
#   Options: crack paint_loss stain blur color_shift
#   Leave empty to mark all channels as damaged.

#SBATCH --job-name=art-restore-inference
#SBATCH --partition=education_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

mkdir -p logs

CHECKPOINT="checkpoints/final.pt"
INPUT_DIR="./data/test"
OUTPUT_DIR="./results"
DAMAGE="crack paint_loss"   # set to "" to mark all channels

python -m src.inference \
    --config    inference/configs/inference.yaml \
    --checkpoint ${CHECKPOINT} \
    --input     ${INPUT_DIR} \
    --output    ${OUTPUT_DIR} \
    --damage    ${DAMAGE}
