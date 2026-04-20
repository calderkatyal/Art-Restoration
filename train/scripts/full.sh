#!/bin/bash
# Resume training from an existing DeepSpeed checkpoint tag directory.
# If the checkpoint was saved after warm-up, the backbone will already be unfrozen
# automatically based on `train.warmup_iterations` and the saved global step.
#
# Usage:
#   sbatch train/scripts/full.sh
#
# Overrides (append to python command as dot-notation, e.g.):
#   train.warmup_iterations=0
#   train.batch_size=4

#SBATCH --job-name=art-restore-full
#SBATCH --partition=education_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

mkdir -p logs

RESUME_CKPT="checkpoints/step_1000"

python -m src.train \
    --config train/configs/train.yaml \
    train.resume_from=${RESUME_CKPT}
