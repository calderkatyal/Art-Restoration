#!/bin/bash
# Stage 2 training: fine-tune all layers with separate LRs for backbone and img_in.
# Run after warmup.sh completes and set WARMUP_CKPT to the saved checkpoint path.
#
# Usage:
#   sbatch train/scripts/full.sh
#
# Overrides (append to python command as dot-notation, e.g.):
#   train.full.backbone_lr=5e-6
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

WARMUP_CKPT="checkpoints/warmup_final.pt"

python -m src.train \
    --config train/configs/train.yaml \
    train.stage=full \
    train.resume_from=${WARMUP_CKPT}
