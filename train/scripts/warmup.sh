#!/bin/bash
# Training from scratch. The first `train.warmup_iterations` optimizer steps
# train `img_in` only; afterward the backbone is unfrozen automatically.
#
# Usage:
#   sbatch train/scripts/warmup.sh
#
# Overrides (append to python command as dot-notation, e.g.):
#   train.batch_size=8
#   train.warmup.lr=5e-5
#   train.optimizer.weight_decay=0.05

#SBATCH --job-name=art-restore-warmup
#SBATCH --partition=education_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/warmup_%j.out
#SBATCH --error=logs/warmup_%j.err

mkdir -p logs

# Precompute null text embedding if not already cached
python -m src.null_emb --config train/configs/train.yaml

python -m src.train --config train/configs/train.yaml
