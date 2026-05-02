#!/bin/bash
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --job-name=diag-old-val
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --output=diag_%j.out
#SBATCH --error=diag_%j.err

# usage:
#   sbatch diag_old_val.sbatch <ckpt.pth> [<resolution>] [<resize_mode>] [<noise>]
#
# Defaults match a vit3d-depth6 active_matter run extracted under the new
# (post-fix) eval.sbatch defaults: resolution=[256, 256], resize_mode=fft,
# noise=0.0. If your run had different settings, override here so the cache
# hash matches what's on disk.

set -euo pipefail

CKPT="${1:?pass the encoder .pth checkpoint path as argument}"
RESOLUTION="${2:-[256, 256]}"
RESIZE_MODE="${3:-fft}"
NOISE="${4:-0.0}"

REPO=/scratch/$USER/physical-representation-learning
OVERLAY=$REPO/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "checkpoint: $CKPT"
echo "resolution: $RESOLUTION"
echo "resize:     $RESIZE_MODE"
echo "noise:      $NOISE"

singularity exec \
  --overlay "$OVERLAY:ro" \
  --bind /scratch/$USER:/scratch/$USER \
  "$IMAGE" bash -c "
    source /ext3/env.sh
    conda activate jepa
    cd $REPO
    python scripts/diag_old_val_split.py \
      '$CKPT' \
      --feature-cache feature_cache \
      --resolution '$RESOLUTION' \
      --resize-mode '$RESIZE_MODE' \
      --noise '$NOISE'
  "

echo "[$(date)] Done."
