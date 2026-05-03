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
#   sbatch diag_old_val.sbatch <train_cache.pt> <val_cache.pt>
#
# Pass cache files by hash directly. From your eval log:
#   train: feature_cache/7e10dd9e7e6f7539.pt
#   val:   feature_cache/61724e8be6e62d7b.pt
#   test:  feature_cache/fd93cf3dd69b5a51.pt
# (val cache is what we sub-sample for the half-split diagnostic; test cache
# isn't needed.)

set -euo pipefail

TRAIN_CACHE="${1:?pass the train cache .pt path}"
VAL_CACHE="${2:?pass the val cache .pt path}"

REPO=/scratch/$USER/physical-representation-learning
OVERLAY=$REPO/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "train cache: $TRAIN_CACHE"
echo "val   cache: $VAL_CACHE"

singularity exec \
  --overlay "$OVERLAY:ro" \
  --bind /scratch/$USER:/scratch/$USER \
  "$IMAGE" bash -c "
    source /ext3/env.sh
    conda activate jepa
    cd $REPO
    python scripts/diag_old_val_split.py \
      --train '$TRAIN_CACHE' \
      --val   '$VAL_CACHE'
  "

echo "[$(date)] Done."
