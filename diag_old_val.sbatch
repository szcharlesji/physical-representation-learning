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
#   sbatch diag_old_val.sbatch <ckpt.pth> [<overrides>]
#
# <overrides>: single shell-quoted dotlist string applied to the run's
# config.yaml, mirroring eval.sbatch's 3rd arg. Default matches eval.sbatch's
# default (clean-noise eval at the run's training resolution), so the cache
# hash matches whatever your y8naqdos / pcyd4gb7 runs wrote.
#
# Example:
#   sbatch diag_old_val.sbatch <ckpt> "train.noise_std=1.0"
#       # if the cache you want to inspect was written with noise=1.0

set -euo pipefail

CKPT="${1:?pass the encoder .pth checkpoint path as argument}"
OVERRIDES="${2:-train.noise_std=0.0 ft.noise_std=0.0}"

REPO=/scratch/$USER/physical-representation-learning
OVERLAY=$REPO/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

RUN_DIR_FULL=$(dirname "$CKPT")
RUN_CONFIG="$RUN_DIR_FULL/config.yaml"
if [[ ! -f "$RUN_CONFIG" ]]; then
  echo "ERROR: $RUN_CONFIG not found."
  exit 2
fi

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "checkpoint: $CKPT"
echo "run config: $RUN_CONFIG"
echo "overrides:  $OVERRIDES"

singularity exec \
  --overlay "$OVERLAY:ro" \
  --bind /scratch/$USER:/scratch/$USER \
  "$IMAGE" bash -c "
    source /ext3/env.sh
    conda activate jepa
    cd $REPO
    python scripts/diag_old_val_split.py \
      --config $RUN_CONFIG \
      --checkpoint $CKPT \
      $OVERRIDES
  "

echo "[$(date)] Done."
