#!/bin/bash
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=jepa-eval-frozen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

# usage: sbatch eval.sbatch /scratch/$USER/physical-representation-learning/checkpoints/<run>/ConvEncoder_<epoch>.pth

set -euo pipefail

CKPT="${1:?pass the encoder .pth checkpoint path as argument}"
PROBES="${2:-linear knn attentive}"   # space-separated, e.g. "linear knn"

REPO=/scratch/$USER/physical-representation-learning
OVERLAY=$REPO/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

# Use the run's saved config.yaml so dataset/model/ft match what the
# encoder was trained with (resize_mode, resolution, backbone, etc.).
RUN_DIR_FULL=$(dirname "$CKPT")
RUN_CONFIG="$RUN_DIR_FULL/config.yaml"
if [[ ! -f "$RUN_CONFIG" ]]; then
  echo "ERROR: $RUN_CONFIG not found. Re-run training to write the run config, or fall back to:"
  echo "  python -m physics_jepa.eval_frozen --config <matching-train-yaml> --checkpoint $CKPT \\"
  echo "      dataset.resize_mode=<bilinear|fft> dataset.resolution=<H,W>"
  exit 2
fi

# Per-checkpoint output dir mirrors post_train_probes' layout.
CKPT_BASE=$(basename "$CKPT" .pth)               # e.g. ConvEncoder_11
RUN_DIR=$(basename "$RUN_DIR_FULL")              # e.g. vit3d-depth6_2026-04-24...
OUT_DIR="$REPO/frozen_eval_out/$RUN_DIR/$CKPT_BASE"

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "checkpoint: $CKPT"
echo "run config: $RUN_CONFIG"
echo "probes:     $PROBES"
echo "out dir:    $OUT_DIR"
nvidia-smi

singularity exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /scratch/$USER:/scratch/$USER \
  "$IMAGE" bash -c "
    source /ext3/env.sh
    conda activate jepa
    cd $REPO
    export THE_WELL_DATA_DIR=$REPO/datasets
    export HDF5_USE_FILE_LOCKING=FALSE
    export WANDB_API_KEY=\${WANDB_API_KEY:-}
    python -m physics_jepa.post_train_probes \
      --config $RUN_CONFIG \
      --checkpoint $CKPT \
      --probes $PROBES
  "

echo "[$(date)] Done — frozen results at $OUT_DIR/results.json"
echo "Headline TEST MSE printed above; rows with split=test are the real held-out numbers."
