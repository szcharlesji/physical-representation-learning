#!/bin/bash
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --job-name=jepa-eval-frozen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

# usage:
#   sbatch eval.sbatch <ckpt.pth> [<probes>] [<overrides>]
#
# Defaults: probes="linear knn attentive", overrides resize the inputs to
# 224x224 and disable Gaussian noise on the eval loader so the encoder is
# probed under clean low-res conditions. Override `OVERRIDES` (3rd arg) to
# match training config exactly:
#   sbatch eval.sbatch <ckpt> "linear knn" ""        # use the run's config as-is
#   sbatch eval.sbatch <ckpt> "linear knn" "dataset.resolution=256 train.noise_std=1.0"
#
# `OVERRIDES` is a single shell-quoted, space-separated string of Hydra-style
# dotlist overrides (post_train_probes.py's positional `overrides` arg).

set -euo pipefail

CKPT="${1:?pass the encoder .pth checkpoint path as argument}"
PROBES="${2:-linear knn attentive}" # space-separated, e.g. "linear knn"
# Default to clean (no-noise) eval at the run's training resolution. Set $3 to
# "" to use the run config exactly, or pass another dotlist to change.
# - train.noise_std=0.0     -> _resolved_noise_std() falls back to this when ft is unset
# - ft.noise_std=0.0        -> belt-and-suspenders for paths that read ft.noise_std first
#
# WARNING: Do NOT add `dataset.resolution=...` here unless you know the encoder
# tolerates resize. ViT3DEncoder has a learnable `pos_embed` of shape
# (1, T'*H'*W', D) baked at training resolution; evaluating at a different
# resolution silently slices the first N pos embeddings (see
# physics_jepa/utils/model_utils.py:499) and tanks accuracy. Conv encoders are
# safe.
OVERRIDES="${3:-train.noise_std=0.0 ft.noise_std=0.0}"

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
CKPT_BASE=$(basename "$CKPT" .pth)  # e.g. ConvEncoder_11
RUN_DIR=$(basename "$RUN_DIR_FULL") # e.g. vit3d-depth6_2026-04-24...
OUT_DIR="$REPO/frozen_eval_out/$RUN_DIR/$CKPT_BASE"

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "checkpoint: $CKPT"
echo "run config: $RUN_CONFIG"
echo "probes:     $PROBES"
echo "overrides:  $OVERRIDES"
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
      --probes $PROBES \
      $OVERRIDES
  "

echo "[$(date)] Done — frozen results at $OUT_DIR/results.json"
echo "Headline TEST MSE printed above; rows with split=test are the real held-out numbers."
