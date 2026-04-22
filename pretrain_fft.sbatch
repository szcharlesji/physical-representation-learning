#!/bin/bash
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=jepa-pretrain-fft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=pretrain_fft_%j.out
#SBATCH --error=pretrain_fft_%j.err
#SBATCH --signal=B:USR1@120

# FFT-resize pretraining run. Uses native cropped resolution and
# the fft_resize_2d path added on branch try-fft.
# usage: sbatch pretrain_fft.sbatch [shearflow|rayleighbenard|activematter]
#        default: activematter

set -euo pipefail

DATASET="${1:-activematter}"
case "$DATASET" in
  shearflow)      CONFIG=configs/train_shearflow_small.yaml ;;
  rayleighbenard) CONFIG=configs/train_rayleighbenard_small.yaml ;;
  activematter)   CONFIG=configs/train_activematter_small.yaml ;;
  *) echo "unknown dataset $DATASET (expected shearflow|rayleighbenard|activematter)"; exit 2 ;;
esac

REPO=/scratch/$USER/physical-representation-learning
OVERLAY=$REPO/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

# FFT runs live in the same W&B project as the baseline; they're tagged by
# `resize_mode=fft` (auto-added from cfg.dataset.resize_mode) and grouped by
# pretrain run so linear/knn/attentive probes cluster with their checkpoint.
# The run_name suffix "-fft-native" is also set in each config.

# Checkpoints land in a separate subtree so artifact paths stay distinguishable.
OUT_PATH=$REPO/checkpoints_fft

trap 'echo "[$(date)] SIGUSR1 received — exiting for requeue"; exit 0' USR1

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname) — dataset=$DATASET, config=$CONFIG"
echo "out_path=$OUT_PATH"
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
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    torchrun --standalone --nproc_per_node=1 \
      -m physics_jepa.train_jepa \
      $CONFIG \
      out_path=$OUT_PATH
  "

echo "[$(date)] Done"
