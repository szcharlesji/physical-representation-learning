#!/bin/bash
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=jepa-eval-curve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=eval_curve_%j.out
#SBATCH --error=eval_curve_%j.err

# One sbatch -> one wandb run with 9 curves (linear/knn/attentive x alpha/zeta/mean)
# across every checkpoint in the run directory. Replaces the per-checkpoint
# `eval.sbatch` + `attentive.sbatch` shell loop.
#
# usage:
#   sbatch eval_all.sbatch <run_dir>
#   sbatch eval_all.sbatch <run_dir> "linear knn"                      # skip attentive
#   sbatch eval_all.sbatch <run_dir> "linear knn attentive" <fcfg>     # custom frozen cfg
#
# <run_dir> must contain ConvEncoder_*.pth and config.yaml (training writes
# config.yaml on first save). The 3rd arg is needed only for runs whose
# config.yaml predates the post_train_eval block; otherwise we read
# cfg.post_train_eval.frozen_config and the default below is unused.

set -euo pipefail

RUN_DIR="${1:?pass run directory containing checkpoints + config.yaml}"
PROBES="${2:-linear knn attentive}"
FROZEN_CONFIG="${3:-configs/train_activematter_frozen.yaml}"

REPO=/scratch/$USER/physical-representation-learning
OVERLAY=$REPO/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

if [[ ! -f "$RUN_DIR/config.yaml" ]]; then
  echo "ERROR: $RUN_DIR/config.yaml not found"
  exit 2
fi

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "run_dir:       $RUN_DIR"
echo "probes:        $PROBES"
echo "frozen_config: $FROZEN_CONFIG (used only if cfg.post_train_eval.frozen_config is unset)"
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
    python -m physics_jepa.eval_run \
      --ckpt_dir $RUN_DIR \
      --probes $PROBES \
      --frozen_config $FROZEN_CONFIG
  "

echo "[$(date)] Done — eval_curve.json under $RUN_DIR"
