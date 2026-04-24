#!/bin/bash
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=jepa-pretrain-exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=pretrain_exp_%j.out
#SBATCH --error=pretrain_exp_%j.err
#SBATCH --signal=B:USR1@120

# Phase 2/3 experiments launcher. Auto-probe is on for all of these so
# every saved checkpoint gets linear + kNN + attentive probes at the end
# of training, clustered under the pretrain run's W&B group.
#
# usage:
#   sbatch pretrain_experiment.sbatch sigreg       # SIGReg regularizer
#   sbatch pretrain_experiment.sbatch vit3d        # ViT3D backbone
#   sbatch pretrain_experiment.sbatch convattn     # conv3d_next + self-attn
#   sbatch pretrain_experiment.sbatch baseline     # identical to pretrain_fft.sbatch activematter

set -euo pipefail

EXP="${1:-sigreg}"
case "$EXP" in
  baseline)  CONFIG=configs/train_activematter_small.yaml ;;
  sigreg)    CONFIG=configs/train_activematter_sigreg.yaml ;;
  vit3d)     CONFIG=configs/train_activematter_vit3d.yaml ;;
  convattn)  CONFIG=configs/train_activematter_convattn.yaml ;;
  *) echo "unknown experiment $EXP (expected baseline|sigreg|vit3d|convattn)"; exit 2 ;;
esac

REPO=/scratch/$USER/physical-representation-learning
OVERLAY=$REPO/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

# Per-experiment checkpoint dir so artifacts don't mingle.
OUT_PATH=$REPO/checkpoints_${EXP}

trap 'echo "[$(date)] SIGUSR1 received — exiting for requeue"; exit 0' USR1

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname) — experiment=$EXP, config=$CONFIG"
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
