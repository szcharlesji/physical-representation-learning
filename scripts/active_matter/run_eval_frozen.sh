#!/bin/bash
source "$(dirname "$0")/../env_setup.sh"

# Pass the pretrained encoder checkpoint path as $1
python -m physics_jepa.eval_frozen \
    --config configs/train_activematter_frozen.yaml \
    --checkpoint "$1"
