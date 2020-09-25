#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"
additional_options="$*"

# Data list containing all data
CONFIG_FILE=config/config_train.json
ENVIRONMENT_FILE=config/environment.json

python3 -u  -m nvmidl.apps.train \
    -m $MMAR_ROOT \
    -c $CONFIG_FILE \
    -e $ENVIRONMENT_FILE \
    --set \
    DATASET_JSON=$MMAR_ROOT/config/dataset_0.json \
    PRETRAIN_WEIGHTS_FILE="" \
    epochs=1000 \
    learning_rate=0.0001 \
    num_training_epoch_per_valid=20 \
    MMAR_CKPT=$MMAR_ROOT/models/model.ckpt \
    multi_gpu=false \
    ${additional_options}
