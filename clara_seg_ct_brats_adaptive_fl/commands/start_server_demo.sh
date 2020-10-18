

#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"
echo "Experiment name is $1"
export CUDA_VISIBLE_DEVICES=

# Data list containing all data
CONFIG_FILE=config/config_train.json
SERVER_FILE=config/config_fed_server/$1.json
ENVIRONMENT_FILE=config/environment.json

python3 -u  -m nvmidl.apps.fed_learn.server.fed_aggregate \
    -m $MMAR_ROOT \
    -c $CONFIG_FILE \
    -e $ENVIRONMENT_FILE \
    -s $SERVER_FILE \
    --set \
    DATASET_JSON=$MMAR_ROOT/config/fl_dataset_0.json \
    MMAR_CKPT_DIR="models/$1" \
    secure_train=false


