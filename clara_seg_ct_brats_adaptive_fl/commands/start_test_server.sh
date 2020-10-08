#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"

export CUDA_VISIBLE_DEVICES=

# Data list containing all data
CONFIG_FILE=config/config_train.json
SERVER_FILE=config/config_fed_server.json
ENVIRONMENT_FILE=config/environment.json

python3 -u  -m nvmidl.apps.fed_learn.server.test_server \
    -m $MMAR_ROOT \
    -s $SERVER_FILE \
    --set \
    secure_train=true
