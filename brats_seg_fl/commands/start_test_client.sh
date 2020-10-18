#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"

# Data list containing all data
CONFIG_FILE=config/config_train.json
CLIENT_FILE=config/config_fed_client.json
ENVIRONMENT_FILE=config/environment.json

python3 -u  -m nvmidl.apps.fed_learn.client.test_client \
    -m $MMAR_ROOT \
    -s $CLIENT_FILE \
    --set \
    secure_train=false \
    uid=$1

