#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"

# Data list containing all data
python3 -u  -m nvmidl.apps.auto_ml.train \
    -m $MMAR_ROOT \
    -c config/config_train_automl.json \
    -n 30 \
    -r a
