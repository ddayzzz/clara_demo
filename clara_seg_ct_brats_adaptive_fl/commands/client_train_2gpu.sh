#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"

# Data list containing all data
CONFIG_FILE=config/config_train.json
CLIENT_FILE=config/config_fed_client.json
ENVIRONMENT_FILE=config/environment.json

mpirun -np 2 -H localhost:2 -bind-to none -map-by slot \
    -x NCCL_DEBUG=DEBUG -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib --allow-run-as-root \
    python3 -u  -m nvmidl.apps.fed_learn.client.fed_local_train \
    -m $MMAR_ROOT \
    -c $CONFIG_FILE \
    -e $ENVIRONMENT_FILE \
    -s $CLIENT_FILE \
    --set \
    DATASET_JSON=$MMAR_ROOT/config/dataset_0.json \
    secure_train=true \
    multi_gpu=true



