#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"
echo "Experiment name is $1"

# 设置成默认的配置文件
CONFIG_FILE=config/config_train.json
CLIENT_FILE=config/config_fed_client.json
ENVIRONMENT_FILE=config/environment.json
DATA_FILE=$MMAR_ROOT/config/fl_dataset_1.json

if [ -n "$1" ] ;then
    echo "Try to use user-defined files"
    if [ -a "$MMAR_ROOT/config/$1/config_train.json" ];then CONFIG_FILE="config/$1/config_train.json"; fi
    if [ -a "$MMAR_ROOT/config/$1/config_fed_client.json" ];then CLIENT_FILE="config/$1/config_fed_client.json"; fi
    if [ -a "$MMAR_ROOT/config/$1/environment.json" ];then ENVIRONMENT_FILE="config/$1/environment.json"; fi
    if [ -a "$MMAR_ROOT/config/$1/fl_dataset_1.json" ];then DATA_FILE="config/$1/fl_dataset_1.json"; fi
fi

echo "CONFIG_FILE: $CONFIG_FILE"
echo "ENVIRONMENT_FILE: $ENVIRONMENT_FILE"
echo "CLIENT_FILE: $CLIENT_FILE"
echo "DATA_FILE: $DATA_FILE"

python3 -u  -m nvmidl.apps.fed_learn.client.fed_local_train \
    -m $MMAR_ROOT \
    -c $CONFIG_FILE \
    -e $ENVIRONMENT_FILE \
    -s $CLIENT_FILE \
    --set \
    DATASET_JSON=$MMAR_ROOT/$DATA_FILE \
    DATA_LIST_KEY=validation \
    secure_train=false \
    uid=client1
