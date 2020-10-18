

#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"
echo "Experiment name is $1"
export CUDA_VISIBLE_DEVICES=

# 设置成默认的配置文件
CONFIG_FILE=config/config_train.json
SERVER_FILE=config/config_fed_server.json
ENVIRONMENT_FILE=config/environment.json

if [ -n "$1" ] ;then
    echo "Try to use user-defined files"
    if [ -a "$MMAR_ROOT/config/$1/config_train.json" ];then CONFIG_FILE="config/$1/config_train.json"; fi
    if [ -a "$MMAR_ROOT/config/$1/config_fed_server.json" ];then SERVER_FILE="config/$1/config_fed_server.json"; fi
    if [ -a "$MMAR_ROOT/config/$1/environment.json" ];then ENVIRONMENT_FILE="config/$1/environment.json"; fi
fi

echo "CONFIG_FILE: $CONFIG_FILE"
echo "ENVIRONMENT_FILE: $ENVIRONMENT_FILE"
echo "SERVER_FILE: $SERVER_FILE"

python3 -u  -m nvmidl.apps.fed_learn.server.fed_aggregate \
    -m $MMAR_ROOT \
    -c $CONFIG_FILE \
    -e $ENVIRONMENT_FILE \
    -s $SERVER_FILE \
    --set \
    DATASET_JSON=$MMAR_ROOT/config/fl_dataset_0.json \
    MMAR_CKPT_DIR=models/$1 \
    secure_train=false


