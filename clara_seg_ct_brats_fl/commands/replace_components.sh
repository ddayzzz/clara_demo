#!/usr/bin/env bash
SRC_PREFIX=$MMAR_ROOT/components
TARGET_PREFIX=/opt/nvidia/medical
FL_PREFIX=$TARGET_PREFIX/fed_learn

#
echo "Replacing some pre-defined components..."
cp -vf $SRC_PREFIX/server_model_manager.py $FL_PREFIX/server/server_model_manager.py
cp -vf $SRC_PREFIX/fed_server.py $FL_PREFIX/server/fed_server.py
