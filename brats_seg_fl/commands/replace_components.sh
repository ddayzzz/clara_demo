#!/usr/bin/env bash
SRC_PREFIX=$MMAR_ROOT/components
TARGET_PREFIX=/opt/nvidia/medical
FL_PREFIX=$TARGET_PREFIX/fed_learn

#
echo "Replacing some pre-defined components..."
cp -vf $SRC_PREFIX/server_model_manager.py $FL_PREFIX/server/server_model_manager.py
cp -vf $SRC_PREFIX/client_model_manager.py $FL_PREFIX/client/client_model_manager.py
cp -vf $SRC_PREFIX/fed_server.py $FL_PREFIX/server/fed_server.py
cp -vf $SRC_PREFIX/fed_client.py $FL_PREFIX/client/fed_client.py
cp -vf $SRC_PREFIX/protos/federated_pb2.py $FL_PREFIX/protos/federated_pb2.py
cp -vf $SRC_PREFIX/protos/federated_pb2_grpc.py $FL_PREFIX/protos/federated_pb2_grpc.py
cp -vf $SRC_PREFIX/protos/federated_pb2.py $TARGET_PREFIX/nvmidl/apps/fed_learn/protos/federated_pb2.py
cp -vf $SRC_PREFIX/protos/federated_pb2_grpc.py $TARGET_PREFIX/nvmidl/apps/fed_learn/protos/federated_pb2_grpc.py
cp -vf $SRC_PREFIX/protos/federated.proto $FL_PREFIX/protos/federated.proto