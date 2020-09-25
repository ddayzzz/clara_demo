#!/usr/bin/env bash

export PYTHONPATH="$PYTHONPATH:/opt/nvidia"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export MMAR_ROOT=${DIR}/..
export PYTHONPATH=$PYTHONPATH:${MMAR_ROOT}/componets
# replace some files
chmod +x $MMAR_ROOT/commands/replace_components.sh
$MMAR_ROOT/commands/replace_components.sh