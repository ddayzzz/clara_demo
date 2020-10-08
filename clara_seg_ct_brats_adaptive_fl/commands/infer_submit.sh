#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"
echo "Inference on round range: start=$1, end=$2, step=$3"
echo "Using data list file: $4"
echo "Output prefix: $5"

start=$1
end=$2
step=$3
for ((i=$start;i<$end;i=i+$step))
do
  echo "Validation on round $i"
  report_path=${MMAR_ROOT}/eval/$5_result_on_round_$i
  mkdir -vp $report_path
  # Data list containing all data
  CONFIG_FILE=config/config_validation.json
  ENVIRONMENT_FILE=config/environment.json

  python3 -u  -m nvmidl.apps.evaluate \
      -m $MMAR_ROOT \
      -c $CONFIG_FILE \
      -e $ENVIRONMENT_FILE \
      --set \
      DATASET_JSON=$4 \
      do_validation=false \
      output_infer_result=true \
      MMAR_EVAL_OUTPUT_PATH=$report_path \
      ROUND_NUM=$i
done
