#!/usr/bin/env bash
set -euo pipefail

DATASET="$1"
TRAINING_DATA_ROOT_PATH="$2"
PART_NUM="$3"

for PART_IDX in $(seq 1 "${PART_NUM}")
do
  python collect_training_data.py \
    --dataset "${DATASET}" \
    --part_idx "${PART_IDX}" \
    --training_data_root_path "${TRAINING_DATA_ROOT_PATH}" &
done

wait
