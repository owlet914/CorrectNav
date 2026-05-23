#!/usr/bin/env bash
set -euo pipefail

DATASET="$1"
TRAINING_DATA_ROOT_PATH="$2"
PART_NUM="$3"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

for PART_IDX in $(seq 1 "${PART_NUM}")
do
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python data/collect_training_data.py \
    --dataset "${DATASET}" \
    --part_idx "${PART_IDX}" \
    --training_data_root_path "${TRAINING_DATA_ROOT_PATH}" &
done

wait
