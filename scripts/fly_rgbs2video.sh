#!/usr/bin/env bash
set -euo pipefail

RAW_TRAINING_DATA_PATH="$1"
TARGET_TRAINING_DATA_PATH="$2"
PART_NUM="$3"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

for PART_IDX in $(seq 1 "${PART_NUM}")
do
  python data/fly_rgbs2video.py \
    --part_idx "${PART_IDX}" \
    --n_part "${PART_NUM}" \
    --raw_training_data_path "${RAW_TRAINING_DATA_PATH}" \
    --target_training_data_path "${TARGET_TRAINING_DATA_PATH}" &
done

wait
