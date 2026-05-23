#!/usr/bin/env bash
set -euo pipefail

DATASET="$1"
PRETRAINED="$2"
TRAINING_DATA_ROOT_PATH="$3"
LOG_DIR="$4"
ID_NUM="$5"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

for ID in $(seq 0 $((ID_NUM - 1)))
do
  CUDA_VISIBLE_DEVICES="${ID}" PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python data/eval_train_fly.py \
    --dataset "${DATASET}" \
    --id "${ID}" \
    --pretrained "${PRETRAINED}" \
    --training_data_root_path "${TRAINING_DATA_ROOT_PATH}" \
    --log_dir "${LOG_DIR}" &
done

wait
