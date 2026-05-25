#!/usr/bin/env bash
set -euo pipefail

DATA_YAML="XXX"
MID_RUN_NAME="XXX"
IMAGE_FOLDER="XXX"
VIDEO_FOLDER="XXX"
PREV_STAGE_CHECKPOINT="XXX"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen_1_5"
GPU_NUM=8
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2

DEEPSPEED_CONFIG="scripts/zero3.json"
MASTER_PORT=30000

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
echo "GLOBAL_BATCH_SIZE: $((GPU_NUM * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"

deepspeed --num_gpus "${GPU_NUM}" --master_port "${MASTER_PORT}" \
    CorrectNav/train/train_mem.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${PREV_STAGE_CHECKPOINT}" \
    --version "${PROMPT_VERSION}" \
    --data_path "${DATA_YAML}" \
    --image_folder "${IMAGE_FOLDER}" \
    --video_folder "${VIDEO_FOLDER}" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower "${VISION_MODEL_VERSION}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name "${MID_RUN_NAME}" \
    --output_dir "./work_dirs/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 12 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2
exit 0;
