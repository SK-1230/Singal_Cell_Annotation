#!/usr/bin/env bash
# train_qwen3_swift_v2.sh
#
# SFT fine-tuning with annotation_output_v2 schema (no-think mode).
#
# Key differences from train_qwen3_swift.sh:
#   - Uses v2 split files (train_messages_no_think_v2.jsonl / val_messages_no_think_v2.jsonl)
#   - Output dir uses a distinct "_v2" suffix to keep checkpoints separate
#   - All paths and key hyperparameters are overridable via environment variables
#   - max_length bumped to 1536 to accommodate the richer v2 prompts/answers
#
# Override examples:
#   MODEL_PATH=/path/to/model bash train_qwen3_swift_v2.sh
#   MAX_LENGTH=2048 EPOCHS=5 bash train_qwen3_swift_v2.sh
set -euo pipefail

cd /data/projects/shuke/code/singal_cell_annotation

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-/data/projects/shuke/code/singal_cell_annotation/my_models/Qwen/Qwen3-4B}"
TRAIN_FILE="${TRAIN_FILE:-/data/projects/shuke/code/singal_cell_annotation/data/splits/train_messages_no_think_v2.jsonl}"
VAL_FILE="${VAL_FILE:-/data/projects/shuke/code/singal_cell_annotation/data/splits/val_messages_no_think_v2.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/projects/shuke/code/singal_cell_annotation/output/qwen3_4b_sc_sft_swift_v2_schema}"

MAX_LENGTH="${MAX_LENGTH:-1536}"
EPOCHS="${EPOCHS:-8}"
LR="${LR:-5e-5}"

# -------------------------------------------------------
# Guard: verify required paths exist
# -------------------------------------------------------
if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: Model path not found: ${MODEL_PATH}"
  exit 1
fi

if [ ! -s "${TRAIN_FILE}" ]; then
  echo "ERROR: Train file not found or empty: ${TRAIN_FILE}"
  echo "       Run 05_make_sft_jsonl.py and 06_split_and_validate_v2.py first."
  exit 1
fi

if [ ! -s "${VAL_FILE}" ]; then
  echo "WARNING: Val file not found or empty: ${VAL_FILE}"
  echo "         Training will proceed without evaluation."
  USE_VAL=false
else
  USE_VAL=true
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p /data/projects/shuke/code/singal_cell_annotation/data/meta

echo "===== train_qwen3_swift_v2.sh config ====="
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "VAL_FILE=${VAL_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "EPOCHS=${EPOCHS}"
echo "LR=${LR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "USE_V2_SCHEMA=true (annotation_output_v2, no-think)"
echo "==========================================="

# -------------------------------------------------------
# Build the val_dataset argument conditionally
# -------------------------------------------------------
VAL_ARGS=""
if [ "${USE_VAL}" = "true" ]; then
  VAL_ARGS="--val_dataset ${VAL_FILE}"
fi

swift sft \
  --model "${MODEL_PATH}" \
  --train_type lora \
  --dataset "${TRAIN_FILE}" \
  ${VAL_ARGS} \
  --torch_dtype bfloat16 \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate "${LR}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --max_length "${MAX_LENGTH}" \
  --logging_steps 1 \
  --eval_steps 5 \
  --save_steps 5 \
  --save_total_limit 2 \
  --dataset_num_proc 1 \
  --dataloader_num_workers 2 \
  --gradient_checkpointing true \
  --loss_scale ignore_empty_think \
  --metric_for_best_model loss \
  --greater_is_better false \
  --output_dir "${OUTPUT_DIR}"

echo "Training finished. Checkpoints saved to: ${OUTPUT_DIR}"

# bash scripts/train/train_qwen3_swift_v2.sh 2>&1 | tee data/meta/train_qwen3_swift_v2_schema.log
