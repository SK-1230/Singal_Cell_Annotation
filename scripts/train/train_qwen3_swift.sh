#!/usr/bin/env bash
set -euo pipefail

cd /data/projects/shuke/code/singal_cell_annotation

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/data/projects/shuke/code/singal_cell_annotation/my_models/Qwen/Qwen3-4B"
TRAIN_FILE="/data/projects/shuke/code/singal_cell_annotation/data/splits/train_messages_no_think.jsonl"
VAL_FILE="/data/projects/shuke/code/singal_cell_annotation/data/splits/val_messages_no_think.jsonl"
OUTPUT_DIR="/data/projects/shuke/code/singal_cell_annotation/output/qwen3_4b_sc_sft_swift_v2"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path not found: ${MODEL_PATH}"
  exit 1
fi

if [ ! -s "${TRAIN_FILE}" ]; then
  echo "Train file not found or empty: ${TRAIN_FILE}"
  exit 1
fi

if [ ! -s "${VAL_FILE}" ]; then
  echo "Val file not found or empty: ${VAL_FILE}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p /data/projects/shuke/code/singal_cell_annotation/data/meta

echo "===== Training config ====="
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "VAL_FILE=${VAL_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "==========================="

swift sft \
  --model "${MODEL_PATH}" \  
  --train_type lora \  
  --dataset "${TRAIN_FILE}" \  
  --val_dataset "${VAL_FILE}" \  
  --torch_dtype bfloat16 \  
  --num_train_epochs 8 \  
  --per_device_train_batch_size 1 \  
  --per_device_eval_batch_size 1 \  
  --gradient_accumulation_steps 4 \  
  --learning_rate 5e-5 \  
  --lr_scheduler_type cosine \  
  --warmup_ratio 0.05 \  
  --weight_decay 0.01 \  
  --lora_rank 8 \  
  --lora_alpha 32 \  
  --lora_dropout 0.05 \  
  --target_modules all-linear \  
  --max_length 1024 \  
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

echo "Training finished."

# bash train_qwen3_swift.sh 2>&1 | tee data/meta/train_qwen3_swift_v2.log