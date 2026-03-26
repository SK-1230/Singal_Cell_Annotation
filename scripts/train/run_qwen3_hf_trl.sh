#!/usr/bin/env bash
set -euo pipefail

cd /data/projects/shuke/code/singal_cell_annotation

export CUDA_VISIBLE_DEVICES=0

PYTHON_BIN="python"
SCRIPT_PATH="/data/projects/shuke/code/singal_cell_annotation/train_qwen3_hf_trl.py"
LOG_DIR="/data/projects/shuke/code/singal_cell_annotation/data/meta"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/train_qwen3_hf_trl_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "===== Launch training ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SCRIPT_PATH=${SCRIPT_PATH}"
echo "LOG_FILE=${LOG_FILE}"
echo "==========================="

"${PYTHON_BIN}" "${SCRIPT_PATH}" 2>&1 | tee "${LOG_FILE}"
