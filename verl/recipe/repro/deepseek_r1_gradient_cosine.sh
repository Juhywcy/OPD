#!/usr/bin/env bash

# One-batch diagnostic only: samples trajectories and measures gradient cosine;
# it does not create an optimizer or update model parameters.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${VERL_ROOT}/.." && pwd)"
cd "${VERL_ROOT}"

# This diagnostic is single-process, so logical cuda:0 is physical GPU 4.
# The remaining visible cards are reserved for a future distributed variant.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

MODEL_PATH="${MODEL_PATH:-/root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
DATA_FILE="${DATA_FILE:-${WORKSPACE_ROOT}/datasets/test_data/DeepScaler/train.parquet}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_RESPONSES="${N_RESPONSES:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
ADVANTAGE_MODE="${ADVANTAGE_MODE:-raw}"
LOSS_AGGREGATION="${LOSS_AGGREGATION:-sequence-mean}"
# Full 8B gradients can be expensive. Set GRADIENT_PARAMETER_REGEX=lm_head for
# a quick directional probe, or leave .* for the exact full-model cosine.
GRADIENT_PARAMETER_REGEX="${GRADIENT_PARAMETER_REGEX:-.*}"
GRADIENT_CHUNK_SIZE="${GRADIENT_CHUNK_SIZE:-64}"

exec python3 -m recipe.repro.gradient_cosine_similarity \
  --model "${MODEL_PATH}" \
  --data "${DATA_FILE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-responses "${N_RESPONSES}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --advantage-mode "${ADVANTAGE_MODE}" \
  --loss-aggregation "${LOSS_AGGREGATION}" \
  --gradient-parameter-regex "${GRADIENT_PARAMETER_REGEX}" \
  --gradient-chunk-size "${GRADIENT_CHUNK_SIZE}"
