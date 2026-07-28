#!/usr/bin/env bash

# Experiments 3 and 4 from paper/OAL/LaTeX/experiment_todo.md.
# This script only performs generation/forward/backward diagnostics. It never
# constructs an optimizer and never updates MODEL_PATH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the final POV-OPD Hugging Face checkpoint}"
: "${DATA_FILE:?Set DATA_FILE to a frozen held-out VERL-format parquet file}"

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/root/models/hbx/JustRL-DeepSeek-1.5B}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/audit_results/pov_final}"

[[ "${MODEL_PATH}" = /* ]] || MODEL_PATH="${REPO_ROOT}/${MODEL_PATH}"
[[ "${TEACHER_MODEL_PATH}" = /* ]] || TEACHER_MODEL_PATH="${REPO_ROOT}/${TEACHER_MODEL_PATH}"
[[ "${DATA_FILE}" = /* ]] || DATA_FILE="${REPO_ROOT}/${DATA_FILE}"
[[ "${OUTPUT_DIR}" = /* ]] || OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
cd "${REPO_ROOT}/verl"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

BATCH_PROMPTS="${BATCH_PROMPTS:-1}"
NUM_BATCHES="${NUM_BATCHES:-64}"
N_RESPONSES="${N_RESPONSES:-4}"
DATASET_OFFSET="${DATASET_OFFSET:-0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
SCORE_MICRO_BATCH_SIZE="${SCORE_MICRO_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bf16}"
OUTCOME_ADVANTAGE_MODE="${OUTCOME_ADVANTAGE_MODE:-zscore}"
LOSS_AGGREGATION="${LOSS_AGGREGATION:-token-mean}"

# This default intentionally produces a gradient-subspace audit. Use ".*" for
# the full model only after checking the memory/runtime cost on a small run.
GRADIENT_PARAMETER_REGEX="${GRADIENT_PARAMETER_REGEX:-lm_head}"
REFERENCE_NORM_THRESHOLD="${REFERENCE_NORM_THRESHOLD:-1e-10}"
CANDIDATE_NORM_THRESHOLD="${CANDIDATE_NORM_THRESHOLD:-1e-10}"

PT_OAL_WINDOW_SIZE="${PT_OAL_WINDOW_SIZE:-128}"
PT_OAL_BASELINE_BLOCKS="${PT_OAL_BASELINE_BLOCKS:-2}"

LENGTH_CALIPER="${LENGTH_CALIPER:-0.20}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
SEED="${SEED:-42}"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=true

echo "[pov-audit] output: ${OUTPUT_DIR}"
echo "[pov-audit] GPUs: ${CUDA_VISIBLE_DEVICES}; workers: ${NPROC_PER_NODE}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m recipe.repro.pov_gradient_audit \
  --model "${MODEL_PATH}" \
  --teacher-model "${TEACHER_MODEL_PATH}" \
  --data "${DATA_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-prompts "${BATCH_PROMPTS}" \
  --num-batches "${NUM_BATCHES}" \
  --num-responses "${N_RESPONSES}" \
  --dataset-offset "${DATASET_OFFSET}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --score-micro-batch-size "${SCORE_MICRO_BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --outcome-advantage-mode "${OUTCOME_ADVANTAGE_MODE}" \
  --loss-aggregation "${LOSS_AGGREGATION}" \
  --gradient-parameter-regex "${GRADIENT_PARAMETER_REGEX}" \
  --reference-norm-threshold "${REFERENCE_NORM_THRESHOLD}" \
  --candidate-norm-threshold "${CANDIDATE_NORM_THRESHOLD}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --prefix-window-size "${PT_OAL_WINDOW_SIZE}" \
  --prefix-baseline-blocks "${PT_OAL_BASELINE_BLOCKS}" \
  --seed "${SEED}"

python -m recipe.repro.prefix_horizon_audit \
  --audit-records "${OUTPUT_DIR}/batches.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --length-caliper "${LENGTH_CALIPER}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --seed "${SEED}"

echo "[pov-audit] gradient summary: ${OUTPUT_DIR}/gradient_summary.json"
echo "[pov-audit] prefix summary:   ${OUTPUT_DIR}/prefix_horizon_summary.json"
