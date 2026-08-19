#!/usr/bin/env bash

# Raw sampled-token OPD baseline: Qwen3.5-2B student, Qwen3.5-9B teacher.
# The validation suite covers all paper benchmarks, including GPQA-Diamond.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

# Make the in-tree ``verl`` package importable even when the launcher starts
# from another directory (for example through Merlin/SDP).
export PYTHONPATH="${REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

# Optional isolated runtime for Qwen3.5-capable vLLM wheels. A pip --target
# directory is sufficient because every Ray worker inherits PYTHONPATH.
if [[ -n "${QWEN35_DEPS:-}" ]]; then
    QWEN35_RUNTIME_SHIM="${REPO_ROOT}/verl/recipe/repro/qwen35_runtime"
    export PYTHONPATH="${QWEN35_RUNTIME_SHIM}:${QWEN35_DEPS}${PYTHONPATH:+:${PYTHONPATH}}"
    export PATH="${QWEN35_DEPS}/bin:${PATH}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
fi

# The base image's FlashAttention extension is tied to its original PyTorch.
# Qwen3.5's isolated runtime therefore uses PyTorch SDPA for FSDP models and
# vLLM's Triton backend for rollout generation.
export HF_ATTN_IMPLEMENTATION=${HF_ATTN_IMPLEMENTATION:-sdpa}

AU_PLANNING_ROOT=${AU_PLANNING_ROOT:-/mnt/bn/search-tiktok-nas-au/zhanghaoxin.2025/planning}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${AU_PLANNING_ROOT}/dataset/Qwen3.5-2B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${AU_PLANNING_ROOT}/dataset/Qwen3.5-9B}

export ADV_ESTIMATOR=token_reward_direct
export OAL_ENABLED=False
export PT_OAL_OUTCOME_VALIDATION_ENABLED=False
export PT_OAL_PREFIX_TRUST_ENABLED=False
export OPD_TOPK_RENORMALIZE=True
export LOG_PROB_TOP_K=0
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export N_RESPONSES=${N_RESPONSES:-4}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-1}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.60}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
export REWARD_USE_REMOVE_PADDING=${REWARD_USE_REMOVE_PADDING:-False}
# The 2B actor comfortably fits on 8xH100.  Disabling activation offload avoids
# an incompatibility between verl's legacy saved-tensor hook and PyTorch 2.10.
export ACTOR_ACTIVATION_OFFLOAD=${ACTOR_ACTIVATION_OFFLOAD:-False}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8192}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-8192}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TOP_P=${TOP_P:-0.95}
export ENABLE_THINKING=${ENABLE_THINKING:-True}

export TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-True}
export TEST_FREQ=${TEST_FREQ:-20}
export SAVE_FREQ=${SAVE_FREQ:-20}
export DUMP_VALIDATION_GENERATIONS=${DUMP_VALIDATION_GENERATIONS:-False}
export PROJECT_PATH=${PROJECT_PATH:-${AU_PLANNING_ROOT}/checkpoint}

TEST_DATA_DIR=${TEST_DATA_DIR:-datasets/test_data}
VALIDATION_FILES=(
    "${TEST_DATA_DIR}/AMC23/test.parquet"
    "${TEST_DATA_DIR}/AIME24/test.parquet"
    "${TEST_DATA_DIR}/AIME25/test.parquet"
    "${TEST_DATA_DIR}/MATH-500/test.parquet"
    "${TEST_DATA_DIR}/Minerva/test.parquet"
    "${TEST_DATA_DIR}/GPQA/test.parquet"
)
export TEST_FILE="['${VALIDATION_FILES[0]}','${VALIDATION_FILES[1]}','${VALIDATION_FILES[2]}','${VALIDATION_FILES[3]}','${VALIDATION_FILES[4]}','${VALIDATION_FILES[5]}']"
export CUSTOM_REWARD_FUNCTION_PATH=verl/recipe/repro/math_gpqa_reward.py
export CUSTOM_REWARD_FUNCTION_NAME=reward_func

export PROJECT_NAME=${PROJECT_NAME:-PrefixTrustOALOPD}
export RUN_NAME=${RUN_NAME:-OPD_sampled_token-renormTrue-bf16_stu_Qwen3.5-2B-tch_Qwen3.5-9B}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${RUN_NAME}}

PREFLIGHT=(
    python3 verl/recipe/repro/qwen35_preflight.py
    --student "${ACTOR_MODEL_PATH}"
    --teacher "${REWARD_MODEL_PATH}"
)
for validation_file in "${VALIDATION_FILES[@]}"; do
    PREFLIGHT+=(--dataset "${validation_file}")
done
"${PREFLIGHT[@]}"

if [[ "${PREFLIGHT_ONLY:-False}" =~ ^([Tt]rue|1|[Yy]es)$ ]]; then
    exit 0
fi

exec bash scripts/train/run_prefix_trust_oal_opd.sh
