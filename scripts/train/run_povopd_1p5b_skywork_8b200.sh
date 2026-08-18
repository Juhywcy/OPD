#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

# Matched 1.5B -> Skywork-7B POVOPD run for one node with eight H100/B200 GPUs.
# Algorithmic settings follow run_prefix_trust_oal_opd.sh; only resource
# settings are specialized for the eight-GPU worker.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    GPU_NAME=${GPU_NAME%%$'\n'*}
    case "$GPU_NAME" in
        *B200*) export TORCH_CUDA_ARCH_LIST=10.0 ;;
        *H100*) export TORCH_CUDA_ARCH_LIST=9.0 ;;
    esac
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-/opt/tiger/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/opt/tiger/models/Skywork/Skywork-OR1-Math-7B}

export ADV_ESTIMATOR=${ADV_ESTIMATOR:-prefix_trust_oal_opd}
export OAL_ENABLED=${OAL_ENABLED:-True}
export PT_OAL_OUTCOME_VALIDATION_ENABLED=${PT_OAL_OUTCOME_VALIDATION_ENABLED:-True}
export PT_OAL_PREFIX_TRUST_ENABLED=${PT_OAL_PREFIX_TRUST_ENABLED:-True}
export PT_OAL_WINDOW_SIZE=${PT_OAL_WINDOW_SIZE:-128}
export PT_OAL_FUSION_MODE=${PT_OAL_FUSION_MODE:-conflict_attenuation}
export OPD_TOPK_RENORMALIZE=${OPD_TOPK_RENORMALIZE:-True}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8196}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-31744}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export N_RESPONSES=${N_RESPONSES:-4}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export MODEL_DTYPE=${MODEL_DTYPE:-fp32}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}

export ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-2}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-24}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
export REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-False}
export ACTOR_ROLLOUT_REF_NCCL_TIMEOUT=${ACTOR_ROLLOUT_REF_NCCL_TIMEOUT:-7200}

export DATA_SEED=${DATA_SEED:-42}
export LR_SCHEDULER=${LR_SCHEDULER:-constant}
export TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-False}
export TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','swanlab']"}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-20}
export PROJECT_PATH=${PROJECT_PATH:-/opt/tiger/checkpoint}
export RUN_NAME=${RUN_NAME:-POVOPD_conflict_attenuation_fp32_1p5B_Skywork7B_8GPU_seed42_full1epoch}

# Empty means use trainer.total_epochs=1 over the complete dataset.  A positive
# override remains available for short diagnostics.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS-}

if [[ -d "$PWD/.deps" ]]; then
    export PYTHONPATH="$PWD/.deps:$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
fi

exec bash scripts/train/run_prefix_trust_oal_opd.sh
