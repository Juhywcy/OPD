#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

MODE=${1:-conflict_attenuation}
case "$MODE" in
    opd)
        export ADV_ESTIMATOR=token_reward_direct
        export OAL_ENABLED=False
        export PT_OAL_FUSION_MODE=weakest_evidence
        ;;
    conflict_attenuation|conflict_interpolation|weakest_evidence)
        export ADV_ESTIMATOR=prefix_trust_oal_opd
        export OAL_ENABLED=True
        export PT_OAL_FUSION_MODE=$MODE
        ;;
    *)
        echo "Usage: $0 {opd|conflict_attenuation|conflict_interpolation|weakest_evidence}" >&2
        exit 2
        ;;
esac

# Matched 7B diagnostic: every method uses exactly the same data, rollout,
# validation limits, seed, and optimizer settings.  Only MODE changes.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export PARALLEL_SIZE=${PARALLEL_SIZE:-2}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-/opt/tiger/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/opt/tiger/models/Skywork/Skywork-OR1-Math-7B}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-12288}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-31744}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export N_RESPONSES=${N_RESPONSES:-4}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export MODEL_DTYPE=${MODEL_DTYPE:-fp32}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}

# Resource-only changes for four H100s.  They are identical for all methods
# and leave the algorithmic hyperparameters unchanged.
export ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-4}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
export REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-False}

export PT_OAL_OUTCOME_VALIDATION_ENABLED=${PT_OAL_OUTCOME_VALIDATION_ENABLED:-True}
export PT_OAL_PREFIX_TRUST_ENABLED=${PT_OAL_PREFIX_TRUST_ENABLED:-True}
export PT_OAL_WINDOW_SIZE=${PT_OAL_WINDOW_SIZE:-128}
export OPD_TOPK_RENORMALIZE=${OPD_TOPK_RENORMALIZE:-True}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}

export DATA_SEED=${DATA_SEED:-42}
export TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-True}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-40}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-20}
export TRAINER_LOGGER=${TRAINER_LOGGER:-"['console']"}
export PROJECT_PATH=${PROJECT_PATH:-checkpoint/7b_4h100_matched}
export RUN_NAME=${RUN_NAME:-7b4h100_${MODE}_seed${DATA_SEED}_step${TOTAL_TRAINING_STEPS}}

if [ -d "$PWD/.deps" ]; then
    export PYTHONPATH="$PWD/.deps:$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
fi

exec bash scripts/train/run_prefix_trust_oal_opd.sh
