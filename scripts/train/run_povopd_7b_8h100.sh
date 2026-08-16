#!/bin/bash

set -euo pipefail

# Eight-H100 launcher for the matched DeepSeek-R1-Distill-Qwen-7B ->
# Skywork-OR1-Math-7B experiment.  The algorithm, data, rollout count,
# validation setup, seed, and optimizer settings remain identical to the
# four-GPU diagnostic; only the distributed resource layout is changed.

MODE=${1:-conflict_attenuation}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export PARALLEL_SIZE=${PARALLEL_SIZE:-2}

# Keep the conservative per-GPU token/memory settings already validated on
# four H100s.  With eight ranks, actor parameters can remain resident while
# optimizer state stays offloaded, avoiding the four-GPU actor-offload cost.
export ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-4}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
export REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-False}
export ACTOR_ROLLOUT_REF_NCCL_TIMEOUT=${ACTOR_ROLLOUT_REF_NCCL_TIMEOUT:-7200}

export PROJECT_PATH=${PROJECT_PATH:-checkpoint/7b_8h100_matched}
export RUN_NAME=${RUN_NAME:-7b8h100_${MODE}_seed${DATA_SEED:-42}_step${TOTAL_TRAINING_STEPS:-40}}

exec bash "$(dirname "$0")/run_povopd_7b_4h100.sh" "$MODE"
