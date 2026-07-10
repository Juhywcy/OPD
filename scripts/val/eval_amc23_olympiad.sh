#!/usr/bin/env bash
set -euo pipefail
set -x

# Run AMC23 + Olympiad-Bench evaluation through verl's PPO validation engine.
# The generated answers are dumped by verl itself to:
#   ${VALIDATION_LOG_DIR}/${EXPERIMENT_NAME}/0.jsonl
#
# Example:
#   ACTOR_MODEL_PATH=/path/to/checkpoint bash scripts/val/eval_amc23_olympiad.sh
#
# Useful overrides:
#   GPU_IDS=4,5,6,7 N_VAL_RESPONSES=16 MAX_VAL_RESP_LENGTH=24000 \
#   LOGGER="['console','swanlab']" bash scripts/val/eval_amc23_olympiad.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    LOG_DIR=${LOG_DIR:-logs/val}
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/eval_amc23_olympiad_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "=========================================="
    echo "Log file: ${LOG_FILE}"
    echo "Start time: $(date)"
    echo "=========================================="
fi

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}

TEST_DATA_DIR=${TEST_DATA_DIR:-datasets/test_data}
TEST_DATASET=${TEST_FILE:-"['${TEST_DATA_DIR}/AMC23/test.parquet','${TEST_DATA_DIR}/Olympiad-Bench/test.parquet']"}

# main_ppo builds a train dataset even in val_only mode. This file is only used
# for dataloader construction; no training step runs when trainer.val_only=True.
DUMMY_TRAIN_FILE=${DUMMY_TRAIN_FILE:-${TEST_DATA_DIR}/AMC23/test.parquet}

ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_PATH:-}}
if [ -z "${ACTOR_MODEL_PATH}" ]; then
    echo "ERROR: please set ACTOR_MODEL_PATH=/path/to/model or MODEL_PATH=/path/to/model" >&2
    exit 1
fi

ACTOR_MODEL_NAME=$(basename "${ACTOR_MODEL_PATH}")
PROJECT_NAME=${PROJECT_NAME:-verl-val-math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_${ACTOR_MODEL_NAME}_amc23_olympiad_$(date +%Y-%m-%d_%H-%M-%S)}
VALIDATION_LOG_DIR=${VALIDATION_LOG_DIR:-validation_log}

GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}
if [ -n "${GPU_IDS}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
    DEFAULT_N_GPUS=${#GPU_ID_ARRAY[@]}
else
    DEFAULT_N_GPUS=8
fi

N_GPUS=${N_GPUS:-${DEFAULT_N_GPUS}}
NNODES=${NNODES:-1}
PARALLEL_SIZE=${PARALLEL_SIZE:-1}
if (( N_GPUS % PARALLEL_SIZE != 0 )); then
    echo "ERROR: N_GPUS (${N_GPUS}) must be divisible by PARALLEL_SIZE (${PARALLEL_SIZE})." >&2
    exit 1
fi
DP_SIZE=$((N_GPUS / PARALLEL_SIZE))
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${DP_SIZE}}
if (( PPO_MINI_BATCH_SIZE < DP_SIZE )); then
    echo "ERROR: PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE}) must be >= DP_SIZE (${DP_SIZE})." >&2
    echo "verl normalizes it as ppo_mini_batch_size * rollout.n / DP_SIZE during worker init." >&2
    exit 1
fi
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}
LOGGER=${LOGGER:-"['console','swanlab']"}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8192}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-24000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH))}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH))}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}

N_VAL_RESPONSES=${N_VAL_RESPONSES:-16}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
ENABLE_THINKING=${ENABLE_THINKING:-False}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-True}

MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

mkdir -p "${VALIDATION_LOG_DIR}/${EXPERIMENT_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "trainer.n_gpus_per_node=${N_GPUS}"
echo "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"

if command -v ray >/dev/null 2>&1; then
    ray stop --force || true
    ray start --head
    sleep 5
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.shuffle=False \
    data.validation_shuffle=False \
    data.train_files="${DUMMY_TRAIN_FILE}" \
    data.val_files="${TEST_DATASET}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESP_LENGTH}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.trust_remote_code="${TRUST_REMOTE_CODE}" \
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code="${TRUST_REMOTE_CODE}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${PARALLEL_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE}" \
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
    actor_rollout_ref.rollout.dtype="${MODEL_DTYPE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.temperature="${VAL_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${VAL_TOP_P}" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
    actor_rollout_ref.rollout.val_kwargs.n="${N_VAL_RESPONSES}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens="${MAX_VAL_RESP_LENGTH}" \
    critic.enable=False \
    reward_model.enable=False \
    custom_reward_function.path="verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS:-0}" \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.validation_data_dir="${VALIDATION_LOG_DIR}/${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

echo "=========================================="
echo "Validation generations: ${VALIDATION_LOG_DIR}/${EXPERIMENT_NAME}/0.jsonl"
echo "End time: $(date)"
echo "=========================================="
