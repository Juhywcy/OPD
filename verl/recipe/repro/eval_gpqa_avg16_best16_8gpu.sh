#!/usr/bin/env bash
set -euo pipefail

# Eight-GPU verl validation for the local GPQA-Diamond parquet.
#
# Usage:
#   ACTOR_MODEL_PATH=/path/to/hf_checkpoint \
#     bash verl/recipe/repro/eval_gpqa_avg16_best16_8gpu.sh
#
# A positional model path is also accepted:
#   bash verl/recipe/repro/eval_gpqa_avg16_best16_8gpu.sh /path/to/hf_checkpoint
#
# A raw verl FSDP actor is accepted directly and is merged automatically:
#   ACTOR_MODEL_PATH=checkpoint/run/global_step_279/actor \
#     bash verl/recipe/repro/eval_gpqa_avg16_best16_8gpu.sh
# The source actor/model/optimizer/extra shards are never deleted or modified.
# Optional: set FSDP_MERGE_CACHE_DIR, FORCE_MODEL_REMERGE=True, or
# FSDP_STRICT_SHARD_HASH=True to override the cache, rebuild it, or hash every
# large shard respectively.
#
# Defaults target a 1.5B model on 8 x A800: TP=1, DP=8, 16 sampled
# responses per question, and 8192 maximum generated tokens.  The script saves
# every generation, then computes exact avg@16/best@16 (plus pass@16) itself.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-python3}
REQUESTED_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_PATH:-${1:-}}}
if [[ -z "${REQUESTED_MODEL_PATH}" ]]; then
    echo "ERROR: set ACTOR_MODEL_PATH to an HF model/Hub id or a verl actor checkpoint." >&2
    exit 2
fi

is_true() {
    case "${1:-}" in
        1|true|True|TRUE|yes|Yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

GPQA_FILE=${GPQA_FILE:-datasets/test_data/GPQA/test.parquet}
if [[ ! -f "${GPQA_FILE}" ]]; then
    echo "ERROR: GPQA parquet not found: ${GPQA_FILE}" >&2
    exit 2
fi

GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
VISIBLE_GPU_COUNT=${#GPU_ID_ARRAY[@]}
N_GPUS=${N_GPUS:-8}
if (( VISIBLE_GPU_COUNT != N_GPUS )); then
    echo "ERROR: GPU_IDS exposes ${VISIBLE_GPU_COUNT} GPUs, but N_GPUS=${N_GPUS}." >&2
    exit 2
fi

NNODES=${NNODES:-1}
PARALLEL_SIZE=${PARALLEL_SIZE:-1}
if (( N_GPUS % PARALLEL_SIZE != 0 )); then
    echo "ERROR: N_GPUS (${N_GPUS}) must be divisible by PARALLEL_SIZE (${PARALLEL_SIZE})." >&2
    exit 2
fi
DP_SIZE=$((N_GPUS / PARALLEL_SIZE))
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${DP_SIZE}}
if (( PPO_MINI_BATCH_SIZE < DP_SIZE )); then
    echo "ERROR: PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE}) must be at least DP_SIZE (${DP_SIZE})." >&2
    exit 2
fi

N_VAL_RESPONSES=${N_VAL_RESPONSES:-16}
EXPECTED_QUESTIONS=${EXPECTED_QUESTIONS:-198}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH))}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
ENABLE_THINKING=${ENABLE_THINKING:-False}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-True}
if (( N_VAL_RESPONSES <= 0 || VAL_BATCH_SIZE <= 0 )); then
    echo "ERROR: N_VAL_RESPONSES and VAL_BATCH_SIZE must be positive." >&2
    exit 2
fi
if (( MAX_MODEL_LEN < MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH )); then
    echo "ERROR: MAX_MODEL_LEN must cover MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH." >&2
    exit 2
fi

# Resolve a normal HF model directly, or safely merge a raw verl FSDP actor
# into a fingerprinted cache before Ray/vLLM starts using GPU memory.
MODEL_RESOLVER=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_model_for_eval.py"
    --model-path "${REQUESTED_MODEL_PATH}"
)
if [[ -n "${FSDP_MERGE_CACHE_DIR:-}" ]]; then
    MODEL_RESOLVER+=(--cache-dir "${FSDP_MERGE_CACHE_DIR}")
fi
if is_true "${TRUST_REMOTE_CODE}"; then
    MODEL_RESOLVER+=(--trust-remote-code)
fi
if is_true "${FORCE_MODEL_REMERGE:-False}"; then
    MODEL_RESOLVER+=(--force-remerge)
fi
if is_true "${FSDP_STRICT_SHARD_HASH:-False}"; then
    MODEL_RESOLVER+=(--strict-shard-hash)
fi
if is_true "${DRY_RUN:-0}"; then
    MODEL_RESOLVER+=(--dry-run)
fi
ACTOR_MODEL_PATH=$("${MODEL_RESOLVER[@]}")

PROJECT_NAME=${PROJECT_NAME:-verl-val-gpqa}
MODEL_NAME=$(basename "${REQUESTED_MODEL_PATH%/}")
if [[ "${MODEL_NAME}" == "actor" || "${MODEL_NAME}" == "huggingface" ]]; then
    MODEL_NAME=$(basename "$(dirname "${REQUESTED_MODEL_PATH%/}")")
fi
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_${MODEL_NAME}_gpqa_n${N_VAL_RESPONSES}_$(date +%Y-%m-%d_%H-%M-%S)}
VALIDATION_LOG_DIR=${VALIDATION_LOG_DIR:-validation_log}
OUTPUT_DIR="${VALIDATION_LOG_DIR}/${EXPERIMENT_NAME}"
GENERATION_FILE="${OUTPUT_DIR}/0.jsonl"
SUMMARY_FILE="${OUTPUT_DIR}/gpqa_avg${N_VAL_RESPONSES}_best${N_VAL_RESPONSES}.json"

LOG_DIR=${LOG_DIR:-logs/val/gpqa}
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}

VERL_CMD=(
    "${PYTHON_BIN}" -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    data.shuffle=False
    data.validation_shuffle=False
    "data.train_files=${GPQA_FILE}"
    "data.val_files=['${GPQA_FILE}']"
    data.train_batch_size=1
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_VAL_RESP_LENGTH}"
    data.val_max_samples=-1
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
    "data.trust_remote_code=${TRUST_REMOTE_CODE}"
    "+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}"
    "actor_rollout_ref.model.path=${ACTOR_MODEL_PATH}"
    "actor_rollout_ref.model.trust_remote_code=${TRUST_REMOTE_CODE}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=False
    actor_rollout_ref.model.enable_activation_offload=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_NUM_BATCHED_TOKENS}"
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${PARALLEL_SIZE}"
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    "actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE}"
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.dtype=${MODEL_DTYPE}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${PARALLEL_SIZE}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.max_num_seqs=256
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    "actor_rollout_ref.rollout.temperature=${VAL_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${VAL_TOP_P}"
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.calculate_log_probs=False
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}"
    "actor_rollout_ref.rollout.val_kwargs.n=${N_VAL_RESPONSES}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
    "+actor_rollout_ref.rollout.val_kwargs.max_tokens=${MAX_VAL_RESP_LENGTH}"
    critic.enable=False
    reward_model.enable=False
    reward_model.reward_manager=naive
    custom_reward_function.path=verl/recipe/repro/gpqa_reward.py
    custom_reward_function.name=reward_func
    trainer.val_before_train=True
    trainer.val_only=True
    trainer.log_val_generations=0
    "trainer.logger=['console','swanlab']"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.validation_data_dir=${OUTPUT_DIR}"
    "trainer.n_gpus_per_node=${N_GPUS}"
    "trainer.nnodes=${NNODES}"
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=1
    "trainer.default_local_dir=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
)

echo "============================================================"
echo "Requested model:       ${REQUESTED_MODEL_PATH}"
echo "Resolved HF model:     ${ACTOR_MODEL_PATH}"
echo "Dataset:               ${GPQA_FILE}"
echo "CUDA_VISIBLE_DEVICES:  ${CUDA_VISIBLE_DEVICES}"
echo "GPU topology:          TP=${PARALLEL_SIZE}, DP=${DP_SIZE}"
echo "Responses/question:    ${N_VAL_RESPONSES}"
echo "Max generated tokens:  ${MAX_VAL_RESP_LENGTH}"
echo "Generation JSONL:      ${GENERATION_FILE}"
echo "Summary JSON:          ${SUMMARY_FILE}"
echo "Log:                   ${LOG_FILE}"
echo "============================================================"

if is_true "${DRY_RUN:-0}"; then
    printf 'DRY RUN command:'
    printf ' %q' "${VERL_CMD[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if command -v ray >/dev/null 2>&1; then
    ray stop --force || true
    ray start --head
    sleep 5
fi

"${VERL_CMD[@]}"

if [[ ! -s "${GENERATION_FILE}" ]]; then
    echo "ERROR: verl completed without a non-empty generation file: ${GENERATION_FILE}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_gpqa_avg16_best16.py" \
    --input "${GENERATION_FILE}" \
    --n "${N_VAL_RESPONSES}" \
    --expected-questions "${EXPECTED_QUESTIONS}" \
    --output "${SUMMARY_FILE}"

echo "GPQA evaluation completed successfully."
