#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/run_outcome_aligned_logit_opd.sh}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-7000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"

OAL_EXPERIMENTS="${OAL_EXPERIMENTS:-8:only_stu:True:oal,32:only_stu:True:oal,16:union:True:oal,16:only_stu:False:oal}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

all_gpu_memory_below_threshold() {
    local gpu_id
    local used_mem

    IFS=',' read -r -a gpu_array <<< "$GPU_IDS"

    for gpu_id in "${gpu_array[@]}"; do
        used_mem="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits)"
        used_mem="${used_mem//[[:space:]]/}"

        if [ "$used_mem" -ge "$MAX_USED_MEM_MB" ]; then
            return 1
        fi
    done

    return 0
}

wait_for_available_gpus() {
    while true; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GPU memory. Threshold: ${MAX_USED_MEM_MB} MiB"
        nvidia-smi --id="$GPU_IDS" --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits

        if all_gpu_memory_below_threshold; then
            echo "All selected GPUs are below ${MAX_USED_MEM_MB} MiB."
            return 0
        fi

        echo "Some selected GPUs are still above threshold. Check again in ${CHECK_INTERVAL_SEC}s."
        sleep "$CHECK_INTERVAL_SEC"
    done
}

IFS=',' read -r -a experiment_array <<< "$OAL_EXPERIMENTS"

for experiment in "${experiment_array[@]}"; do
    experiment="${experiment//[[:space:]]/}"
    IFS=':' read -r topk topk_strategy topk_renormalize split_mode <<< "$experiment"
    wait_for_available_gpus
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running LOG_PROB_TOP_K=${topk}, TOP_K_STRATEGY=${topk_strategy}, OPD_TOPK_RENORMALIZE=${topk_renormalize}, OAL_SPLIT_MODE=${split_mode}"
    LOG_PROB_TOP_K="$topk" \
    TOP_K_STRATEGY="$topk_strategy" \
    OPD_TOPK_RENORMALIZE="$topk_renormalize" \
    OAL_SPLIT_MODE="$split_mode" \
        bash "$TRAIN_SCRIPT"
done

echo "All OAL ablation experiments finished."
