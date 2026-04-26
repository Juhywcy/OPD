#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/run_adaptive_horizon_opd_qwen3_0p6b.sh}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_USED_MEM_MB="${MAX_USED_MEM_MB:-2000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"

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

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GPU memory. Threshold: ${MAX_USED_MEM_MB} MiB"
    nvidia-smi --id="$GPU_IDS" --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits

    if all_gpu_memory_below_threshold; then
        echo "All selected GPUs are below ${MAX_USED_MEM_MB} MiB. Starting training."
        cd "$SCRIPT_DIR"
        exec bash "$TRAIN_SCRIPT"
    fi

    echo "Some selected GPUs are still above threshold. Check again in ${CHECK_INTERVAL_SEC}s."
    sleep "$CHECK_INTERVAL_SEC"
done

