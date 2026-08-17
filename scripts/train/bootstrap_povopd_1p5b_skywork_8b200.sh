#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

: "${SWANLAB_API_KEY:?SWANLAB_API_KEY must be set in the environment}"

MODEL_ROOT=${MODEL_ROOT:-/opt/tiger/models}
ACTOR_DIR=${ACTOR_MODEL_PATH:-$MODEL_ROOT/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_DIR=${REWARD_MODEL_PATH:-$MODEL_ROOT/Skywork/Skywork-OR1-Math-7B}
HF_MIRROR=${HF_MIRROR:-https://hf-mirror.com}
TRAIN_LOG=${TRAIN_LOG:-/opt/tiger/povopd_1p5b_skywork7b.log}
PID_FILE=${PID_FILE:-/opt/tiger/povopd_1p5b_skywork7b.pid}

mkdir -p "$ACTOR_DIR" "$TEACHER_DIR"

# The stock image ships a CUDA 12.6 PyTorch wheel whose kernels stop at sm_90.
# B200 is sm_100, so keep the same torch release expected by vLLM while using
# the official CUDA 12.8 build that contains native Blackwell kernels.
if ! python3 -c 'import sys, torch; sys.exit(0 if "sm_100" in torch.cuda.get_arch_list() else 1)'; then
    python3 -m pip install --user --force-reinstall \
        torch==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128
fi

# The base Merlin image currently installs transformers 5.x, while this VERL
# snapshot requires the 4.x AutoModel compatibility aliases.
python3 -m pip install --user \
    transformers==4.55.4 \
    swanlab \
    matplotlib \
    latex2sympy2_extended \
    math_verify

download_file() {
    local repo=$1
    local filename=$2
    local destination=$3
    local temporary="${destination}.part"

    if [[ -s "$destination" ]]; then
        echo "[bootstrap] already present: $destination"
        return
    fi

    mkdir -p "$(dirname "$destination")"
    echo "[bootstrap] downloading: $repo/$filename"
    if [[ -s "$temporary" ]]; then
        if ! curl -fL --retry 10 --retry-delay 5 -C - \
            -o "$temporary" "$HF_MIRROR/$repo/resolve/main/$filename"; then
            rm -f "$temporary"
        fi
    fi
    if [[ ! -s "$temporary" ]]; then
        curl -fL --retry 10 --retry-delay 5 \
            -o "$temporary" "$HF_MIRROR/$repo/resolve/main/$filename"
    fi
    mv "$temporary" "$destination"
}

# Download the five large weight files concurrently.
download_file deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    model.safetensors "$ACTOR_DIR/model.safetensors" &
weight_pids=("$!")
for shard in 00001 00002 00003 00004; do
    filename="model-${shard}-of-00004.safetensors"
    download_file Skywork/Skywork-OR1-Math-7B \
        "$filename" "$TEACHER_DIR/$filename" &
    weight_pids+=("$!")
done
for pid in "${weight_pids[@]}"; do
    wait "$pid"
done

for filename in config.json generation_config.json tokenizer.json tokenizer_config.json; do
    download_file deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        "$filename" "$ACTOR_DIR/$filename"
done
for filename in config.json generation_config.json model.safetensors.index.json \
    special_tokens_map.json tokenizer.json tokenizer_config.json; do
    download_file Skywork/Skywork-OR1-Math-7B \
        "$filename" "$TEACHER_DIR/$filename"
done

ACTOR_DIR="$ACTOR_DIR" TEACHER_DIR="$TEACHER_DIR" OMP_NUM_THREADS=8 python3 - <<'PY'
import glob
import os

from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

actor_dir = os.environ["ACTOR_DIR"]
teacher_dir = os.environ["TEACHER_DIR"]
files = [os.path.join(actor_dir, "model.safetensors")]
files.extend(sorted(glob.glob(os.path.join(teacher_dir, "model-*.safetensors"))))
counts = []
for path in files:
    with safe_open(path, framework="pt") as handle:
        counts.append((os.path.basename(path), len(handle.keys())))
AutoConfig.from_pretrained(actor_dir, local_files_only=True)
AutoTokenizer.from_pretrained(actor_dir, local_files_only=True)
AutoConfig.from_pretrained(teacher_dir, local_files_only=True)
AutoTokenizer.from_pretrained(teacher_dir, local_files_only=True)
print("[bootstrap] model validation passed:", counts)
PY

export ACTOR_MODEL_PATH="$ACTOR_DIR"
export REWARD_MODEL_PATH="$TEACHER_DIR"
export OMP_NUM_THREADS=8

nohup bash scripts/train/run_povopd_1p5b_skywork_8b200.sh \
    >"$TRAIN_LOG" 2>&1 </dev/null &
train_pid=$!
echo "$train_pid" >"$PID_FILE"
sleep 5
if ! kill -0 "$train_pid" 2>/dev/null; then
    echo "[bootstrap] training exited during startup; tail follows" >&2
    tail -n 120 "$TRAIN_LOG" >&2
    exit 1
fi

echo "[bootstrap] training started: pid=$train_pid log=$TRAIN_LOG"
