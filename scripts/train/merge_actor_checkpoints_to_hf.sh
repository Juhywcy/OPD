#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoint}
BACKEND=${BACKEND:-fsdp}
TARGET_NAME=${TARGET_NAME:-huggingface}
DELETE_ORIGINAL=${DELETE_ORIGINAL:-True}
FORCE_REMERGE=${FORCE_REMERGE:-False}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-False}
TIE_WORD_EMBEDDING=${TIE_WORD_EMBEDDING:-False}
USE_CPU_INITIALIZATION=${USE_CPU_INITIALIZATION:-False}

if [ ! -d "$CHECKPOINT_ROOT" ]; then
    echo "Checkpoint root not found: $CHECKPOINT_ROOT"
    exit 1
fi

bool_arg() {
    case "$1" in
        True|true|1|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

has_hf_model_files() {
    local target_dir="$1"

    [ -f "$target_dir/config.json" ] || return 1
    find "$target_dir" -maxdepth 1 \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) | grep -q .
}

merge_one_actor() {
    local actor_dir="$1"
    local target_dir="${actor_dir}/${TARGET_NAME}"
    local merge_args=()

    if [ "$BACKEND" = "fsdp" ] && [ ! -f "$actor_dir/fsdp_config.json" ]; then
        echo "[skip] Missing fsdp_config.json: $actor_dir"
        return 0
    fi

    if has_hf_model_files "$target_dir" && ! bool_arg "$FORCE_REMERGE"; then
        echo "[skip] HuggingFace model already exists: $target_dir"
    else
        mkdir -p "$target_dir"

        merge_args=(
            python3 -m verl.model_merger merge
            --backend "$BACKEND"
            --local_dir "$actor_dir"
            --target_dir "$target_dir"
        )

        if bool_arg "$TRUST_REMOTE_CODE"; then
            merge_args+=(--trust-remote-code)
        fi
        if bool_arg "$TIE_WORD_EMBEDDING"; then
            merge_args+=(--tie-word-embedding)
        fi
        if bool_arg "$USE_CPU_INITIALIZATION"; then
            merge_args+=(--use_cpu_initialization)
        fi

        echo "[merge] $actor_dir -> $target_dir"
        "${merge_args[@]}"

        if ! has_hf_model_files "$target_dir"; then
            echo "[error] Merge finished, but HuggingFace files were not found in: $target_dir"
            return 1
        fi
    fi

    if bool_arg "$DELETE_ORIGINAL"; then
        echo "[cleanup] Removing original actor checkpoint states under: $actor_dir"
        find "$actor_dir" -mindepth 1 -maxdepth 1 ! -name "$TARGET_NAME" -exec rm -rf {} +
    else
        echo "[keep] DELETE_ORIGINAL=False, original actor states kept: $actor_dir"
    fi
}

mapfile -t actor_dirs < <(find "$CHECKPOINT_ROOT" -type d -path "*/global_step_*/actor" | sort)

if [ "${#actor_dirs[@]}" -eq 0 ]; then
    echo "No actor checkpoint directories found under: $CHECKPOINT_ROOT"
    exit 0
fi

echo "Found ${#actor_dirs[@]} actor checkpoint directories under $CHECKPOINT_ROOT"
echo "BACKEND=$BACKEND DELETE_ORIGINAL=$DELETE_ORIGINAL FORCE_REMERGE=$FORCE_REMERGE TARGET_NAME=$TARGET_NAME"

for actor_dir in "${actor_dirs[@]}"; do
    merge_one_actor "$actor_dir"
done

echo "Done."
