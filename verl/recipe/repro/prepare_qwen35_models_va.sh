#!/usr/bin/env bash

# Run on a VA development machine with the search-nlp-vagcp mount.  Models are
# downloaded once in VA and copied to the AU ByteNAS namespace used by Merlin.

set -euo pipefail

VA_MODEL_ROOT=${VA_MODEL_ROOT:-/mnt/bn/search-nlp-vagcp/zhanghaoxin/planning}
AU_MODEL_URI_ROOT=${AU_MODEL_URI_ROOT:-bytenas://i18n-production:search-tiktok-nas-au/zhanghaoxin.2025/planning/dataset}
HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export HF_ENDPOINT

command -v hf >/dev/null 2>&1 || {
    echo "ERROR: Hugging Face 'hf' CLI is not installed on the VA machine." >&2
    exit 1
}
command -v nastk >/dev/null 2>&1 || {
    echo "ERROR: nastk is not installed on the VA machine." >&2
    exit 1
}

mkdir -p "${VA_MODEL_ROOT}"

download_and_copy() {
    local repo_id=$1
    local model_name=${repo_id#*/}
    local local_dir="${VA_MODEL_ROOT}/${model_name}"

    hf download "${repo_id}" --local-dir "${local_dir}"
    test -s "${local_dir}/config.json"
    find "${local_dir}" -maxdepth 1 -name '*.safetensors' -type f | grep -q .

    echo "Copying ${local_dir} to ${AU_MODEL_URI_ROOT}/"
    nastk cp "${local_dir}" "${AU_MODEL_URI_ROOT}/"
}

download_and_copy Qwen/Qwen3.5-2B
download_and_copy Qwen/Qwen3.5-9B

echo "Qwen3.5 model transfer to AU completed."
