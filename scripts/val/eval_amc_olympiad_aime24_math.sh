#!/usr/bin/env bash

# Evaluate one model on AMC23, Olympiad-Bench, AIME24, and MATH-500.
# This is a thin preset over eval_amc23_olympiad.sh, so evaluation behavior,
# answer grading, token metrics, and all resource overrides remain identical.
#
# Example:
#   ACTOR_MODEL_PATH=/path/to/checkpoint bash scripts/val/eval_amc_olympiad_aime24_math.sh
#
# Override the benchmark set when needed:
#   TEST_FILE="['datasets/test_data/AIME24/test.parquet']" \
#     ACTOR_MODEL_PATH=/path/to/checkpoint bash scripts/val/eval_amc_olympiad_aime24_math.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

TEST_DATA_DIR=${TEST_DATA_DIR:-datasets/test_data}
export TEST_FILE=${TEST_FILE:-"['${TEST_DATA_DIR}/AMC23/test.parquet','${TEST_DATA_DIR}/Olympiad-Bench/test.parquet','${TEST_DATA_DIR}/AIME24/test.parquet','${TEST_DATA_DIR}/MATH-500/test.parquet']"}

if [ -z "${EXPERIMENT_NAME:-}" ]; then
    MODEL_NAME=$(basename "${ACTOR_MODEL_PATH:-${MODEL_PATH:-model}}")
    export EXPERIMENT_NAME="eval_${MODEL_NAME}_amc23_olympiad_aime24_math_$(date +%Y-%m-%d_%H-%M-%S)"
fi

exec bash "${SCRIPT_DIR}/eval_amc23_olympiad.sh"
