#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/src/qa_hf_package"

RUN_COUNT="${1:-5}"
BASE_OUTPUT_NAME="bert_trivia_smoke256"
RUN_CONFIG="configs/runs/bert_trivia_qa_rc.json"

cd "$PROJECT_DIR"

for run_index in $(seq 1 "$RUN_COUNT"); do
  run_name="${BASE_OUTPUT_NAME}_run${run_index}"
  echo "=== Run ${run_index}/${RUN_COUNT}: ${run_name} ==="
  python train_qa.py \
    --run-config "$RUN_CONFIG" \
    --set "output_name=${run_name}" \
    --set max_examples=256
done