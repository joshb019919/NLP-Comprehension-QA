#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PY="${MAIN_PY:-$SCRIPT_DIR/src/main.py}"

COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-5}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-40}"
HEALTH_CHECK_INTERVAL_SECONDS="${HEALTH_CHECK_INTERVAL_SECONDS:-5}"
MIN_AVAILABLE_MEM_MB="${MIN_AVAILABLE_MEM_MB:-2048}"
MAX_GPU_MEMORY_USED_MB="${MAX_GPU_MEMORY_USED_MB:-512}"
ENABLE_GPU_HEALTH_CHECKS="${ENABLE_GPU_HEALTH_CHECKS:-1}"
REQUIRE_SUMMARY_METRICS="${REQUIRE_SUMMARY_METRICS:-0}"
ACTIVE_CHILD_PID=""
ACTIVE_CHILD_PGID=""

COMMON_OVERRIDES=(
  # "max_examples=1"
)

cleanup_active_child() {
  local signal="${1:-INT}"

  if [[ -n "${ACTIVE_CHILD_PGID}" ]]; then
    kill "-${signal}" -- "-${ACTIVE_CHILD_PGID}" 2>/dev/null || true
  elif [[ -n "${ACTIVE_CHILD_PID}" ]]; then
    kill "-${signal}" "${ACTIVE_CHILD_PID}" 2>/dev/null || true
  fi
}

handle_interrupt() {
  local signal_name="${1:-SIGINT}"
  local exit_code=130

  if [[ "${signal_name}" == "SIGTERM" ]]; then
    exit_code=143
  fi

  echo
  echo "[signal] Received ${signal_name}. Stopping the active experiment..." >&2
  cleanup_active_child "${signal_name#SIG}"

  if [[ -n "${ACTIVE_CHILD_PID}" ]]; then
    wait "${ACTIVE_CHILD_PID}" 2>/dev/null || true
  fi

  exit "${exit_code}"
}

trap 'handle_interrupt SIGINT' INT
trap 'handle_interrupt SIGTERM' TERM

print_separator() {
  echo "============================================================"
}

available_mem_mb() {
  awk '/MemAvailable:/ { printf "%d\n", $2 / 1024 }' /proc/meminfo
}

gpu_health_checks_enabled() {
  [[ "$ENABLE_GPU_HEALTH_CHECKS" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1
}

max_gpu_memory_used_mb() {
  if ! gpu_health_checks_enabled; then
    echo 0
    return
  fi

  local max_used
  max_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN { max = 0 } { if ($1 > max) max = $1 } END { print max }')"
  echo "${max_used:-0}"
}

print_health_snapshot() {
  local label="$1"
  local mem_available
  mem_available="$(available_mem_mb)"
  echo "[health] ${label}: available RAM ${mem_available} MiB"

  if gpu_health_checks_enabled; then
    echo "[health] ${label}: GPU memory snapshot"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  fi
}

summary_metrics_path() {
  local run_name="$1"
  find /data/data/huggingface/runs -maxdepth 2 -type f -path "*/${run_name}_*/summary_metrics.json" | head -n 1
}

post_run_verification() {
  local run_name="$1"
  local summary_path=""

  print_health_snapshot "after ${run_name}"
  sync

  if [[ "$COOLDOWN_SECONDS" -gt 0 ]]; then
    echo "[health] Cooling down for ${COOLDOWN_SECONDS}s before the next experiment."
    sleep "$COOLDOWN_SECONDS"
  fi

  if [[ "$REQUIRE_SUMMARY_METRICS" == "1" ]]; then
    summary_path="$(summary_metrics_path "$run_name")"
    if [[ ! -f "$summary_path" ]]; then
      echo "[verify] Expected summary metrics at $summary_path but none were found." >&2
      return 1
    fi
    echo "[verify] Found summary metrics: $summary_path"
  fi
}

run_exp() {
  local name="$1"
  local run_config="$2"
  shift 2
  local common_args=()
  local override
  local rc

  for override in "${COMMON_OVERRIDES[@]}"; do
    common_args+=(--set "$override")
  done

  print_separator
  echo "Running: ${name}"
  echo "Run config: ${run_config}"
  print_health_snapshot "before ${name}"
  print_separator

  if command -v setsid >/dev/null 2>&1; then
    setsid "${PYTHON_BIN}" "${MAIN_PY}" \
      --run-config "${run_config}" \
      --set "output_name=${name}" \
      "${common_args[@]}" \
      "$@" &
  else
    "${PYTHON_BIN}" "${MAIN_PY}" \
      --run-config "${run_config}" \
      --set "output_name=${name}" \
      "${common_args[@]}" \
      "$@" &
  fi

  ACTIVE_CHILD_PID=$!
  ACTIVE_CHILD_PGID=$ACTIVE_CHILD_PID

  if wait "${ACTIVE_CHILD_PID}"; then
    rc=0
  else
    rc=$?
  fi
  ACTIVE_CHILD_PID=""
  ACTIVE_CHILD_PGID=""

  if (( rc != 0 )); then
    echo "[run] Experiment ${name} exited with status ${rc}." >&2
    return "$rc"
  fi

  echo "[run] Experiment ${name} completed successfully."
  post_run_verification "$name"
}

# 1. BERT-base, TriviaQA -> tested on SQuAD 1.1, lr 5e-5, wd 1e-8, 2 ep, adamw_torch, clip 1.0
run_exp "exp01_bertbase_triviaqa_to_squad11_lr5e5_wd1e8_ep2_adamw_clip1p0" "configs/runs/bert_trivia_qa_rc.json" \
  --set "model.model_name_or_path=bert-base-uncased"   \
  --set "dataset.test_dataset_name=rajpurkar/squad"    \
  --set "dataset.test_dataset_config_name=null"        \
  --set "dataset.test_version_2_with_negative=false"   \
  --set "dataset.test_split=validation"                \
  --set "run.learning_rate=5e-5"                       \
  --set "run.weight_decay=1e-8"                        \
  --set "run.epochs=2"                                 \
  --set "run.optim=adamw_torch_fused"                  \
  --set "run.max_grad_norm=1.0"

# 2. clip 0.5
run_exp "exp02_bertbase_triviaqa_to_squad11_lr5e5_wd1e8_ep2_adamw_clip0p5" "configs/runs/bert_trivia_qa_rc.json" \
  --set "model.model_name_or_path=bert-base-uncased"   \
  --set "dataset.test_dataset_name=rajpurkar/squad"    \
  --set "dataset.test_dataset_config_name=null"        \
  --set "dataset.test_version_2_with_negative=false"   \
  --set "dataset.test_split=validation"                \
  --set "run.learning_rate=5e-5"                       \
  --set "run.weight_decay=1e-8"                        \
  --set "run.epochs=2"                                 \
  --set "run.optim=adamw_torch_fused"                  \
  --set "run.max_grad_norm=0.5"

# 3. no clipping -> max_grad_norm=0 disables clipping
run_exp "exp03_bertbase_triviaqa_to_squad11_lr5e5_wd1e8_ep2_adamw_noclip" "configs/runs/bert_trivia_qa_rc.json" \
  --set "model.model_name_or_path=bert-base-uncased"   \
  --set "dataset.test_dataset_name=rajpurkar/squad"    \
  --set "dataset.test_dataset_config_name=null"        \
  --set "dataset.test_version_2_with_negative=false"   \
  --set "dataset.test_split=validation"                \
  --set "run.learning_rate=5e-5"                       \
  --set "run.weight_decay=1e-8"                        \
  --set "run.epochs=2"                                 \
  --set "run.optim=adamw_torch_fused"                  \
  --set "run.max_grad_norm=0"

# 4. "sgd with momentum 0.9" optimizer
run_exp "exp04_bertbase_triviaqa_to_squad11_lr5e5_wd1e8_ep2_defaultoptim_clip1p0" "configs/runs/bert_trivia_qa_rc.json" \
  --set "model.model_name_or_path=bert-base-uncased"   \
  --set "dataset.test_dataset_name=rajpurkar/squad"    \
  --set "dataset.test_dataset_config_name=null"        \
  --set "dataset.test_version_2_with_negative=false"   \
  --set "dataset.test_split=validation"                \
  --set "run.learning_rate=5e-5"                       \
  --set "run.weight_decay=1e-8"                        \
  --set "run.epochs=2"                                 \
  --set "run.optim=sgd"                                \
  --set "run.max_grad_norm=1.0"

# 5. 3 epochs
run_exp "exp05_bertbase_triviaqa_to_squad11_lr5e5_wd1e8_ep3_adamw_clip1p0" "configs/runs/bert_trivia_qa_rc.json" \
  --set "model.model_name_or_path=bert-base-uncased"   \
  --set "dataset.test_dataset_name=rajpurkar/squad"    \
  --set "dataset.test_dataset_config_name=null"        \
  --set "dataset.test_version_2_with_negative=false"   \
  --set "dataset.test_split=validation"                \
  --set "run.learning_rate=5e-5"                       \
  --set "run.weight_decay=1e-8"                        \
  --set "run.epochs=3"                                 \
  --set "run.optim=adamw_torch_fused"                  \
  --set "run.max_grad_norm=1.0"

# 6. wd 1e-7
run_exp "exp06_bertbase_triviaqa_to_squad11_lr5e5_wd1e7_ep2_adamw_clip1p0" "configs/runs/bert_trivia_qa_rc.json" \
  --set "model.model_name_or_path=bert-base-uncased"   \
  --set "dataset.test_dataset_name=rajpurkar/squad"    \
  --set "dataset.test_dataset_config_name=null"        \
  --set "dataset.test_version_2_with_negative=false"   \
  --set "dataset.test_split=validation"                \
  --set "run.learning_rate=5e-5"                       \
  --set "run.weight_decay=1e-7"                        \
  --set "run.epochs=2"                                 \
  --set "run.optim=adamw_torch_fused"                  \
  --set "run.max_grad_norm=1.0"

# 7. lr 5e-3, wd 1e-5
run_exp "exp07_bertbase_triviaqa_to_squad11_lr5e3_wd1e5_ep2_adamw_clip1p0" "configs/runs/bert_trivia_qa_rc.json" \
  --set "model.model_name_or_path=bert-base-uncased"   \
  --set "dataset.test_dataset_name=rajpurkar/squad"    \
  --set "dataset.test_dataset_config_name=null"        \
  --set "dataset.test_version_2_with_negative=false"   \
  --set "dataset.test_split=validation"                \
  --set "run.learning_rate=5e-3"                       \
  --set "run.weight_decay=1e-5"                        \
  --set "run.epochs=2"                                 \
  --set "run.optim=adamw_torch_fused"                  \
  --set "run.max_grad_norm=1.0"

# 8. BERT-base tuned on SQuAD 2.0, tested on TriviaQA
run_exp "exp08_bertbase_squad20_lr5e5_wd1e8_ep2_adamw_clip1p0" "configs/runs/bert_squad_v2.json" \
  --set "model.model_name_or_path=bert-base-uncased"      \
  --set "dataset.test_dataset_name=mandarjoshi/trivia_qa" \
  --set "dataset.test_dataset_config_name=rc"             \
  --set "dataset.test_version_2_with_negative=false"      \
  --set "dataset.test_split=validation"                   \
  --set "run.learning_rate=5e-5"                          \
  --set "run.weight_decay=1e-8"                           \
  --set "run.epochs=2"                                    \
  --set "run.optim=adamw_torch_fused"                     \
  --set "run.max_grad_norm=1.0"                           \
  --set "dataset.null_score_diff_threshold=-2.0"

# 9. DistilBERT, TriviaQA -> tested on SQuAD 1.1
run_exp "exp9_distilbert_triviaqa_to_squad11_lr5e5_wd1e8_ep2_adamw_clip1p0" "configs/runs/distilbert_trivia_qa_rc.json" \
  --set "model.model_name_or_path=distilbert-base-uncased" \
  --set "dataset.test_dataset_name=rajpurkar/squad"        \
  --set "dataset.test_dataset_config_name=null"            \
  --set "dataset.test_version_2_with_negative=false"       \
  --set "dataset.test_split=validation"                    \
  --set "run.learning_rate=5e-5"                           \
  --set "run.weight_decay=1e-8"                            \
  --set "run.epochs=2"                                     \
  --set "run.optim=adamw_torch_fused"                      \
  --set "run.max_grad_norm=1.0"                            

# 10. DistilBERT tuned on SQuAD 2.0, tested on TriviaQA
run_exp "exp10_distilbert_squad20_lr5e5_wd1e8_ep2_adamw_clip1p0" "configs/runs/distilbert_squad_v2.json" \
  --set "model.model_name_or_path=distilbert-base-uncased" \
  --set "dataset.test_dataset_name=mandarjoshi/trivia_qa"  \
  --set "dataset.test_dataset_config_name=rc"              \
  --set "dataset.test_version_2_with_negative=false"       \
  --set "dataset.test_split=validation"                    \
  --set "run.learning_rate=5e-5"                           \
  --set "run.weight_decay=1e-8"                            \
  --set "run.epochs=2"                                     \
  --set "run.optim=adamw_torch_fused"                      \
  --set "run.max_grad_norm=1.0"                            \
  --set "dataset.null_score_diff_threshold=-2.0"
