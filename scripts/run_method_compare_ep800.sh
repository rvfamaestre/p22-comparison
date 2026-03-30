#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-mixed_traffic}"
MPL_DIR="${MPLCONFIGDIR:-/tmp/mpl}"
CONFIG_DIR="config/rerun_20260323_ep800"
BASE_OUTPUT="output/method_compare_random_rerun_20260323_ep800"
LOG_DIR="$BASE_OUTPUT/logs"
STATUS_FILE="$BASE_OUTPUT/status.txt"

mkdir -p "$MPL_DIR" "$LOG_DIR" "$BASE_OUTPUT" "$BASE_OUTPUT/no_rl" "$BASE_OUTPUT/ppo" "$BASE_OUTPUT/sac"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log_status() {
  printf '[%s] %s\n' "$(timestamp)" "$1" | tee -a "$STATUS_FILE"
}

run_training_job() {
  local method="$1"
  local config_path="$2"
  local log_path="$LOG_DIR/${method}_training.log"
  log_status "Starting ${method} training with ${config_path}"
  env MPLCONFIGDIR="$MPL_DIR" /usr/bin/time -p \
    -o "$BASE_OUTPUT/${method}/training_duration.txt" \
    conda run -n "$ENV_NAME" python run_training.py --config "$config_path" \
    > "$log_path" 2>&1
  log_status "Finished ${method} training"
}

run_eval_job() {
  local method="$1"
  local tag="$2"
  local config_path="$3"
  local log_path="$LOG_DIR/${method}_${tag}_simulation.log"
  log_status "Starting ${method} ${tag} simulation with ${config_path}"
  env MPLCONFIGDIR="$MPL_DIR" conda run -n "$ENV_NAME" \
    python run_simulation.py --config "$config_path" \
    > "$log_path" 2>&1
  log_status "Finished ${method} ${tag} simulation"
}

build_report() {
  local log_path="$LOG_DIR/report_build.log"
  log_status "Building comparison report"
  env MPLCONFIGDIR="$MPL_DIR" conda run -n "$ENV_NAME" python \
    scripts/build_rl_comparison_report.py \
    --base-output "$BASE_OUTPUT" \
    --methods no_rl ppo sac \
    --training-methods ppo sac \
    > "$log_path" 2>&1
  log_status "Finished comparison report"
}

log_status "800-episode comparison pipeline started"

run_training_job "ppo" "$CONFIG_DIR/train_ppo_h50_random.yaml" &
PPO_PID=$!
run_training_job "sac" "$CONFIG_DIR/train_sac_h50_random.yaml" &
SAC_PID=$!

wait "$PPO_PID"
wait "$SAC_PID"

run_eval_job "no_rl" "h25" "$CONFIG_DIR/eval_no_rl_h25_random.yaml" &
PID_1=$!
run_eval_job "no_rl" "h50" "$CONFIG_DIR/eval_no_rl_h50_random.yaml" &
PID_2=$!
run_eval_job "no_rl" "h75" "$CONFIG_DIR/eval_no_rl_h75_random.yaml" &
PID_3=$!
run_eval_job "ppo" "h25" "$CONFIG_DIR/eval_ppo_h25_random.yaml" &
PID_4=$!
run_eval_job "ppo" "h50" "$CONFIG_DIR/eval_ppo_h50_random.yaml" &
PID_5=$!
run_eval_job "ppo" "h75" "$CONFIG_DIR/eval_ppo_h75_random.yaml" &
PID_6=$!
run_eval_job "sac" "h25" "$CONFIG_DIR/eval_sac_h25_random.yaml" &
PID_7=$!
run_eval_job "sac" "h50" "$CONFIG_DIR/eval_sac_h50_random.yaml" &
PID_8=$!
run_eval_job "sac" "h75" "$CONFIG_DIR/eval_sac_h75_random.yaml" &
PID_9=$!

wait "$PID_1" "$PID_2" "$PID_3" "$PID_4" "$PID_5" "$PID_6" "$PID_7" "$PID_8" "$PID_9"

build_report

log_status "800-episode comparison pipeline completed successfully"
