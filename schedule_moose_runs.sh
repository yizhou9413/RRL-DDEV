#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
LAUNCH_GAP_SECONDS="${LAUNCH_GAP_SECONDS:-2}"
LOCK_FILE="${LOCK_FILE:-/tmp/schedule_moose_runs.lock}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    log "another scheduler instance is already running: $LOCK_FILE"
    exit 1
  fi
fi

launch_cmd() {
  local cmd="$1"
  nohup bash -lc "$cmd" >/dev/null 2>&1 &
  local pid=$!
  log "launch pid=${pid} -> $cmd"
}

schedule_group() {
  local delay_seconds="$1"
  shift
  local group_name="$1"
  shift

  (
    local total="$#"
    local index=0

    log "group ${group_name} armed, will start in ${delay_seconds}s"
    sleep "$delay_seconds"
    log "group ${group_name} starting"

    for cmd in "$@"; do
      index=$((index + 1))
      launch_cmd "$cmd"
      if [[ "$index" -lt "$total" && "$LAUNCH_GAP_SECONDS" -gt 0 ]]; then
        sleep "$LAUNCH_GAP_SECONDS"
      fi
    done

    log "group ${group_name} launched ${total} command(s)"
  ) &
}

run5_cmd() {
  local device="$1"
  local num_parallel="$2"
  local seed="$3"
  local disturbance_every="$4"
  printf '%s run5.py train moose --car-preset tesla_model_3 --device %s --num-parallel %s --seed %s --distured --disturbance-every %s --quiet >/dev/null 2>&1' \
    "$PYTHON_BIN" "$device" "$num_parallel" "$seed" "$disturbance_every"
}

run_rarl_cmd() {
  local device="$1"
  local seed="$2"
  printf '%s run_rarl.py train moose --car-preset tesla_model_3 --device %s --num-parallel 30000 --seed %s >/dev/null 2>&1' \
    "$PYTHON_BIN" "$device" "$seed"
}

declare -a commands_1m=()
declare -a commands_8h=()
declare -a commands_14h=()

for seed in 1 2 3; do
  commands_1m+=("$(run5_cmd cuda:0 30 "$seed" 1)")
done

for seed in 1 2 3; do
  commands_8h+=("$(run5_cmd cuda:0 30000 "$seed" 1)")
done

for seed in 1 2 3; do
  commands_8h+=("$(run5_cmd cuda:1 30000 "$seed" 2)")
done

for seed in 4 5 6; do
  commands_8h+=("$(run5_cmd cuda:2 30000 "$seed" 1)")
done

for seed in 4 5 6; do
  commands_8h+=("$(run5_cmd cuda:3 30000 "$seed" 2)")
done

commands_14h+=("$(run_rarl_cmd cuda:0 1)")
commands_14h+=("$(run_rarl_cmd cuda:1 2)")
commands_14h+=("$(run_rarl_cmd cuda:3 3)")
commands_14h+=("$(run_rarl_cmd cuda:1 4)")

schedule_group 60 "1m-run5" "${commands_1m[@]}"
schedule_group $((8 * 3600)) "8h-run5" "${commands_8h[@]}"
schedule_group $((14 * 3600)) "14h-rarl" "${commands_14h[@]}"

log "scheduler is active"
log "launch gap between commands in the same group: ${LAUNCH_GAP_SECONDS}s"
log "recommended start command: nohup bash schedule_moose_runs.sh >/tmp/schedule_moose_runs.log 2>&1 &"

wait
