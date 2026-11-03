#!/usr/bin/env bash
# One worker per GPU. Pulls the next job from a shared queue under flock and
# runs it, so an uneven mix of molecule sizes self-balances across the box.
#
# Usage: landscape_sweep_worker.sh <gpu> <root> <python> [key=value ...]
#
# <root>/queue.txt holds one "model molecule start" job per line. Each finished
# job writes its output under <root>/out and appends one line to
# <root>/progress.log. Extra key=value pairs go to every job, after the
# protocol's own: neighbours=6, the setting every published run used.
set -u
gpu=$1
root=$2
python=$3
shift 3
extra=("$@")
script=$(dirname "$(readlink -f "$0")")/landscape_sweep.py
queue=$root/queue.txt
lock=$root/queue.lock
mkdir -p "$root/out"
while true; do
  job=$(flock "$lock" -c "head -1 '$queue'; sed -i '1d' '$queue'")
  [ -z "$job" ] && break
  read -r model molecule start <<< "$job"
  label="${model}__${molecule}__start${start}"
  start_s=$(date -u +%s)
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$script" \
      "$molecule" "$start" "$root/out" "$label" model="$model" neighbours=6 \
      "${extra[@]}" > "$root/out/$label.log" 2>&1
  rc=$?
  echo "[$(date -u +%H:%M:%S)] $label rc=$rc gpu=$gpu secs=$(( $(date -u +%s) - start_s ))" \
      >> "$root/progress.log"
done
echo "[$(date -u +%H:%M:%S)] worker gpu=$gpu drained" >> "$root/progress.log"
