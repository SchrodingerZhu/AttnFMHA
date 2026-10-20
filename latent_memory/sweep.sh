#!/usr/bin/env bash
# Run one train.py job per line of a sweep file, 4 at a time, one CPU thread each.
# Each line holds train.py arguments, e.g.:
#   --name slot_shift --set mem_type=slot addr=shift
# Usage: ./sweep.sh sweeps/stage1.txt results/stage1.jsonl [parallel]
set -euo pipefail
SWEEP=$1; RESULTS=$2; PAR=${3:-4}
mkdir -p "$(dirname "$RESULTS")" logs
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
grep -v '^\s*\(#\|$\)' "$SWEEP" | xargs -P "$PAR" -I{} sh -c \
  'name=$(echo "{}" | sed -n "s/.*--name \([^ ]*\).*/\1/p"); python3 train.py {} --results '"$RESULTS"' > logs/$name.log 2>&1 || echo "FAILED $name"'
