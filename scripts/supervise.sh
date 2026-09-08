#!/usr/bin/env bash
# Runs the remaining jobs unattended, all at the full 106 subjects.
#
# No time limits and no downgrading -- the point is to get the real numbers.
# What this does instead is survive the things that actually go wrong:
#   - a job crashing (retry it, up to 3 times)
#   - memory pressure (wait for headroom before starting the next job)
#   - one job blocking another (run them in sequence, not in parallel)
# Everything is timestamped into results/supervisor.log.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/supervisor.log
say() { echo "[$(date '+%a %H:%M')] $*" | tee -a "$LOG"; }

# Block until the machine has room. The windowed array alone is ~4 GB, and at
# 71 MB free the OS starts killing things.
wait_for_memory() {
  local need_mb=${1:-1500} waited=0
  while :; do
    local free=$(vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} END{gsub(/\./,"",f); gsub(/\./,"",i); print int((f+i)*4096/1048576)}')
    [ "${free:-0}" -ge "$need_mb" ] && break
    [ $waited -eq 0 ] && say "waiting for memory (need ${need_mb}MB, have ${free}MB)"
    sleep 60; waited=$((waited+60))
    [ $waited -ge 1800 ] && { say "proceeding anyway after 30 min"; break; }
  done
}

# run <name> <logfile> <command...>   -- retries up to 3 times on failure
run() {
  local name=$1 log=$2; shift 2
  local attempt=1
  while [ $attempt -le 3 ]; do
    wait_for_memory 1500
    say "START $name (attempt $attempt, 106 subjects)"
    local t0=$(date +%s)
    if "$@" > "$log" 2>&1; then
      say "OK $name after $((($(date +%s)-t0)/60)) min"
      return 0
    fi
    say "FAIL $name attempt $attempt (rc=$?) -- see $log"
    tail -3 "$log" | sed 's/^/    /' | tee -a "$LOG"
    attempt=$((attempt+1))
    sleep 30
  done
  say "GAVE UP on $name after 3 attempts"
  return 1
}

run R2    results/log_R2.txt    $PY -u scripts/leakage.py --subjects 109
run R6    results/log_R6.txt    $PY -u scripts/transfer.py --subjects 109 --pipeline csp_lda
run SPLIT results/log_split.txt $PY -u scripts/split_check.py --subjects 109
run R35   results/log_R35.txt   $PY -u scripts/baseline_and_fit.py --subjects 109 --n-perm 200

say "regenerating figures"
$PY scripts/figures.py >> "$LOG" 2>&1
say "ALL DONE -- every result at 106 subjects"
