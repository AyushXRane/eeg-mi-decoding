#!/usr/bin/env bash
# Prefetch EDFs straight into MNE's cache dir. eegbci.load_data() is serial and
# PhysioNet gives ~150 KB/s per connection, so 190 files takes ~50 min that way.
# Parallel curl into the same layout cuts it to ~6 and load_data() then hits cache.
#   ./scripts/fetch.sh 32 "3 4 7 8 11 12" 8
set -u
DEST="${MNE_DATA:-$HOME/mne_data}/MNE-eegbci-data/files/eegmmidb/1.0.0"
BASE="https://physionet.org/files/eegmmidb/1.0.0"
N="${1:-32}"
RUNS="${2:-3 4 7 8 11 12}"
JOBS="${3:-8}"

for i in $(seq 1 "$N"); do
  case "$i" in 88|89|92|100) continue ;; esac   # nonstandard sfreq / trial dur
  s=$(printf 'S%03d' "$i")
  for r in $RUNS; do printf '%s %s\n' "$s" "$(printf 'R%02d' "$r")"; done
done | xargs -P "$JOBS" -n 2 bash -c '
  d="'"$DEST"'/$0"; f="$d/$0$1.edf"
  [ -s "$f" ] && exit 0
  mkdir -p "$d"
  # -f so an HTTP error is a failure, not an error page saved as an .edf
  curl -fsS --retry 3 --max-time 180 -o "$f" "'"$BASE"'/$0/$0$1.edf" \
    || { rm -f "$f"; echo "FAILED $0$1" >&2; }
'
echo "cached: $(find "$DEST" -name '*.edf' -size +100k | wc -l | tr -d ' ') files"
