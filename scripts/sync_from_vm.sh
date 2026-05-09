#!/usr/bin/env bash
# Pull MOEB benchmark artifacts from the GPU node into the local repo.
# Run from anywhere; outputs land in `results/` and `logs/` next to this
# repo. After sync, regenerates the summary CSV/MD and the plot pack.
#
# Required env / args:
#   MOEB_KEY  — SSH private key (default: /tmp/moeb_key)
#   MOEB_HOST — user@host of the GPU node    (default: opc@<edit-me>)
#
# Usage:
#   MOEB_KEY=~/.ssh/my_key MOEB_HOST=ubuntu@1.2.3.4 ./sync_from_vm.sh

set -euo pipefail

KEY=${MOEB_KEY:-/tmp/moeb_key}
HOST=${MOEB_HOST:-opc@SET_ME}
LOCAL=$(cd "$(dirname "$0")/.." && pwd)

if [ ! -f "$KEY" ]; then
  echo "[sync] SSH key not found at $KEY" >&2
  echo "       export MOEB_KEY=/path/to/key and re-run." >&2
  exit 1
fi
if [ "$HOST" = "opc@SET_ME" ]; then
  echo "[sync] MOEB_HOST is unset. export MOEB_HOST=user@host" >&2
  exit 1
fi

RSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ServerAliveInterval=15"

echo "[sync] results"
rsync -a -e "$RSH" --exclude='.cache' "$HOST:moeb/results/" "$LOCAL/results/"

echo "[sync] driver scripts"
rsync -a -e "$RSH" --include='run_*.sh' --exclude='*' \
  "$HOST:moeb/" "$LOCAL/scripts/" 2>/dev/null || true
rsync -a -e "$RSH" "$HOST:moeb/tracks/"   "$LOCAL/scripts/tracks/"
rsync -a -e "$RSH" "$HOST:moeb/analysis/" "$LOCAL/scripts/analysis/"

echo "[sync] logs"
mkdir -p "$LOCAL/logs"
rsync -a -e "$RSH" --include='*.log' --include='*.txt' --exclude='*' \
  "$HOST:moeb/logs/" "$LOCAL/logs/"

echo "[regen] aggregate"
MOEB_RES="$LOCAL/results/a100_40gb_8x" \
  python3 "$LOCAL/scripts/analysis/aggregate_results.py" | tail -2

echo "[regen] plots"
python3 "$LOCAL/scripts/analysis/make_plots.py" 2>&1 | tail -12

echo "[done] $(date -u +%FT%TZ)"
