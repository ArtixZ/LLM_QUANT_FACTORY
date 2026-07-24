#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CHECKPOINT="${PROJECT_ROOT}/output/reevaluation/canonical_library_long_only_2015_2024_v1/checkpoint-505.json"

checkpoint="${DEFAULT_CHECKPOINT}"
watch_seconds=""

usage() {
  cat <<'EOF'
Usage:
  scripts/check_factor_reevaluation.sh [--checkpoint PATH]
  scripts/check_factor_reevaluation.sh --watch [SECONDS] [--checkpoint PATH]

Options:
  --watch [SECONDS]  Refresh continuously; default interval is 5 seconds.
  --checkpoint PATH  Read a different batch checkpoint.
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      [[ $# -ge 2 ]] || { echo "--checkpoint requires a path" >&2; exit 2; }
      checkpoint="$2"
      shift 2
      ;;
    --watch)
      watch_seconds="5"
      if [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]]; then
        watch_seconds="$2"
        shift
      fi
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 1
fi

python_bin="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3)"
fi

render() {
  CHECKPOINT="${checkpoint}" "${python_bin}" - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

checkpoint = Path(os.environ["CHECKPOINT"]).expanduser().resolve()
payload = json.loads(checkpoint.read_text(encoding="utf-8"))
metadata = payload.get("metadata", {})
results = payload.get("results", {})
factor_ids = [str(value) for value in metadata.get("factor_ids", [])]
total = len(factor_ids) or int(metadata.get("candidate_family_size", 0))
successful = sum(item.get("metrics") is not None for item in results.values())
failed = sum(
    item.get("metrics") is None and bool(item.get("error")) for item in results.values()
)
completed = successful + failed
remaining = max(total - successful, 0)

ps = subprocess.run(
    ["ps", "-Ao", "pid=,ppid=,%cpu=,command="],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
matching = []
for line in ps:
    if "batch_reevaluate_factor_library.py" not in line:
        continue
    if checkpoint.name not in line and str(checkpoint) not in line:
        continue
    parts = line.strip().split(None, 3)
    if len(parts) == 4:
        matching.append((int(parts[0]), int(parts[1]), float(parts[2]), parts[3]))

pids = {row[0] for row in matching}
parents = [row for row in matching if row[1] not in pids]
workers = max(len(matching) - len(parents), 0)
cpu = sum(row[2] for row in matching)
if successful >= total > 0:
    status = "COMPLETED"
elif matching:
    status = "RUNNING"
elif failed:
    status = "STOPPED_WITH_ERRORS"
else:
    status = "PAUSED"

elapsed_values = [
    float(item["elapsed_seconds"])
    for item in results.values()
    if item.get("metrics") is not None and item.get("elapsed_seconds") is not None
]
average_seconds = sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0
effective_workers = workers or 1
eta_seconds = remaining * average_seconds / effective_workers if matching else 0.0

def duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"

ratio = successful / total if total else 0.0
width = 36
filled = min(width, int(ratio * width))
bar = "#" * filled + "-" * (width - filled)
mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime).astimezone()
print(f"Factor library reevaluation  {status}")
print(f"Protocol : {metadata.get('canonical_protocol') or metadata.get('protocol', '--')}")
print(f"Window   : {metadata.get('public_validation_start', '--')} -> {metadata.get('public_validation_end', '--')}")
print(f"Recent   : {metadata.get('recent_evaluation_start', '--')} -> {metadata.get('recent_evaluation_end', '--')}")
print(f"Progress : [{bar}] {successful}/{total} ({ratio * 100:6.2f}%)")
print(f"Results  : success={successful}  failed={failed}  remaining={remaining}")
print(f"Runtime  : processes={len(matching)}  workers={workers}  aggregate_cpu={cpu:.1f}%")
print(f"Timing   : mean_factor={duration(average_seconds)}  estimated_eta={duration(eta_seconds) if matching else '--'}")
print(f"Updated  : {mtime:%Y-%m-%d %H:%M:%S %Z}")
print(f"File     : {checkpoint}")

if failed:
    print("\nErrors:")
    shown = 0
    for factor_id, item in results.items():
        if item.get("metrics") is not None or not item.get("error"):
            continue
        print(f"  {factor_id}: {str(item['error']).splitlines()[0]}")
        shown += 1
        if shown == 5:
            remaining_errors = failed - shown
            if remaining_errors > 0:
                print(f"  ... and {remaining_errors} more")
            break
PY
}

if [[ -n "${watch_seconds}" ]]; then
  while true; do
    printf '\033[2J\033[H'
    render
    sleep "${watch_seconds}"
  done
else
  render
fi
