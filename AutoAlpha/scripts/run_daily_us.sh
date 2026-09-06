#!/usr/bin/env bash
# Daily wrapper for the LLM_QUANT_FACTORY US equity paper pipeline.
# Invoked by launchd (com.quantfactory.daily) Mon-Fri at 13:35 PT (16:35 ET),
# 35 minutes after the close so IBKR has settled the session's daily bars.
#
# Pipeline:
#   1. Source any files listed in QUANTFACTORY_ENV_FILES (colon-separated) so
#      notification credentials are present under launchd's empty environment.
#   2. Wait for the IB Gateway API port (a separate job owns starting it).
#   3. Run daily_run.py, which syncs data, rebuilds the panel, forms the target
#      book, previews orders, and pushes the digest through notify.sh.
#
# daily_run.py reports its own success and failure. This wrapper only covers
# the cases it cannot: the gateway never coming up, or Python never starting.
#
# Scheduled runs are deliberately preview-only. Broker submission remains a
# foreground operation through daily_run.py --submit --confirm --managed-account.

set -uo pipefail   # NOT -e: the EXIT trap must fire on every path.

# Resolve the project from this script's own location so a clone works anywhere.
PROJECT_ROOT="${QUANTFACTORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="$PROJECT_ROOT/logs/scheduler"
IBKR_HOST="${IBKR_HOST:-127.0.0.1}"
IBKR_PORT="${IBKR_PORT:-4002}"
GATEWAY_WAIT_SECS="${GATEWAY_WAIT_SECS:-180}"

mkdir -p "$LOG_DIR"
DATE_TAG="$(date +%Y-%m-%d)"
RUN_LOG="$LOG_DIR/run-${DATE_TAG}.log"
cd "$PROJECT_ROOT" || exit 3

log() { printf '[run_daily_us] %s\n' "$*" >> "$RUN_LOG"; }
# NOTIFY is resolved after the env files below, so .env.local can set it.
notify() {
  [[ -n "$NOTIFY" && -x "$NOTIFY" ]] && "$NOTIFY" "$1" "$2" "${3:-info}" "quantfactory" || true
}

source_env() {
  [[ -n "${1:-}" && -f "$1" ]] || return 0
  set -a
  # shellcheck disable=SC1090
  source "$1"
  set +a
}

# $PROJECT_ROOT/.env.local holds this host's paths. It is gitignored, so the
# repo stays portable while the job stays configured; under launchd's empty
# environment it is the only thing that makes notification work. It is sourced
# first because it is what sets QUANTFACTORY_ENV_FILES, the colon-separated
# list of credential files loaded after it.
source_env "$PROJECT_ROOT/.env.local"
IFS=':' read -r -a ENV_FILES <<< "${QUANTFACTORY_ENV_FILES:-}"
for env in "${ENV_FILES[@]:-}"; do
  source_env "$env"
done

# Resolved here, not above: .env.local is what supplies these under launchd.
PYTHON="${QUANTFACTORY_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
NOTIFY="${QUANTFACTORY_NOTIFY_SCRIPT:-}"

{
  echo "============================================================"
  echo "[run_daily_us] start  $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[run_daily_us] python $PYTHON"
  echo "[run_daily_us] gw     $IBKR_HOST:$IBKR_PORT"
  echo "============================================================"
} >> "$RUN_LOG"

STAGE="init"
RUN_EXIT_CODE=255
on_exit() {
  case "$STAGE" in
    gateway_wait)
      notify "Daily run could not start" \
        "IB Gateway never came up on $IBKR_HOST:$IBKR_PORT within ${GATEWAY_WAIT_SECS}s. Log: $RUN_LOG" \
        "error"
      ;;
    ran)
      # daily_run.py sends its own digest on success and its own error on a
      # handled failure. Only an unhandled crash needs a wrapper-level alert.
      if [[ "$RUN_EXIT_CODE" != "0" && "$RUN_EXIT_CODE" != "1" ]]; then
        notify "Daily run crashed rc=$RUN_EXIT_CODE" \
          "Unhandled failure. Last 5 lines: $(tail -5 "$RUN_LOG" | tr '\n' ' ')" \
          "error"
      fi
      ;;
    *)
      notify "Daily run aborted ($STAGE)" "Failed before Python started. Log: $RUN_LOG" "error"
      ;;
  esac
}
trap on_exit EXIT

STAGE="gateway_wait"
log "Waiting for IB Gateway on $IBKR_HOST:$IBKR_PORT (up to ${GATEWAY_WAIT_SECS}s)..."
gateway_up=0
for i in $(seq 1 $(( GATEWAY_WAIT_SECS / 3 ))); do
  if nc -z -w 1 "$IBKR_HOST" "$IBKR_PORT" 2>/dev/null; then
    log "Gateway port OK after $((i * 3))s"
    gateway_up=1
    break
  fi
  sleep 3
done
if [[ "$gateway_up" == "0" ]]; then
  log "FATAL: gateway never became reachable"
  exit 2
fi

STAGE="ran"
log "preview only; scheduled broker submission is disabled"
log "Launching daily_run.py..."
"$PYTHON" -u scripts/daily_run.py >> "$RUN_LOG" 2>&1
RUN_EXIT_CODE=$?
log "daily_run.py exited rc=$RUN_EXIT_CODE"
exit "$RUN_EXIT_CODE"
