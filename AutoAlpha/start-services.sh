#!/usr/bin/env bash
# 启动 AutoAlpha (8788) / AutoCombine (8888) / QuantCombine (8889)
# 用法: ./start-services.sh [--no-resume]   # --no-resume 只起服务，不恢复任务
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RT="$ROOT/runtime-full-llm"
mkdir -p "$RT/logs" "$RT/pids"

# 只有显式传入任务 ID 才恢复，公开源码默认不得绑定任何本机研究状态。
AUTOALPHA_TASK_ID="${AUTOALPHA_RESUME_TASK_ID:-}"
AUTOCOMBINE_TASK_ID="${AUTOCOMBINE_RESUME_TASK_ID:-}"

RESUME=1
[[ "${1:-}" == "--no-resume" ]] && RESUME=0

port_alive() {
  "$ROOT/.venv/bin/python" - "$1" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.25)
    raise SystemExit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)
PY
}

port_pid() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -n1
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$port" 2>/dev/null | awk '{print $1}'
  fi
}

start_one() {
  local name="$1" port="$2" bin="$3"; shift 3
  local label="com.autoalpha.local.$name.$port"
  local log="$RT/logs/$name-$port.log"
  local executable="$ROOT/.venv/bin/$bin"
  if [[ ! -x "$executable" ]]; then
    echo "[error] 缺少 $executable；请先运行 uv sync --frozen --all-groups"
    return 1
  fi
  if port_alive "$port"; then
    echo "[skip] $name 已在 $port 端口运行"
    return
  fi
  if [[ "$(uname -s)" == "Darwin" ]] && command -v launchctl >/dev/null 2>&1; then
    launchctl remove "$label" >/dev/null 2>&1 || true
    launchctl submit -l "$label" -o "$log" -e "$log" -- \
      /usr/bin/env AUTOALPHA_RUNTIME="$RT" "$@" "$executable"
    echo "[start] $name (launchd: $label) -> http://127.0.0.1:$port"
  else
    nohup env AUTOALPHA_RUNTIME="$RT" "$@" "$executable" \
      >> "$log" 2>&1 </dev/null &
    disown "$!" 2>/dev/null || true
    echo $! > "$RT/pids/$name-$port.pid"
    echo "[start] $name (pid $!) -> http://127.0.0.1:$port"
  fi
}

start_one autoalpha    8788 autoalpha-service    AUTOALPHA_PORT=8788
start_one autocombine  8888 autocombine-service  AUTOCOMBINE_PORT=8888
start_one quantcombine 8889 quantcombine-service QUANTCOMBINE_PORT=8889

# 等待端口就绪
for port in 8788 8888 8889; do
  for _ in {1..30}; do
    port_alive "$port" && break
    sleep 0.5
  done
  port_alive "$port" || { echo "[error] 端口 $port 未就绪，查看 $RT/logs"; exit 1; }
  pid="$(port_pid "$port")"
  if [[ -n "$pid" ]]; then
    echo "$pid" \
      > "$RT/pids/$(case "$port" in 8788) echo autoalpha;; 8888) echo autocombine;; *) echo quantcombine;; esac)-$port.pid"
  fi
done
echo "[ok] 三个服务端口均已就绪"

if (( RESUME )); then
  if [[ -n "$AUTOALPHA_TASK_ID" ]]; then
    echo "[resume] AutoAlpha 任务 $AUTOALPHA_TASK_ID"
    curl -sf -X POST "http://127.0.0.1:8788/api/research-tasks/$AUTOALPHA_TASK_ID/start" >/dev/null \
      && echo "  -> RUNNING" || echo "  -> 恢复失败（可能已在运行或被协议阻塞）"
  fi
  if [[ -n "$AUTOCOMBINE_TASK_ID" ]]; then
    echo "[resume] AutoCombine 任务 $AUTOCOMBINE_TASK_ID"
    curl -sf -X POST "http://127.0.0.1:8888/api/tasks/$AUTOCOMBINE_TASK_ID/start" >/dev/null \
      && echo "  -> RUNNING/SEARCHING" || echo "  -> 恢复失败（可能已在运行）"
  fi
fi

echo "AutoAlpha:    http://127.0.0.1:8788"
echo "AutoCombine:  http://127.0.0.1:8888"
echo "QuantCombine: http://127.0.0.1:8889"
