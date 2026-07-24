#!/usr/bin/env zsh
# 停止 AutoAlpha (8788) / AutoCombine (8888) / QuantCombine (8889)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RT="$ROOT/runtime-full-llm"

stop_one() {
  local name="$1" port="$2"
  local pidfile="$RT/pids/$name-$port.pid" pid=""

  [[ -f "$pidfile" ]] && pid="$(cat "$pidfile")"
  # PID 文件失效时按端口兜底查找
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    pid="$(lsof -nP -tiTCP:$port -sTCP:LISTEN 2>/dev/null | head -n1)"
  fi

  if [[ -z "$pid" ]]; then
    echo "[skip] $name 未在运行"
    rm -f "$pidfile"
    return
  fi

  kill "$pid" 2>/dev/null
  for _ in {1..20}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[force] $name (pid $pid) 未响应 SIGTERM，发送 SIGKILL"
    kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$pidfile"
  echo "[stop] $name (pid $pid) 已停止"
}

stop_one autoalpha    8788
stop_one autocombine  8888
stop_one quantcombine 8889

echo "[ok] 完成（运行中的任务已由服务落盘为检查点，下次启动后可恢复）"
