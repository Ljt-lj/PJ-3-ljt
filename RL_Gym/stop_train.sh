#!/usr/bin/env bash
# 停止后台训练与心跳进程

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

stop_pid() {
    local pid_file="$1"
    local label="$2"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid="$(cat "${pid_file}")"
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            echo "已停止 ${label} (PID ${pid})"
        fi
        rm -f "${pid_file}"
    fi
}

stop_pid "${SCRIPT_DIR}/.train.pid" "训练"
stop_pid "${SCRIPT_DIR}/.keepalive.pid" "keepalive"
rm -f "${SCRIPT_DIR}/.train_heartbeat"

echo "完成"
