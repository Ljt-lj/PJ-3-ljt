#!/usr/bin/env bash
# Stop background training and keepalive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

stop_pid_file() {
    local pid_file="$1"
    local label="$2"
    if [[ ! -f "${pid_file}" ]]; then
        return 0
    fi
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
        echo "stopped ${label} pid=${pid}"
    fi
    rm -f "${pid_file}"
}

stop_pid_file "${SCRIPT_DIR}/.train.pid" "training"
stop_pid_file "${SCRIPT_DIR}/.keepalive.pid" "keepalive"
rm -f "${SCRIPT_DIR}/.train_heartbeat"

echo "done"
