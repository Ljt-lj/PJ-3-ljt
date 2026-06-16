#!/usr/bin/env bash
# View background training status and recent logs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PID_FILE="${SCRIPT_DIR}/.train.pid"
HEARTBEAT="${SCRIPT_DIR}/.train_heartbeat"
LOG_FILE="${SCRIPT_DIR}/logs/train_latest.log"

if [[ -f "${PID_FILE}" ]]; then
    train_pid="$(cat "${PID_FILE}")"
    if kill -0 "${train_pid}" 2>/dev/null; then
        echo "status: running pid=${train_pid}"
    else
        echo "status: not running"
        rm -f "${PID_FILE}"
    fi
else
    echo "status: not running"
fi

if [[ -f "${HEARTBEAT}" ]]; then
    echo "heartbeat: $(cat "${HEARTBEAT}")"
fi

if [[ -L "${LOG_FILE}" ]] || [[ -f "${LOG_FILE}" ]]; then
    echo ""
    echo "last 15 log lines:"
    tail -n 15 "${LOG_FILE}"
else
    echo "no log file"
fi
