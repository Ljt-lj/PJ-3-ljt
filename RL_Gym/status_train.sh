#!/usr/bin/env bash
# 查看后台训练状态与最近日志

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PID_FILE="${SCRIPT_DIR}/.train.pid"
HEARTBEAT="${SCRIPT_DIR}/.train_heartbeat"
LOG_FILE="${SCRIPT_DIR}/logs/train_latest.log"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "状态: 运行中 (PID $(cat "${PID_FILE}))"
else
    echo "状态: 未运行"
    rm -f "${PID_FILE}"
fi

if [[ -f "${HEARTBEAT}" ]]; then
    echo "最近心跳: $(cat "${HEARTBEAT}")"
fi

if [[ -L "${LOG_FILE}" ]] || [[ -f "${LOG_FILE}" ]]; then
    echo ""
    echo "最近日志 (最后 15 行):"
    tail -n 15 "${LOG_FILE}"
else
    echo "暂无日志文件"
fi
