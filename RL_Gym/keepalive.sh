#!/usr/bin/env bash
# 定时心跳（默认每 30 分钟），降低 DSW 实例因空闲被判定休眠的概率
# 与训练脚本配合使用，由 run_train_daemon.sh 启动

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${KEEPALIVE_INTERVAL:-1800}"
HEARTBEAT_FILE="${SCRIPT_DIR}/.train_heartbeat"
PID_FILE="${SCRIPT_DIR}/.keepalive.pid"

echo $$ > "${PID_FILE}"
echo "keepalive 已启动 (PID $$, 间隔 ${INTERVAL}s, 心跳文件 ${HEARTBEAT_FILE})"

while true; do
    date '+%Y-%m-%d %H:%M:%S' > "${HEARTBEAT_FILE}"
    sleep "${INTERVAL}"
done
