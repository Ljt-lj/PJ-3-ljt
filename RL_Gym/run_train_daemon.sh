#!/usr/bin/env bash
# 后台启动训练 + 心跳保活，断开终端后仍继续运行
# 用法: bash run_train_daemon.sh
# 查看: bash status_train.sh
# 停止: bash stop_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/train_latest.log"

if [[ -f "${SCRIPT_DIR}/.train.pid" ]] && kill -0 "$(cat "${SCRIPT_DIR}/.train.pid")" 2>/dev/null; then
    echo "已有训练任务在运行 (PID $(cat "${SCRIPT_DIR}/.train.pid"))"
    echo "查看进度: bash status_train.sh"
    exit 1
fi

nohup bash -c "
    set -euo pipefail
    cd '${SCRIPT_DIR}'
    bash keepalive.sh &
    KEEPALIVE_PID=\$!
    echo \"keepalive PID: \$KEEPALIVE_PID\"
    bash train_amd.sh
    kill \"\$KEEPALIVE_PID\" 2>/dev/null || true
    rm -f '${SCRIPT_DIR}/.keepalive.pid'
" > "${LOG_FILE}" 2>&1 &

TRAIN_WRAPPER_PID=$!
echo "${TRAIN_WRAPPER_PID}" > "${SCRIPT_DIR}/.train.pid"
ln -sf "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

echo "训练已在后台启动"
echo "  包装进程 PID: ${TRAIN_WRAPPER_PID}"
echo "  日志文件: ${LOG_FILE}"
echo ""
echo "常用命令:"
echo "  bash status_train.sh    # 查看进度"
echo "  tail -f ${LOG_FILE}     # 实时日志"
echo "  bash stop_train.sh      # 停止训练"
