#!/usr/bin/env bash
# Start training in background with keepalive heartbeat

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/train_latest.log"
PID_FILE="${SCRIPT_DIR}/.train.pid"
WRAPPER_SCRIPT="${LOG_DIR}/train_wrapper_${TIMESTAMP}.sh"

if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}")"
    if kill -0 "${old_pid}" 2>/dev/null; then
        echo "training already running pid=${old_pid}"
        echo "check: bash status_train.sh"
        exit 1
    fi
    rm -f "${PID_FILE}"
fi

cat > "${WRAPPER_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${SCRIPT_DIR}"
bash keepalive.sh &
KEEPALIVE_PID=\$!
bash train_amd.sh
kill "\${KEEPALIVE_PID}" 2>/dev/null || true
rm -f "${SCRIPT_DIR}/.keepalive.pid"
EOF
chmod +x "${WRAPPER_SCRIPT}"

nohup bash "${WRAPPER_SCRIPT}" > "${LOG_FILE}" 2>&1 &
TRAIN_WRAPPER_PID=$!
echo "${TRAIN_WRAPPER_PID}" > "${PID_FILE}"
ln -sf "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

echo "training started in background"
echo "  wrapper pid=${TRAIN_WRAPPER_PID}"
echo "  log=${LOG_FILE}"
echo ""
echo "  bash status_train.sh"
echo "  tail -f ${LOG_FILE}"
echo "  bash stop_train.sh"
