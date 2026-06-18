#!/usr/bin/env bash
# 根据已训练权重生成提交视频与实验报告数据

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-submission}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
SEED_PROBES="${SEED_PROBES:-20}"

ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --eval-episodes "${EVAL_EPISODES}"
    --seed-probes "${SEED_PROBES}"
)

if [[ -n "${MODEL_PATH}" ]]; then
    ARGS+=(--model-path "${MODEL_PATH}")
fi

python eval_submit.py "${ARGS[@]}"

echo ""
echo "提交物目录: ${SCRIPT_DIR}/${OUTPUT_DIR}/"
echo "  pacman_submit.mp4  - 提交视频"
echo "  report.md          - 实验报告草稿"
echo "  report.json        - 结构化数据"
