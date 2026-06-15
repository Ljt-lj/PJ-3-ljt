#!/usr/bin/env bash
# RL实践 - MsPacman DQN 训练（AMD GPU / Linux 云端）
# ROCm 版 PyTorch 同样通过 torch.cuda 接口使用 GPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 自动激活 setup_env.sh 创建的虚拟环境
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/bin/activate"
elif [[ -z "${CONDA_DEFAULT_ENV:-}" ]] && command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate rl_gym 2>/dev/null || true
fi

if ! python -c "import gymnasium" &>/dev/null; then
    echo "错误: 未安装 gymnasium。请先运行: bash setup_env.sh"
    exit 1
fi

EXP_NAME="${EXP_NAME:-MsPacman-v5}"
ENV_ID="${ENV_ID:-ALE/MsPacman-v5}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-5000000}"
BUFFER_SIZE="${BUFFER_SIZE:-400000}"
SEED="${SEED:-1}"

python dqn_atari.py \
    --exp-name "${EXP_NAME}" \
    --capture-video True \
    --save-model True \
    --env-id "${ENV_ID}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --buffer-size "${BUFFER_SIZE}" \
    --seed "${SEED}" \
    --cuda True
