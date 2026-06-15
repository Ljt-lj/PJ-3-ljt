#!/usr/bin/env bash
# RL实践 - MsPacman DQN 训练（AMD GPU / Linux 云端）
# ROCm 版 PyTorch 同样通过 torch.cuda 接口使用 GPU

set -euo pipefail

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
