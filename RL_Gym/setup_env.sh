#!/usr/bin/env bash
# RL实践 - 环境安装脚本（Linux 云端 / AMD GPU）
# 推荐 Python 3.8~3.10（requirements.txt 要求 < 3.11）

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> 创建虚拟环境 (Python ${PYTHON_VERSION})"
if command -v conda &>/dev/null; then
    conda create -n rl_gym python="${PYTHON_VERSION}" -y
    eval "$(conda shell.bash hook)"
    conda activate rl_gym
else
    python"${PYTHON_VERSION}" -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

echo "==> 安装基础依赖"
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1" "ale-py==0.8.1"

# AMD GPU (ROCm): 若云端已预装 ROCm PyTorch 可跳过；否则按 ROCm 版本安装，例如：
# pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
if [[ "${USE_ROCM:-0}" == "1" ]]; then
    ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.0}"
    echo "==> 安装 ROCm 版 PyTorch: ${ROCM_INDEX}"
    pip install torch --index-url "${ROCM_INDEX}"
fi

echo "==> 检测 GPU"
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
PY

echo "==> 验证 QNetwork 前向传播"
python - <<'PY'
import gymnasium as gym
import torch
from dqn_atari import QNetwork

class FakeEnv:
    single_action_space = gym.spaces.Discrete(9)

env = FakeEnv()
net = QNetwork(env)
x = torch.zeros(1, 4, 84, 84)
out = net(x)
assert out.shape == (1, 9), f"unexpected shape: {out.shape}"
print("QNetwork OK, output shape:", tuple(out.shape))
PY

echo "==> 环境就绪。运行训练: bash train_amd.sh"
