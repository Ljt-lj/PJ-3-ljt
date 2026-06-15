#!/usr/bin/env bash
# RL实践 - 环境安装脚本（Linux 云端 / AMD GPU）
# 推荐 Python 3.10~3.12

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-.venv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "==> 创建虚拟环境 (Python ${PYTHON_VERSION})"
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | awk '{print $1}' | grep -qx rl_gym; then
        echo "    复用已有 conda 环境: rl_gym"
        conda activate rl_gym
    else
        conda create -n rl_gym python="${PYTHON_VERSION}" -y
        conda activate rl_gym
    fi
else
    PY_BIN=""
    for candidate in "python${PYTHON_VERSION}" python3 python; do
        if command -v "${candidate}" &>/dev/null; then
            PY_BIN="${candidate}"
            break
        fi
    done
    if [[ -z "${PY_BIN}" ]]; then
        echo "错误: 未找到可用的 Python 解释器"
        exit 1
    fi
    echo "    使用解释器: ${PY_BIN}"
    if [[ ! -d "${VENV_DIR}" ]]; then
        "${PY_BIN}" -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

echo "==> 安装训练依赖（requirements_train.txt）"
pip install --upgrade pip wheel
pip install -r requirements_train.txt

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

echo "==> 验证 Atari 环境"
python - <<'PY'
import ale_py
import gymnasium as gym

env = gym.make("ALE/MsPacman-v5")
print("Atari env OK:", env.spec.id)
env.close()
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
