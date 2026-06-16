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
    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        if [[ -d "${VENV_DIR}" ]]; then
            echo "    删除损坏的虚拟环境: ${VENV_DIR}"
            rm -rf "${VENV_DIR}"
        fi
        echo "    创建虚拟环境: ${VENV_DIR}"
        if ! "${PY_BIN}" -m venv "${VENV_DIR}"; then
            echo "错误: 无法创建虚拟环境。Debian/Ubuntu 可尝试: apt install python3-venv"
            exit 1
        fi
    else
        echo "    复用已有虚拟环境: ${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

install_torch() {
    local index_url="$1"
    local label="$2"
    echo "==> 安装 ${label} PyTorch: ${index_url}"
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install torch --index-url "${index_url}"
}

echo "==> 安装训练依赖（requirements_train.txt，先跳过 torch）"
pip install --upgrade pip wheel
grep -v '^torch' requirements_train.txt > /tmp/requirements_no_torch.txt
pip install -r /tmp/requirements_no_torch.txt

# 其他依赖可能已拉入错误版本的 torch，先卸载再按 GPU 类型重装
if command -v rocm-smi &>/dev/null; then
    install_torch "${ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.3}" "AMD ROCm"
elif command -v nvidia-smi &>/dev/null; then
    install_torch "${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}" "NVIDIA CUDA"
else
    echo "==> 未检测到 GPU，安装 CPU 版 PyTorch"
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install torch
fi

echo "==> 检测 GPU"
python - <<'PY'
import shutil
import sys
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

has_rocm = shutil.which("rocm-smi") is not None
has_nvidia = shutil.which("nvidia-smi") is not None
version = torch.__version__

if has_rocm and "+cu" in version:
    print("错误: 机器是 AMD ROCm，但安装了 NVIDIA CUDA 版 PyTorch。", file=sys.stderr)
    print("请执行: pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/rocm6.3", file=sys.stderr)
    sys.exit(1)
if has_rocm and not torch.cuda.is_available():
    print("警告: 检测到 ROCm GPU，但 torch.cuda.is_available() 为 False。", file=sys.stderr)
    print("可尝试: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2", file=sys.stderr)
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
