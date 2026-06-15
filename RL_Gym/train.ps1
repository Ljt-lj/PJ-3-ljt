# RL实践 - MsPacman DQN 训练（Windows 本地，默认 CPU）
# 完整训练建议在云端 AMD GPU 上运行 train_amd.sh

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python dqn_atari.py `
    --exp-name "MsPacman-v5" `
    --capture-video True `
    --save-model True `
    --env-id "ALE/MsPacman-v5" `
    --total-timesteps 5000000 `
    --buffer-size 400000 `
    --seed 1 `
    --cuda False
