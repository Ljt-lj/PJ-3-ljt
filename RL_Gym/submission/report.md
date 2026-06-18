# RL实践 - MsPacman DQN 实验报告（自动生成草稿）

生成时间: 2026-06-18T19:56:45

## 1. 任务与环境
- 环境: `ALE/MsPacman-v5`
- 算法: DQN (Deep Q-Network)
- 观测: 84x84 灰度图，4 帧堆叠

## 2. 网络结构
- Conv(4→32, k=8, s=4) → Conv(32→64, k=4, s=2) → Conv(64→64, k=3, s=1)
- Flatten → FC(3136→512) → FC(512→动作数)

## 3. 超参数
- 总训练步数: 5,000,000
- 学习率: 0.0001
- Replay Buffer: 400,000
- Batch size: 32
- Gamma: 0.99
- Learning starts: 80,000
- Epsilon: 1.0 → 0.01

## 4. 评估结果
- 评估局数: 10
- 平均得分: **48.8**
- 最高得分: **118.0**
- 最低得分: 4.0
- 标准差: 46.7
- 各局得分: [118.0, 4.0, 5.0, 93.0, 20.0, 12.0, 95.0, 19.0, 8.0, 114.0]

## 5. 提交视频
- 文件: `submission/pacman_submit.mp4`
- 录制种子: 3
- 该局得分: **127.0**

## 6. 运行环境
- 设备: cuda (AMD Radeon Graphics)
- PyTorch: 2.9.1+rocm6.3
- 权重文件: `runs/ALE_MsPacman-v5__MsPacman-v5__1__1781765213/MsPacman-v5.pth`

## 7. 简要分析（请按需补充）
- Agent 通过卷积网络从像素输入学习 Q 值，使用 epsilon-greedy 探索。
- 训练后期评估得分反映了吃豆、避鬼等行为的学习效果。
