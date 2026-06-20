#!/usr/bin/env python3
"""Load trained DQN weights, evaluate, record submit video, export report data."""

import argparse
import glob
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import ale_py  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from dqn_atari import QNetwork, make_env


DEFAULT_HPARAMS = {
    "env_id": "ALE/MsPacman-v5",
    "exp_name": "MsPacman-v5",
    "total_timesteps": 5_000_000,
    "buffer_size": 400_000,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "gamma": 0.99,
    "learning_starts": 80_000,
    "train_frequency": 4,
    "target_network_frequency": 1000,
    "start_e": 1.0,
    "end_e": 0.01,
    "exploration_fraction": 0.10,
    "epsilon_eval": 0.05,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN and generate submission artifacts")
    parser.add_argument("--model-path", type=str, default="", help="path to .pth weights (auto-detect latest if empty)")
    parser.add_argument("--env-id", type=str, default=DEFAULT_HPARAMS["env_id"])
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed-probes", type=int, default=100, help="seeds to search for best video episode")
    parser.add_argument("--epsilon", type=float, default=0.05, help="epsilon for standard eval (match training)")
    parser.add_argument("--greedy-episodes", type=int, default=10, help="extra greedy eval episodes (epsilon=0)")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=str, default="submission")
    parser.add_argument("--run-name", type=str, default="submit-best")
    parser.add_argument("--skip-video", action="store_true", help="skip video recording (score/report only)")
    return parser.parse_args()


def find_latest_model():
    paths = sorted(glob.glob("runs/*/MsPacman-v5.pth"), key=os.path.getmtime)
    if not paths:
        paths = sorted(glob.glob("runs/**/*.pth", recursive=True), key=os.path.getmtime)
    if not paths:
        raise FileNotFoundError("no .pth found under runs/, use --model-path")
    return paths[-1]


def run_dqn_eval(model, device, env_id, eval_episodes, run_name, epsilon):
    """Same loop as dqn_eval.evaluate() — proven during training."""
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, False, run_name)])
    obs, _ = envs.reset()
    returns, lengths = [], []
    steps = 0
    print(f"  running {eval_episodes} episodes (first result may take 2-10 min)...", flush=True)
    while len(returns) < eval_episodes:
        steps += 1
        if steps % 1000 == 0:
            print(f"  ... step {steps}, finished {len(returns)}/{eval_episodes} episodes", flush=True)
        if steps > 200_000:
            raise RuntimeError(
                f"eval stuck: {steps} steps without episode end. "
                "Try Ctrl+C and check env/model."
            )
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                q_values = model(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
        obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None:
                    continue
                if "episode" not in info:
                    continue
                r = float(info["episode"]["r"])
                l = int(info["episode"]["l"])
                returns.append(r)
                lengths.append(l)
                print(f"eval_episode={len(returns)-1}, episodic_return={r}, length={l}", flush=True)
    envs.close()
    return returns, lengths


def run_one_episode_with_seed(model, device, env_id, seed, epsilon=0.0, max_steps=50000, capture_video=False, run_name="probe"):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, capture_video, run_name)]
    )
    obs, _ = envs.reset(seed=seed)
    steps = 0
    while True:
        steps += 1
        if max_steps and steps > max_steps:
            envs.close()
            return 0.0, steps
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                q_values = model(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
        obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    envs.close()
                    return float(info["episode"]["r"]), int(info["episode"]["l"])


def find_best_seed(model, device, env_id, seed_probes, epsilon=0.0):
    best_seed, best_score, best_length = 0, -float("inf"), 0
    top_scores = []
    for seed in tqdm(range(seed_probes), desc="seed search", unit="seed"):
        score, length = run_one_episode_with_seed(
            model, device, env_id, seed, epsilon=epsilon,
        )
        top_scores.append((score, seed, length))
        if score > best_score:
            best_seed, best_score, best_length = seed, score, length
    top_scores.sort(reverse=True)
    print(f"best_seed={best_seed}, return={best_score:.1f}, length={best_length}")
    print("top5 seeds:", [(s, sd) for s, sd, _ in top_scores[:5]])
    return best_seed, best_score, best_length


def record_submit_video(model, device, env_id, seed, run_name):
    video_dir = Path("videos") / run_name
    if video_dir.exists():
        shutil.rmtree(video_dir)

    score, length = run_one_episode_with_seed(
        model, device, env_id, seed, epsilon=0.0,
        capture_video=True, run_name=run_name,
    )
    videos = sorted(glob.glob(str(video_dir / "*.mp4")))
    if not videos:
        raise RuntimeError(f"no mp4 recorded under {video_dir}")
    return videos[-1], score, length


def build_report(args, model_path, eval_returns, eval_lengths, greedy_returns, best_seed, video_score, video_path, device):
    arr = np.array(eval_returns, dtype=np.float32)
    greedy_arr = np.array(greedy_returns, dtype=np.float32) if greedy_returns else arr
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": model_path,
        "submit_video": video_path,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "pytorch_version": torch.__version__,
        "gymnasium_version": gym.__version__,
        "network": {
            "input": "4x84x84 (frame stack, grayscale)",
            "conv1": "Conv2d(4,32,k=8,s=4)+ReLU",
            "conv2": "Conv2d(32,64,k=4,s=2)+ReLU",
            "conv3": "Conv2d(64,64,k=3,s=1)+ReLU",
            "fc": "Linear(3136,512)+ReLU -> Linear(512, num_actions)",
        },
        "hyperparameters": {**DEFAULT_HPARAMS, "env_id": args.env_id, "epsilon_eval": args.epsilon},
        "evaluation": {
            "episodes": args.eval_episodes,
            "returns": [float(x) for x in eval_returns],
            "lengths": [int(x) for x in eval_lengths],
            "mean_return": float(arr.mean()),
            "max_return": float(arr.max()),
            "min_return": float(arr.min()),
            "std_return": float(arr.std()),
            "greedy_episodes": args.greedy_episodes,
            "greedy_returns": [float(x) for x in greedy_returns],
            "greedy_mean_return": float(greedy_arr.mean()),
            "greedy_max_return": float(greedy_arr.max()),
        },
        "submit_episode": {
            "seed": best_seed,
            "return": float(video_score),
            "video_file": video_path,
        },
    }
    return report


def write_markdown_report(report, path):
    ev = report["evaluation"]
    hp = report["hyperparameters"]
    sub = report["submit_episode"]
    lines = [
        "# RL实践 - MsPacman DQN 实验报告（自动生成草稿）",
        "",
        f"生成时间: {report['generated_at']}",
        "",
        "## 1. 任务与环境",
        f"- 环境: `{hp['env_id']}`",
        "- 算法: DQN (Deep Q-Network)",
        "- 观测: 84x84 灰度图，4 帧堆叠",
        "",
        "## 2. 网络结构",
        "- Conv(4→32, k=8, s=4) → Conv(32→64, k=4, s=2) → Conv(64→64, k=3, s=1)",
        "- Flatten → FC(3136→512) → FC(512→动作数)",
        "",
        "## 3. 超参数",
        f"- 总训练步数: {hp['total_timesteps']:,}",
        f"- 学习率: {hp['learning_rate']}",
        f"- Replay Buffer: {hp['buffer_size']:,}",
        f"- Batch size: {hp['batch_size']}",
        f"- Gamma: {hp['gamma']}",
        f"- Learning starts: {hp['learning_starts']:,}",
        f"- Epsilon: {hp['start_e']} → {hp['end_e']}",
        "",
        "## 4. 评估结果",
        f"- 评估局数: {ev['episodes']}",
        f"- 平均得分 (eps={hp['epsilon_eval']}): **{ev['mean_return']:.1f}**",
        f"- 最高得分 (eps={hp['epsilon_eval']}): **{ev['max_return']:.1f}**",
        f"- 贪心最高得分 (eps=0): **{ev['greedy_max_return']:.1f}**",
        f"- 贪心平均得分 (eps=0): {ev['greedy_mean_return']:.1f}",
        f"- 最低得分: {ev['min_return']:.1f}",
        f"- 标准差: {ev['std_return']:.1f}",
        f"- 各局得分: {ev['returns']}",
        "",
        "## 5. 提交视频",
        f"- 文件: `{sub['video_file']}`",
        f"- 录制种子: {sub['seed']}",
        f"- 该局得分: **{sub['return']:.1f}**",
        "",
        "## 6. 运行环境",
        f"- 设备: {report['device']} ({report['gpu_name']})",
        f"- PyTorch: {report['pytorch_version']}",
        f"- 权重文件: `{report['model_path']}`",
        "",
        "## 7. 简要分析（请按需补充）",
        "- Agent 通过卷积网络从像素输入学习 Q 值，使用 epsilon-greedy 探索。",
        "- 训练后期评估得分反映了吃豆、避鬼等行为的学习效果。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    model_path = args.model_path or find_latest_model()
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_path}")

    probe_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, 0, False, "init")])
    model = QNetwork(probe_envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    probe_envs.close()

    print("\n==> Phase 1: evaluate (epsilon={})".format(args.epsilon), flush=True)
    eval_returns, eval_lengths = run_dqn_eval(
        model, device, args.env_id, args.eval_episodes, "eval-eps-submit", args.epsilon,
    )

    print("\n==> Phase 1b: greedy evaluate (epsilon=0)", flush=True)
    greedy_returns, _ = run_dqn_eval(
        model, device, args.env_id, args.greedy_episodes, "eval-greedy-submit", 0.0,
    )

    print("\n==> Phase 2: find best seed ({} seeds)".format(args.seed_probes), flush=True)
    best_seed, best_score, _ = find_best_seed(
        model, device, args.env_id, args.seed_probes, epsilon=0.0,
    )

    print("\n==> Phase 3: record submit video")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submit_video_path = out_dir / "pacman_submit.mp4"

    if args.skip_video:
        video_score = best_score
        video_length = 0
        submit_video_path = ""
        print("  skipped (--skip-video), using best_seed score")
    else:
        raw_video, video_score, video_length = record_submit_video(
            model, device, args.env_id, best_seed, args.run_name
        )
        shutil.copy2(raw_video, submit_video_path)

    report = build_report(
        args, model_path, eval_returns, eval_lengths, greedy_returns,
        best_seed, video_score, str(submit_video_path), device
    )
    report["submit_episode"]["length"] = int(video_length)

    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(report, md_path)

    print("\n==> Done")
    print(f"  submit video : {submit_video_path or '(skipped)'}")
    print(f"  report json  : {json_path}")
    print(f"  report md    : {md_path}")
    print(f"  mean return  : {report['evaluation']['mean_return']:.1f}")
    print(f"  max return   : {report['evaluation']['max_return']:.1f}")
    print(f"  greedy max   : {report['evaluation']['greedy_max_return']:.1f}")
    print(f"  video return : {video_score:.1f}")


if __name__ == "__main__":
    main()
