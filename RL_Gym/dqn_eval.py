import random
from typing import Callable, Iterator, Tuple

import gymnasium as gym
import numpy as np
import torch


def iter_finished_episodes(
    infos: dict,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
    ep_return: float | None = None,
    ep_length: int | None = None,
) -> Iterator[Tuple[float, int]]:
    """Yield (return, length) for episodes finished on this step (Gymnasium 0.x/1.x)."""
    if "final_info" in infos:
        for info in infos["final_info"]:
            if info is not None and "episode" in info:
                yield float(info["episode"]["r"]), int(info["episode"]["l"])
        return

    if "episode" in infos and "_episode" in infos:
        ep = infos["episode"]
        mask = infos["_episode"]
        n = len(mask)
        for i in range(n):
            if not mask[i]:
                continue
            r = ep["r"][i] if isinstance(ep["r"], np.ndarray) else ep["r"]
            l = ep["l"][i] if isinstance(ep["l"], np.ndarray) else ep["l"]
            yield float(r), int(l)
        return

    if (
        terminated is not None
        and truncated is not None
        and ep_return is not None
        and ep_length is not None
        and (terminated.any() or truncated.any())
    ):
        yield float(ep_return), int(ep_length)


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episode: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    ep_return, ep_length = 0.0, 0
    while len(episodic_returns) < eval_episode:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        ep_return += float(rewards[0])
        ep_length += 1
        for r, _ in iter_finished_episodes(
            infos, terminated, truncated, ep_return, ep_length
        ):
            print(f"eval_episode={len(episodic_returns)}, episodic_return={r}")
            episodic_returns.append(r)
            ep_return, ep_length = 0.0, 0
        obs = next_obs

    envs.close()
    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from dqn_atari import QNetwork, make_env

    model_path = hf_hub_download(repo_id="cleanrl/CartPole-v1-dqn-seed1", filename="dqn.cleanrl_model")
    # model_path = ".pth"
    evaluate(
        model_path,
        make_env,
        "CartPole-v1",
        eval_episode=0,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False
    )           