import gymnasium as gym
import torch

from dqn_atari import QNetwork


class FakeEnv:
    single_action_space = gym.spaces.Discrete(9)


def main():
    net = QNetwork(FakeEnv())
    x = torch.zeros(2, 4, 84, 84)
    out = net(x)
    assert out.shape == (2, 9), f"unexpected shape: {out.shape}"
    print("QNetwork OK, output shape:", tuple(out.shape))


if __name__ == "__main__":
    main()
