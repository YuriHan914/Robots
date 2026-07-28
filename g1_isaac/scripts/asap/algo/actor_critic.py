# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asymmetric actor-critic network, ported from ASAP's ``humanoidverse.agents.modules`` (``BaseModule``
+ ``PPOActor``/``PPOCritic``): plain MLPs for the actor (Gaussian policy mean) and critic (value
function), with a state-independent, learned per-action log-std - no auxiliary/adversarial losses.

The actor is trained on the (non-privileged) "actor" observation and the critic on the (privileged)
"critic" observation exposed by ``G1AsapMotionEnv`` via ``obs["policy"]`` /
``extras["observations"]["critic"]`` respectively.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
}


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    act_cls = _ACTIVATIONS[activation]
    dims = [input_dim, *hidden_dims]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), act_cls()]
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 0.8,
    ) -> None:
        super().__init__()
        self.actor = _build_mlp(num_actor_obs, num_actions, hidden_dims, activation)
        self.critic = _build_mlp(num_critic_obs, 1, hidden_dims, activation)
        # state-independent learned std (ASAP's PPOActor.std), not network-produced
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs: torch.Tensor) -> None:
        mean = self.actor(actor_obs)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Sample an action from the current policy distribution (call `update_distribution` first)."""
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (the Gaussian mean), used at eval/play time."""
        return self.actor(actor_obs)

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)
