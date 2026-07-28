# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-policy rollout buffer + GAE, ported from ASAP's ``humanoidverse.agents.modules.data_utils.
RolloutStorage`` (trimmed to the fixed set of tensors PPO needs, instead of its generic named-key
registry, since this script only ever runs one algorithm)."""

from __future__ import annotations

import torch


class RolloutStorage:
    def __init__(
        self,
        num_envs: int,
        num_steps_per_env: int,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        device: torch.device | str,
    ) -> None:
        self.device = device
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env

        shape = (num_steps_per_env, num_envs)
        self.actor_obs = torch.zeros(*shape, num_actor_obs, device=device)
        self.critic_obs = torch.zeros(*shape, num_critic_obs, device=device)
        self.actions = torch.zeros(*shape, num_actions, device=device)
        self.rewards = torch.zeros(*shape, 1, device=device)
        self.dones = torch.zeros(*shape, 1, dtype=torch.bool, device=device)
        self.values = torch.zeros(*shape, 1, device=device)
        self.actions_log_prob = torch.zeros(*shape, 1, device=device)
        self.action_mean = torch.zeros(*shape, num_actions, device=device)
        self.action_std = torch.zeros(*shape, num_actions, device=device)

        self.advantages = torch.zeros(*shape, 1, device=device)
        self.returns = torch.zeros(*shape, 1, device=device)

        self.step = 0

    def add_transitions(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        actions_log_prob: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        if self.step >= self.num_steps_per_env:
            raise RuntimeError("Rollout buffer overflow: call `compute_returns` and reset before adding more.")
        self.actor_obs[self.step] = actor_obs
        self.critic_obs[self.step] = critic_obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards.view(-1, 1)
        self.dones[self.step] = dones.view(-1, 1)
        self.values[self.step] = values.view(-1, 1)
        self.actions_log_prob[self.step] = actions_log_prob.view(-1, 1)
        self.action_mean[self.step] = action_mean
        self.action_std[self.step] = action_std
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        """Backward-recursion GAE, matching ASAP's ``PPO._compute_returns``."""
        advantage = torch.zeros(self.num_envs, 1, device=self.device)
        next_values = last_values.view(-1, 1)
        for step in reversed(range(self.num_steps_per_env)):
            not_done = (~self.dones[step]).float()
            delta = self.rewards[step] + gamma * not_done * next_values - self.values[step]
            advantage = delta + gamma * lam * not_done * advantage
            self.returns[step] = advantage + self.values[step]
            next_values = self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1.0e-8)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
        batch_size = self.num_envs * self.num_steps_per_env
        mini_batch_size = batch_size // num_mini_batches

        def flatten(x: torch.Tensor) -> torch.Tensor:
            return x.flatten(0, 1)

        actor_obs = flatten(self.actor_obs)
        critic_obs = flatten(self.critic_obs)
        actions = flatten(self.actions)
        values = flatten(self.values)
        returns = flatten(self.returns)
        old_actions_log_prob = flatten(self.actions_log_prob)
        advantages = flatten(self.advantages)
        old_mu = flatten(self.action_mean)
        old_sigma = flatten(self.action_std)

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for i in range(num_mini_batches):
                start, end = i * mini_batch_size, (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield (
                    actor_obs[batch_idx],
                    critic_obs[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    advantages[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )
