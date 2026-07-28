# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO algorithm, ported from ASAP's ``humanoidverse.agents.ppo.ppo.PPO``: asymmetric actor-critic
(privileged critic observations), GAE advantages, clipped surrogate + clipped value loss, entropy bonus,
and an adaptive (KL-based) learning-rate schedule. Separate actor/critic Adam optimizers, matching ASAP.

No AMP-style discriminator/auxiliary loss - this is standard on-policy PPO, driven entirely by the
tracking reward computed in ``G1AsapMotionEnv``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


class PPO:
    def __init__(
        self,
        actor_critic: ActorCritic,
        num_envs: int,
        num_steps_per_env: int,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        device: torch.device | str,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        actor_learning_rate: float = 1.0e-3,
        critic_learning_rate: float = 1.0e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
    ) -> None:
        self.device = device
        self.actor_critic = actor_critic.to(device)
        self.storage = RolloutStorage(
            num_envs, num_steps_per_env, num_actor_obs, num_critic_obs, num_actions, device
        )

        self.actor_optimizer = torch.optim.Adam(
            list(self.actor_critic.actor.parameters()) + [self.actor_critic.log_std], lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(self.actor_critic.critic.parameters(), lr=critic_learning_rate)

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl

        self._transition: dict[str, torch.Tensor] = {}

    # rollout collection

    def act(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        actions = self.actor_critic.act(actor_obs)
        self._transition = {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
            "actions": actions.detach(),
            "values": self.actor_critic.evaluate(critic_obs).detach(),
            "actions_log_prob": self.actor_critic.get_actions_log_prob(actions).detach(),
            "action_mean": self.actor_critic.action_mean.detach(),
            "action_std": self.actor_critic.action_std.detach(),
        }
        return self._transition["actions"]

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, timeouts: torch.Tensor | None = None) -> None:
        # bootstrap the value function at time-outs (episode cut off, not a true termination): the
        # transition still "continues" in expectation, so add back the discounted value of what would
        # have come next - matches scripts/skrl/train.py's `time_limit_bootstrap`/rsl_rl's convention.
        if timeouts is not None:
            rewards = rewards + self.gamma * self._transition["values"].squeeze(-1) * timeouts.float()
        self.storage.add_transitions(
            self._transition["actor_obs"],
            self._transition["critic_obs"],
            self._transition["actions"],
            rewards,
            dones,
            self._transition["values"],
            self._transition["actions_log_prob"],
            self._transition["action_mean"],
            self._transition["action_std"],
        )

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # learning

    def update(self) -> dict[str, float]:
        mean_surrogate_loss = 0.0
        mean_value_loss = 0.0
        mean_entropy = 0.0
        num_updates = self.num_learning_epochs * self.num_mini_batches

        for (
            actor_obs_b,
            critic_obs_b,
            actions_b,
            values_b,
            returns_b,
            old_actions_log_prob_b,
            advantages_b,
            old_mu_b,
            old_sigma_b,
        ) in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            self.actor_critic.update_distribution(actor_obs_b)
            actions_log_prob_b = self.actor_critic.get_actions_log_prob(actions_b)
            values_pred_b = self.actor_critic.evaluate(critic_obs_b)
            mu_b = self.actor_critic.action_mean
            sigma_b = self.actor_critic.action_std
            entropy_b = self.actor_critic.entropy

            # adaptive learning rate, from the KL divergence between the old and new policy
            if self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_b / old_sigma_b + 1.0e-5)
                        + (old_sigma_b.pow(2) + (old_mu_b - mu_b).pow(2)) / (2.0 * sigma_b.pow(2))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = kl.mean()
                    if kl_mean > self.desired_kl * 2.0:
                        lr = max(1.0e-5, self.actor_learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        lr = min(1.0e-2, self.actor_learning_rate * 1.5)
                    else:
                        lr = self.actor_learning_rate
                    self.actor_learning_rate = lr
                    self.critic_learning_rate = lr
                    for param_group in self.actor_optimizer.param_groups:
                        param_group["lr"] = lr
                    for param_group in self.critic_optimizer.param_groups:
                        param_group["lr"] = lr

            # clipped surrogate loss
            ratio = torch.exp(actions_log_prob_b - old_actions_log_prob_b.squeeze(-1))
            surrogate = -advantages_b.squeeze(-1) * ratio
            surrogate_clipped = -advantages_b.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # clipped value loss
            if self.use_clipped_value_loss:
                values_clipped = values_b.squeeze(-1) + (values_pred_b - values_b.squeeze(-1)).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values_pred_b - returns_b.squeeze(-1)).pow(2)
                value_losses_clipped = (values_clipped - returns_b.squeeze(-1)).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_b.squeeze(-1) - values_pred_b).pow(2).mean()

            actor_loss = surrogate_loss - self.entropy_coef * entropy_b.mean()
            critic_loss = self.value_loss_coef * value_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor_critic.actor.parameters()) + [self.actor_critic.log_std], self.max_grad_norm
            )
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            mean_surrogate_loss += surrogate_loss.item()
            mean_value_loss += value_loss.item()
            mean_entropy += entropy_b.mean().item()

        self.storage.clear()
        return {
            "surrogate_loss": mean_surrogate_loss / num_updates,
            "value_loss": mean_value_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "actor_lr": self.actor_learning_rate,
            "critic_lr": self.critic_learning_rate,
        }

    # checkpointing

    def save(self, path: str, extra: dict | None = None) -> None:
        state = {
            "model_state_dict": self.actor_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load(self, path: str, load_optimizer: bool = True) -> dict:
        state = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(state["model_state_dict"])
        if load_optimizer:
            self.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
        return state
