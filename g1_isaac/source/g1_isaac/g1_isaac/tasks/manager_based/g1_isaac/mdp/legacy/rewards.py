# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the G1 locomotion (stand/walk/run) task.

`feet_air_time`, `feet_slide`, `track_lin_vel_xy_yaw_frame_exp`, `track_ang_vel_z_world_exp`, and
`stand_still_joint_deviation_l1` are carried over unmodified from
``tasks/direct/g1_isaac/locomotion_example/mdp/rewards.py`` (itself IsaacLab's own generic
velocity-locomotion reward set).

`goal_reached_bonus` was this task's one addition for the target-point ("go to (x, y)") mode
implemented by `mdp.commands.GoalConditionedVelocityCommand` - kept sparse and separate from
`track_lin_vel_xy_exp` so goal-seeking envs would be trained by the exact same velocity-tracking signal
as random-velocity envs, rather than a second, possibly-conflicting distance-based reward. As of
2026-09-01, target-point tracking is no longer part of training at all (`rel_goal_envs=0.0` in
`CommandsCfg.base_velocity` - see that module's docstring): it's a fixed distance/bearing formula, not
something a policy needs to learn, so it's applied at inference time instead, directly on the trained
velocity-tracking policy. `goal_reached_bonus` is kept here, unwired from `RewardsCfg`, in case
`rel_goal_envs` is ever raised again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import GoalConditionedVelocityCommand


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value.

    Kept from the default IsaacLab cartpole scaffold this package started as (used by the still-registered
    `Template-G1-Isaac-v0` cartpole task's `RewardsCfg.pole_pos`, in `g1_isaac_env_cfg.py`) - not used by
    the G1 locomotion task below, but removing it would break that unrelated existing registration.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def both_feet_airborne(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize steps where neither foot has ground contact, scaled down at running speeds.

    Returns up to 1.0 (meant to be used with a negative weight - the `RewTerm.weight` itself is the
    penalty's max, applied in full at zero commanded speed) for envs where all bodies selected by
    `sensor_cfg` (the two ankle-roll feet) currently have contact force below `threshold`. Uses the same
    contact-force convention as `feet_slide`/`undesired_contacts` (net force norm > threshold = "in
    contact"), so a step with at least one foot down is never penalized, including normal single-stance
    walking.

    A real running gait has a legitimate flight phase where both feet leave the ground at once, so the
    penalty is ramped down by commanded speed (`‖vx,vy‖`, same quantity `feet_air_time` gates on) - full
    weight at zero speed, linearly down to zero at the command term's own configured top speed
    (`max(|ranges.lin_vel_x[0]|, |ranges.lin_vel_x[1]|)`, read directly from `CommandsCfg.base_velocity`
    instead of a separately-tuned threshold, so this stays in sync if that range ever changes).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    airborne = (~in_contact.any(dim=1)).float()

    command_ranges = env.command_manager.get_term(command_name).cfg.ranges
    max_speed = max(abs(command_ranges.lin_vel_x[0]), abs(command_ranges.lin_vel_x[1]))
    speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    scale = torch.clamp(1.0 - speed / max_speed, min=0.0, max=1.0)
    return airborne * scale


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def goal_reached_bonus(
    env: ManagerBasedRLEnv, command_name: str, distance_threshold: float, speed_threshold: float
) -> torch.Tensor:
    """Sparse bonus for goal envs that have arrived and (nearly) stopped at their target point.

    Zero for non-goal envs. Deliberately the *only* target-navigation-specific reward term - the
    approach/heading behavior itself is already driven by `track_lin_vel_xy_exp`/`track_ang_vel_z_exp`
    tracking the velocity `GoalConditionedVelocityCommand` computes toward the goal each step.
    """
    command_term: GoalConditionedVelocityCommand = env.command_manager.get_term(command_name)
    asset = env.scene[SceneEntityCfg("robot").name]
    distance = torch.norm(command_term.goal_pos_w[:, :2] - asset.data.root_pos_w[:, :2], dim=-1)
    speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=-1)
    reached = command_term.is_goal_env & (distance < distance_threshold) & (speed < speed_threshold)
    return reached.float()
