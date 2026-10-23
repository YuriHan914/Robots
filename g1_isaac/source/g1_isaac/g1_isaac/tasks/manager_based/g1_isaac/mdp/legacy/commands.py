# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity command extended with a target-point ("go to (x, y)") mode.

`GoalConditionedVelocityCommand` subclasses IsaacLab's own
:class:`isaaclab.envs.mdp.commands.velocity_command.UniformVelocityCommand` (the same command term
``locomotion_example/velocity_env_cfg.py`` uses for stand/walk) and only changes what a *subset* of
envs' commanded (vx, vy, wz) actually comes from: instead of a fixed random draw, a `rel_goal_envs`
fraction of resamples get a random target point instead, and every step their commanded velocity is
recomputed as "the velocity that would carry the robot toward that point, decelerating to a stop once
close" (same idea as `isaaclab.envs.mdp.commands.pose_2d_command.UniformPose2dCommand`, but expressed as
a velocity command rather than a pose command). Everything downstream - the observation
(`mdp.generated_commands`), the tracking rewards (`track_lin_vel_xy_exp`/`track_ang_vel_z_exp`), and the
terrain curriculum (`terrain_levels_vel`) - reads `command`/`vel_command_b` exactly as it would for a
plain random-velocity env, so target-point navigation needs no reward/observation terms of its own
beyond the sparse `goal_reached_bonus` in `mdp/rewards.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GoalConditionedVelocityCommand(UniformVelocityCommand):
    """Velocity command where a fraction of envs track a target point instead of a fixed velocity.

    For envs *not* selected as "goal envs" this behaves exactly like the parent
    :class:`UniformVelocityCommand` (random vx/vy/wz on the shared ``resampling_time_range`` cadence,
    optional heading control, optional standing). For goal envs, a target (x, y) is sampled around the
    robot's *current* position (radius drawn from ``cfg.goal_pos_range``), and every step the commanded
    velocity is recomputed from the direction/distance to that point: speed ramps down linearly with
    distance (``min(max_speed, goal_decel_gain * distance)``) and stops entirely inside
    ``goal_arrival_radius``; the angular velocity command turns the base to face the target, reusing the
    parent's ``heading_control_stiffness`` proportional gain. Unlike the parent's fixed-time resampling,
    a goal env keeps its target until it actually arrives - the shared timer is held open every step
    while still travelling, and arrival triggers an immediate resample instead (see ``_update_command``).
    """

    cfg: GoalConditionedVelocityCommandCfg

    # Used to keep a still-travelling goal env's shared `time_left` timer from ever hitting zero (see
    # _update_command) - only needs to outlast the longest possible episode.
    _GOAL_ENV_TIME_LEFT = 1.0e6

    def __init__(self, cfg: GoalConditionedVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_goal_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["error_pos_to_goal"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"\n\tGoal-point probability: {self.cfg.rel_goal_envs}"
        return msg

    def _update_metrics(self):
        super()._update_metrics()
        goal_env_ids = self.is_goal_env.nonzero(as_tuple=False).flatten()
        if len(goal_env_ids) > 0:
            self.metrics["error_pos_to_goal"][goal_env_ids] = torch.norm(
                self.goal_pos_w[goal_env_ids, :2] - self.robot.data.root_pos_w[goal_env_ids, :2], dim=-1
            )

    def _resample_command(self, env_ids: Sequence[int]):
        # standard random vx/vy/wz (+ optional heading/standing) draw for every env, including the ones
        # about to be turned into goal envs below - this way a goal env that later gets deselected (next
        # resample) always has a valid non-goal command already sitting in vel_command_b/heading_target.
        super()._resample_command(env_ids)

        r = torch.empty(len(env_ids), device=self.device)
        self.is_goal_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_goal_envs
        # a goal-seeking env should actually walk toward its target, not be pinned in place by the
        # standing-env draw it may have also picked up above.
        self.is_standing_env[env_ids] = self.is_standing_env[env_ids] & ~self.is_goal_env[env_ids]

        goal_env_ids = torch.as_tensor(env_ids, device=self.device)[self.is_goal_env[env_ids]]
        if len(goal_env_ids) == 0:
            return
        angle = torch.empty(len(goal_env_ids), device=self.device).uniform_(-torch.pi, torch.pi)
        radius = torch.empty(len(goal_env_ids), device=self.device).uniform_(*self.cfg.goal_pos_range)
        self.goal_pos_w[goal_env_ids, 0] = self.robot.data.root_pos_w[goal_env_ids, 0] + radius * torch.cos(angle)
        self.goal_pos_w[goal_env_ids, 1] = self.robot.data.root_pos_w[goal_env_ids, 1] + radius * torch.sin(angle)
        self.goal_pos_w[goal_env_ids, 2] = self.robot.data.root_pos_w[goal_env_ids, 2]

    def _update_command(self):
        # parent handles: heading-command angular velocity for is_heading_env, zeroing for
        # is_standing_env - goal envs are excluded from is_standing_env (see _resample_command) and get
        # their vx/vy/wz fully overwritten below regardless of is_heading_env, so no conflict either way.
        super()._update_command()

        goal_env_ids = self.is_goal_env.nonzero(as_tuple=False).flatten()
        if len(goal_env_ids) == 0:
            return
        target_vec_w = self.goal_pos_w[goal_env_ids, :2] - self.robot.data.root_pos_w[goal_env_ids, :2]
        target_vec_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w[goal_env_ids]),
            torch.cat([target_vec_w, torch.zeros_like(target_vec_w[:, :1])], dim=-1),
        )
        distance = torch.norm(target_vec_b[:, :2], dim=-1)
        bearing = torch.atan2(target_vec_b[:, 1], target_vec_b[:, 0])

        max_speed = max(abs(self.cfg.ranges.lin_vel_x[0]), abs(self.cfg.ranges.lin_vel_x[1]))
        desired_speed = torch.clamp(self.cfg.goal_decel_gain * distance, min=0.0, max=max_speed)
        arrived = distance < self.cfg.goal_arrival_radius
        desired_speed = torch.where(arrived, torch.zeros_like(desired_speed), desired_speed)

        vx = torch.clamp(desired_speed * torch.cos(bearing), *self.cfg.ranges.lin_vel_x)
        vy = torch.clamp(desired_speed * torch.sin(bearing), *self.cfg.ranges.lin_vel_y)
        wz = torch.where(
            arrived,
            torch.zeros_like(bearing),
            torch.clamp(self.cfg.heading_control_stiffness * bearing, *self.cfg.ranges.ang_vel_z),
        )
        self.vel_command_b[goal_env_ids, 0] = vx
        self.vel_command_b[goal_env_ids, 1] = vy
        self.vel_command_b[goal_env_ids, 2] = wz

        # goal envs don't follow the shared rel_goal_envs-driven resampling_time_range cadence - a
        # target is kept until the env actually arrives, not abandoned mid-approach just because the
        # fixed window elapsed. Still-travelling envs get their shared `time_left` timer topped up every
        # step so CommandManager.compute()'s periodic check never fires for them; arrival triggers an
        # immediate resample instead (this happens after RewardManager.compute() already ran for this
        # step - see ManagerBasedRLEnv.step() - so goal_reached_bonus still sees the pre-resample goal).
        still_going_ids = goal_env_ids[~arrived]
        if len(still_going_ids) > 0:
            self.time_left[still_going_ids] = self._GOAL_ENV_TIME_LEFT
        arrived_ids = goal_env_ids[arrived]
        if len(arrived_ids) > 0:
            self._resample(arrived_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer = self._make_goal_pos_visualizer()
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _make_goal_pos_visualizer(self):
        from isaaclab.markers import VisualizationMarkers

        return VisualizationMarkers(self.cfg.goal_pos_visualizer_cfg)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if hasattr(self, "goal_pos_visualizer"):
            goal_env_ids = self.is_goal_env.nonzero(as_tuple=False).flatten()
            if len(goal_env_ids) > 0:
                self.goal_pos_visualizer.visualize(self.goal_pos_w[goal_env_ids])


@configclass
class GoalConditionedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`GoalConditionedVelocityCommand`."""

    class_type: type = GoalConditionedVelocityCommand

    rel_goal_envs: float = 0.0
    """Fraction of resampled envs that get a target-point command instead of a random velocity."""

    goal_pos_range: tuple[float, float] = MISSING
    """Distance range [m] a goal env's target point is sampled at, around its current position."""

    goal_decel_gain: float = 2.0
    """Proportional gain converting distance-to-goal [m] into desired approach speed [m/s]."""

    goal_arrival_radius: float = 0.3
    """Distance [m] inside which a goal env is considered "arrived" and its command decays to zero."""

    goal_pos_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_position"
    )
    """Marker shown at each goal env's current target point."""
    goal_pos_visualizer_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
