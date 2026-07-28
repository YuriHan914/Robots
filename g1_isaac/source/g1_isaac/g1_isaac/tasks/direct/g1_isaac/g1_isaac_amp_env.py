# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 imitation-learning (AMP) direct environment.

Trains a policy to imitate a retargeted mocap clip (see ``scripts/play_motion.py`` and
``data/dance1_retarget_g1.csv``) using skrl's Adversarial Motion Priors agent: a discriminator is
trained to distinguish policy rollouts from the reference motion, and the resulting "style reward"
(instead of a hand-designed task reward) drives the policy.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .g1_isaac_env_cfg import G1AmpEnvCfg
from .motions import MotionLoader


class G1AmpEnv(DirectRLEnv):
    cfg: G1AmpEnvCfg

    def __init__(self, cfg: G1AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # load reference motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, fps=self.cfg.motion_fps, device=self.device)
        # the robot has more DOFs than the motion clip covers (e.g. hand/finger joints on this asset,
        # see scripts/play_motion.py); only the clip's 29 body joints are part of the action/AMP-observation
        # space, in the clip's own column order - the remaining joints are always driven to their default pose
        self.body_joint_ids, _ = self.robot.find_joints(self._motion_loader.dof_names, preserve_order=True)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # non-body joints (e.g. fingers) are always held at their default pose
        joint_pos_target = self.robot.data.default_joint_pos.clone()
        joint_pos_target[:, self.body_joint_ids] = (
            self.robot.data.default_joint_pos[:, self.body_joint_ids] + self.actions * self.cfg.action_scale
        )
        self.robot.set_joint_position_target(joint_pos_target)

    def _get_observations(self) -> dict:
        # build task/AMP observation (body-joint + root state only; the clip has no per-body key points)
        obs = compute_obs(
            self.robot.data.joint_pos[:, self.body_joint_ids],
            self.robot.data.joint_vel[:, self.body_joint_ids],
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_ang_vel_w,
        )

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # mostly imitation-driven: the AMP discriminator supplies the style reward, blended with this
        # constant survival signal (see skrl_amp_cfg.yaml task_reward_scale / style_reward_scale)
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample the reference motion
        dof_positions, dof_velocities, root_positions, root_rotations, root_linear_velocities, root_angular_velocities = (
            self._motion_loader.sample(num_samples=num_samples, times=times)
        )

        # get root transforms
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = root_positions + self.scene.env_origins[env_ids]
        root_state[:, 2] += self.cfg.reset_root_height_offset  # avoid ground penetration on reset
        root_state[:, 3:7] = root_rotations
        root_state[:, 7:10] = root_linear_velocities
        root_state[:, 10:13] = root_angular_velocities
        # get DOFs state: non-body joints (e.g. fingers) are held at their default pose
        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_vel = self.robot.data.default_joint_vel[env_ids].clone()
        dof_pos[:, self.body_joint_ids] = dof_positions
        dof_vel[:, self.body_joint_ids] = dof_velocities

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the ones specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        dof_positions, dof_velocities, root_positions, root_rotations, root_linear_velocities, root_angular_velocities = (
            self._motion_loader.sample(num_samples=num_samples, times=times)
        )
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions,
            dof_velocities,
            root_positions,
            root_rotations,
            root_linear_velocities,
            root_angular_velocities,
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
        ),
        dim=-1,
    )
    return obs
