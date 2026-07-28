# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 motion-tracking environment, ported from ASAP's ``humanoidverse.envs.motion_tracking`` task.

Reproduces the core pieces of ASAP's (https://github.com/LeCAR-Lab/ASAP) motion-tracking formulation
using Isaac Lab's own primitives instead of ASAP's own IsaacGym-based simulator abstraction and
MJCF/skeleton-based motion library (``humanoidverse.utils.motion_lib``):

* **Reward** - DeepMimic-style ``exp(-error / sigma)`` tracking rewards over body position, body
  rotation, body linear/angular velocity, feet position, joint position/velocity, and a "3-point"
  hand+head keypoint term (ASAP's ``r_vr_3point``, approximated with real end-effector-ish links - see
  ``cfg.keypoint_body_names``); plus curriculum-scaled torque/action-rate/joint-limit penalties
  (ASAP's ``_reward_penalty_torques``/``_reward_limits_*``) and an always-on termination penalty (see
  ``g1_isaac_env_cfg.G1AsapEnvCfg`` for weights/sigmas and
  ``humanoidverse/envs/motion_tracking/motion_tracking.py``/``legged_robot_base.py`` for the ASAP
  formulas these are adapted from).
* **Asymmetric actor/critic observations** - the actor ("policy") observation excludes base linear
  velocity and tracking-error terms (sim2real-friendly); the critic observation is privileged
  (includes both). Both carry a short history, using Isaac Lab's own
  :class:`isaaclab.utils.buffers.CircularBuffer` in place of ASAP's ``HistoryHandler``.
* **Reward-penalty curriculum** - a single scalar factor in ``[0, 1]``, driven by a running average of
  episode length vs. a target fraction of the max episode length, that scales the action-rate penalty
  and tightens the motion-tracking termination threshold as the policy improves (a trimmed version of
  ASAP's ``_update_reward_penalty_curriculum``/``terminate_when_motion_far`` curricula).
* **Multi-clip motion library** - see ``motions.MotionLibrary``: each env is assigned one reference
  clip (weighted by clip duration), periodically resampled.

Per-body reference kinematics (needed for the body/feet tracking rewards) are obtained without porting
ASAP's own forward-kinematics engine: a second, non-colliding, gravity-disabled instance of the same G1
asset (``robot_ref``) is teleported to the interpolated reference-motion joint/root state every step via
``write_joint_state_to_sim``/``write_root_pose_to_sim`` followed by
:meth:`isaaclab.sim.SimulationContext.forward` (Isaac Sim's kinematic-only articulation update, i.e. no
dynamics integration) - so its ``body_pos_w``/``body_quat_w``/``body_lin_vel_w``/``body_ang_vel_w`` give
the reference body kinematics directly from Isaac Lab's own FK.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.math import quat_error_magnitude

from .g1_isaac_env_cfg import G1AsapEnvCfg
from .motions import MotionLibrary, resolve_motion_files


class G1AsapMotionEnv(DirectRLEnv):
    cfg: G1AsapEnvCfg

    def __init__(self, cfg: G1AsapEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # reference motion(s)
        motion_files = resolve_motion_files(self.cfg.motion_file)
        self._motion_library = MotionLibrary(
            motion_files=motion_files, fps=self.cfg.motion_fps, num_envs=self.num_envs, device=self.device
        )
        self._motion_library.load_motions(random_sample=not self.cfg.deterministic_motion_assignment)
        self._resample_interval_steps = max(1, round(self.cfg.motion_resample_interval_s / self.step_dt))
        self._steps_since_resample = 0

        # the robot has more DOFs than the motion clip(s) cover (e.g. finger joints); only the clip's
        # body joints are part of the action/tracking space, in the clip's own column order - the
        # remaining joints are always driven to their default pose (see G1AmpEnv for the same pattern)
        self.body_joint_ids, self._body_joint_names = self.robot.find_joints(
            self._motion_library.dof_names, preserve_order=True
        )
        self.num_body_joints = len(self.body_joint_ids)
        self.feet_body_ids, _ = self.robot.find_bodies(self.cfg.feet_body_names)
        self.num_bodies = self.robot.num_bodies
        # ASAP's "3-point" hand+head keypoint reward (see G1AsapEnvCfg.keypoint_body_names) - matched
        # defensively since exact body names vary by asset; find_bodies() raises if a single pattern
        # in the list fails to match anything, so patterns are resolved one at a time and any that
        # don't match this asset are simply skipped (rather than failing the whole env construction).
        self.keypoint_body_ids: list[int] = []
        keypoint_names: list[str] = []
        for pattern in self.cfg.keypoint_body_names:
            try:
                ids, names = self.robot.find_bodies(pattern)
            except ValueError:
                print(
                    f"[WARN] G1AsapMotionEnv: keypoint pattern '{pattern}' matched no body on the robot "
                    f"(available: {self.robot.body_names}); skipping it."
                )
                continue
            self.keypoint_body_ids.extend(ids)
            keypoint_names.extend(names)
        if self.cfg.keypoint_body_names and not self.keypoint_body_ids:
            print(
                "[WARN] G1AsapMotionEnv: none of cfg.keypoint_body_names matched a body on the robot "
                f"(available: {self.robot.body_names}); the keypoint tracking reward will contribute 0. "
                "Adjust cfg.keypoint_body_names to match your G1 asset's actual link names."
            )
        elif keypoint_names:
            print(f"[INFO] G1AsapMotionEnv: keypoint bodies for r_keypoint_pos: {keypoint_names}")

        # per-env bookkeeping
        self._motion_start_times = torch.zeros(self.num_envs, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.num_body_joints, device=self.device)
        self._last_actions = torch.zeros_like(self._actions)

        # reward-penalty curriculum (single scalar factor, see class docstring)
        self._curriculum_factor = float(self.cfg.curriculum_initial_factor)
        self._episode_len_ema = 0.0

        # observation history (backed by Isaac Lab's own CircularBuffer; each buffer's max_len already
        # includes the current frame, so `.buffer` alone gives "current + `history_length` past frames")
        self._actor_history = CircularBuffer(
            max_len=1 + self.cfg.history_length, batch_size=self.num_envs, device=self.device
        )
        self._critic_history = CircularBuffer(
            max_len=1 + self.cfg.history_length, batch_size=self.num_envs, device=self.device
        )

        # reference-state cache populated by `_refresh_reference()`, read by `_get_dones`/`_get_rewards`/
        # `_get_observations` (see those methods for the exact call order within one `env.step()`)
        self._ref_motion_phase = torch.zeros(self.num_envs, device=self.device)
        self._dof_pos_diff = torch.zeros(self.num_envs, self.num_body_joints, device=self.device)
        self._dof_vel_diff = torch.zeros_like(self._dof_pos_diff)
        self._body_pos_diff = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self._body_rot_err = torch.zeros(self.num_envs, self.num_bodies, device=self.device)
        self._body_lin_vel_diff = torch.zeros_like(self._body_pos_diff)
        self._body_ang_vel_diff = torch.zeros_like(self._body_pos_diff)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.robot_ref = Articulation(self.cfg.ref_robot_cfg)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # add articulations to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["robot_ref"] = self.robot_ref
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._last_actions = self._actions.clone()
        self._actions = actions.clone()

    def _apply_action(self) -> None:
        # non-body joints (e.g. fingers) are always held at their default pose
        joint_pos_target = self.robot.data.default_joint_pos.clone()
        joint_pos_target[:, self.body_joint_ids] = (
            self.robot.data.default_joint_pos[:, self.body_joint_ids] + self._actions * self.cfg.action_scale
        )
        self.robot.set_joint_position_target(joint_pos_target)

    # reference-motion / curriculum bookkeeping

    def _refresh_reference(self) -> None:
        """Teleport ``robot_ref`` to the current reference-motion state and cache tracking diffs.

        Must be called at least once before dones/rewards/observations are computed off of it. Cheap
        enough (one FK pass, no dynamics) to call more than once per `env.step()` (see call sites).
        """
        motion_ids = self._motion_library.motion_ids
        ref_time = self.episode_length_buf.to(torch.float32) * self.step_dt + self._motion_start_times
        times_np = ref_time.detach().cpu().numpy()
        dof_pos, dof_vel, root_pos, root_rot, root_lin_vel, root_ang_vel = self._motion_library.sample(
            motion_ids, times_np
        )

        ref_dof_pos = self.robot.data.default_joint_pos.clone()
        ref_dof_pos[:, self.body_joint_ids] = dof_pos
        ref_dof_vel = torch.zeros_like(self.robot.data.default_joint_vel)
        ref_dof_vel[:, self.body_joint_ids] = dof_vel

        root_pose = torch.cat([root_pos + self.scene.env_origins, root_rot], dim=-1)
        root_vel = torch.cat([root_lin_vel, root_ang_vel], dim=-1)
        self.robot_ref.write_root_pose_to_sim(root_pose)
        self.robot_ref.write_root_velocity_to_sim(root_vel)
        self.robot_ref.write_joint_state_to_sim(ref_dof_pos, ref_dof_vel)
        # kinematic-only articulation update (no dynamics step): recomputes robot_ref's body_pos_w/
        # body_quat_w/body_lin_vel_w/body_ang_vel_w from the joint/root state just written above
        self.sim.forward()
        self.robot_ref.update(self.physics_dt)

        motion_length = self._motion_library.get_motion_length(motion_ids)
        self._ref_motion_phase = torch.clamp(ref_time / motion_length.clamp_min(1.0e-6), 0.0, 1.0)
        self._dof_pos_diff = dof_pos - self.robot.data.joint_pos[:, self.body_joint_ids]
        self._dof_vel_diff = dof_vel - self.robot.data.joint_vel[:, self.body_joint_ids]
        self._body_pos_diff = self.robot_ref.data.body_pos_w - self.robot.data.body_pos_w
        self._body_rot_err = quat_error_magnitude(self.robot_ref.data.body_quat_w, self.robot.data.body_quat_w)
        self._body_lin_vel_diff = self.robot_ref.data.body_lin_vel_w - self.robot.data.body_lin_vel_w
        self._body_ang_vel_diff = self.robot_ref.data.body_ang_vel_w - self.robot.data.body_ang_vel_w

    def _maybe_resample_motions(self) -> None:
        """Periodically reassign every env's reference clip (matches ASAP's `resample_time_interval_s`).

        Mostly relevant once ``motion_file`` points at a directory of multiple clips; with a single
        clip this just restarts every env's tracking phase from time zero every ``motion_resample_
        interval_s`` seconds of sim time.
        """
        self._steps_since_resample += 1
        if self._steps_since_resample < self._resample_interval_steps:
            return
        self._steps_since_resample = 0
        self._motion_library.load_motions(self.robot._ALL_INDICES, random_sample=True)
        self._motion_start_times[:] = 0.0

    def _update_curriculum(self, env_ids: torch.Tensor) -> None:
        """Update the reward-penalty curriculum factor from the just-finished episodes in `env_ids`."""
        if len(env_ids) == 0:
            return
        mean_len = self.episode_length_buf[env_ids].float().mean().item()
        alpha = self.cfg.curriculum_episode_length_ema_alpha
        self._episode_len_ema = (1.0 - alpha) * self._episode_len_ema + alpha * mean_len
        target = self.cfg.curriculum_target_episode_length_frac * self.max_episode_length
        if self._episode_len_ema > target:
            self._curriculum_factor = min(1.0, self._curriculum_factor + self.cfg.curriculum_increment)
        else:
            self._curriculum_factor = max(
                self.cfg.curriculum_min_factor, self._curriculum_factor - self.cfg.curriculum_increment
            )

    # gym API

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._maybe_resample_motions()
        self._refresh_reference()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        fell = torch.zeros_like(time_out)
        motion_far = torch.zeros_like(time_out)
        if self.cfg.early_termination:
            fell = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height
            # curriculum'd: loose threshold early in training, tightens as the policy survives longer
            threshold = self.cfg.termination_motion_far_threshold_start - self._curriculum_factor * (
                self.cfg.termination_motion_far_threshold_start - self.cfg.termination_motion_far_threshold_end
            )
            motion_far = torch.amax(torch.norm(self._body_pos_diff, dim=-1), dim=-1) > threshold
            died = fell | motion_far
        # per-cause termination breakdown for diagnostics (see scripts/asap/train.py's tensorboard logging) -
        # "fell"/"motion_far" are mutually-possible-but-rare-overlap subsets of `died`, "time_out" is separate
        self.extras["termination"] = {"fell": fell, "motion_far": motion_far, "time_out": time_out}
        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        def tracking(sq_err_mean: torch.Tensor, sigma: float, weight: float) -> torch.Tensor:
            return weight * torch.exp(-sq_err_mean / sigma)

        feet_pos_diff = self._body_pos_diff[:, self.feet_body_ids]

        r_body_pos = tracking(
            self._body_pos_diff.pow(2).mean(dim=(-2, -1)), self.cfg.reward_body_pos_sigma, self.cfg.reward_body_pos_weight
        )
        r_feet_pos = tracking(
            feet_pos_diff.pow(2).mean(dim=(-2, -1)), self.cfg.reward_feet_pos_sigma, self.cfg.reward_feet_pos_weight
        )
        if self.keypoint_body_ids:
            keypoint_pos_diff = self._body_pos_diff[:, self.keypoint_body_ids]
            r_keypoint_pos = tracking(
                keypoint_pos_diff.pow(2).mean(dim=(-2, -1)),
                self.cfg.reward_keypoint_pos_sigma,
                self.cfg.reward_keypoint_pos_weight,
            )
        else:
            r_keypoint_pos = torch.zeros(self.num_envs, device=self.device)
        r_body_rot = tracking(
            self._body_rot_err.pow(2).mean(dim=-1), self.cfg.reward_body_rot_sigma, self.cfg.reward_body_rot_weight
        )
        r_body_lin_vel = tracking(
            self._body_lin_vel_diff.pow(2).mean(dim=(-2, -1)),
            self.cfg.reward_body_lin_vel_sigma,
            self.cfg.reward_body_lin_vel_weight,
        )
        r_body_ang_vel = tracking(
            self._body_ang_vel_diff.pow(2).mean(dim=(-2, -1)),
            self.cfg.reward_body_ang_vel_sigma,
            self.cfg.reward_body_ang_vel_weight,
        )
        r_joint_pos = tracking(
            self._dof_pos_diff.pow(2).mean(dim=-1), self.cfg.reward_joint_pos_sigma, self.cfg.reward_joint_pos_weight
        )
        r_joint_vel = tracking(
            self._dof_vel_diff.pow(2).mean(dim=-1), self.cfg.reward_joint_vel_sigma, self.cfg.reward_joint_vel_weight
        )

        # body-joint-only slices of the robot's torque/limit tensors (only these are policy-actuated;
        # non-body joints, e.g. fingers, are always held at their default pose - see `_apply_action`)
        applied_torque = self.robot.data.applied_torque[:, self.body_joint_ids]
        dof_pos = self.robot.data.joint_pos[:, self.body_joint_ids]
        dof_vel = self.robot.data.joint_vel[:, self.body_joint_ids]
        soft_pos_lower = self.robot.data.soft_joint_pos_limits[:, self.body_joint_ids, 0]
        soft_pos_upper = self.robot.data.soft_joint_pos_limits[:, self.body_joint_ids, 1]
        vel_limit = self.robot.data.joint_vel_limits[:, self.body_joint_ids]
        torque_limit = self.robot.data.joint_effort_limits[:, self.body_joint_ids]

        penalty_torque = self.cfg.reward_torque_weight * applied_torque.pow(2).sum(dim=-1)
        dof_pos_violation = (soft_pos_lower - dof_pos).clamp(min=0.0) + (dof_pos - soft_pos_upper).clamp(min=0.0)
        penalty_dof_pos_limit = self.cfg.reward_dof_pos_limit_weight * dof_pos_violation.sum(dim=-1)
        dof_vel_violation = (dof_vel.abs() - vel_limit * self.cfg.dof_vel_limit_margin).clamp(min=0.0)
        penalty_dof_vel_limit = self.cfg.reward_dof_vel_limit_weight * dof_vel_violation.sum(dim=-1)
        torque_violation = (applied_torque.abs() - torque_limit * self.cfg.torque_limit_margin).clamp(min=0.0)
        penalty_torque_limit = self.cfg.reward_torque_limit_weight * torque_violation.sum(dim=-1)
        penalty_action_rate = self.cfg.reward_action_rate_weight * (self._actions - self._last_actions).pow(2).sum(
            dim=-1
        )

        # reward-penalty curriculum: all the above (regularization, not tracking) penalties are scaled
        # together by the single curriculum factor - see `_update_curriculum`
        penalties = self._curriculum_factor * (
            penalty_torque + penalty_dof_pos_limit + penalty_dof_vel_limit + penalty_torque_limit + penalty_action_rate
        )
        # termination is safety-critical: always fully applied, not curriculum-scaled
        penalty_termination = self.cfg.reward_termination_weight * self.reset_terminated.float()

        return (
            r_body_pos
            + r_feet_pos
            + r_keypoint_pos
            + r_body_rot
            + r_body_lin_vel
            + r_body_ang_vel
            + r_joint_pos
            + r_joint_vel
            + penalties
            + penalty_termination
        )

    def _get_observations(self) -> dict:
        # recompute against the (possibly just-reset) current state so this reflects what the NEXT
        # action should be conditioned on, not the pre-reset transition `_get_dones`/`_get_rewards` used
        self._refresh_reference()

        actor_feat = torch.cat(
            (
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self.robot.data.joint_pos[:, self.body_joint_ids] - self.robot.data.default_joint_pos[:, self.body_joint_ids],
                self.robot.data.joint_vel[:, self.body_joint_ids],
                self._actions,
                self._ref_motion_phase.unsqueeze(-1),
            ),
            dim=-1,
        )
        critic_feat = torch.cat(
            (
                self.robot.data.root_lin_vel_b,
                actor_feat,
                self._body_pos_diff.reshape(self.num_envs, -1),
                self._dof_pos_diff,
            ),
            dim=-1,
        )

        self._actor_history.append(actor_feat)
        self._critic_history.append(critic_feat)
        actor_obs = self._actor_history.buffer.reshape(self.num_envs, -1)
        critic_obs = self._critic_history.buffer.reshape(self.num_envs, -1)

        self.extras["observations"] = {"critic": critic_obs}
        return {"policy": actor_obs}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self._update_curriculum(env_ids)

        self.robot.reset(env_ids)
        self.robot_ref.reset(env_ids)
        super()._reset_idx(env_ids)

        self._motion_library.load_motions(env_ids, random_sample=not self.cfg.deterministic_motion_assignment)
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

        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._actor_history.reset(env_ids)
        self._critic_history.reset(env_ids)

    # reset strategies (same semantics as G1AmpEnv, generalized to per-env motion clip assignment)

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self._motion_start_times[env_ids] = 0.0
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_ids = self._motion_library.motion_ids[env_ids]
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_library.sample_times(motion_ids)
        dof_positions, dof_velocities, root_positions, root_rotations, root_linear_velocities, root_angular_velocities = (
            self._motion_library.sample(motion_ids, times)
        )

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = root_positions + self.scene.env_origins[env_ids]
        root_state[:, 2] += self.cfg.reset_root_height_offset  # avoid ground penetration on reset
        root_state[:, 3:7] = root_rotations
        root_state[:, 7:10] = root_linear_velocities
        root_state[:, 10:13] = root_angular_velocities
        # non-body joints (e.g. fingers) are held at their default pose
        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_vel = self.robot.data.default_joint_vel[env_ids].clone()
        dof_pos[:, self.body_joint_ids] = dof_positions
        dof_vel[:, self.body_joint_ids] = dof_velocities

        self._motion_start_times[env_ids] = torch.tensor(times, dtype=torch.float32, device=self.device)
        return root_state, dof_pos, dof_vel
