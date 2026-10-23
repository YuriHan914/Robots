# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim-side raw-DDS bridge for the G1 web controller.

Loads the same G1 robot/USD used by the ASAP motion-tracking task (``g1_isaac_asap_env.py`` /
``g1_isaac_amp_env.py`` both spawn ``G1_LOCAL_USD_PATH`` via ``G1AmpEnvCfg``/``G1AsapEnvCfg``) into
Isaac Sim, then exposes it to an external web controller (``unitree_g1_web_controller_complete.py``)
over raw DDS (Eclipse CycloneDDS, no ROS 2) *instead of* Isaac Sim's own in-viewport teleop extension
- the same transport family Unitree's own ``unitree_sdk2py`` SDK uses against a real G1. This script
owns the simulation loop; the web controller is a thin browser<->DDS relay with no physics/inference
logic of its own (see that script's docstring).

Three control modes, switched by DDS command (see the topic table below):

* ``stand``  - hold the default pose.
* ``manual`` - joystick ``cmd_vel`` (vx, vy, yaw rate) drives the unitree_rl_lab velocity-tracking
  policy given by ``--loco_policy_dir`` (rsl_rl ``exported/policy.pt`` + ``params/deploy.yaml``), after a
  staged blend from the current pose into the policy's own stance - see
  :class:`_LocomotionPolicyController`. Releasing the joystick keeps the policy running at zero
  velocity.
* ``goto``   - given a target ``(x, y)`` and a gait (``walk``/``run``), steers the same locomotion
  policy toward the point (see ``_goto_velocity_command``) and keeps it stepping in place on arrival.
* ``policy`` - runs the skrl-trained AMP dance policy (``scripts/skrl/train.py`` checkpoint, task
  ``Template-G1-Isaac-AMP-Dance-Direct-v0``) via the environment's own ``step()``, the same inference
  path as ``scripts/skrl/play.py``.

A forward-facing RGB camera (``isaaclab.sensors.Camera``) is mounted on the robot after the env is
created (never part of the registered task's own cfg, so training/eval scripts using the same task
are unaffected) and streamed to the web UI's CAM FEED panel as JPEG frames over DDS - see
``_attach_head_camera``/``_encode_camera_frame``. This forces ``--enable_cameras`` on and adds real
RTX-rendering overhead to every step (see ``AppLauncher``'s own docs) - pass ``--no_camera`` to skip
mounting it entirely if that's an issue.

The environment's own automatic reset-on-fall/timeout is disabled (``early_termination=False`` and
``episode_length_buf`` is zeroed every step) - a fallen robot just stays fallen in sim instead of
snapping back, and the *only* way to reset is the web UI's home/reset buttons (``g1/home_position`` /
``g1/reset_sim`` below), which call ``env.reset()`` explicitly.

DDS topics (all message types defined in ``g1_dds_types.py``; commands are web -> sim, telemetry is
sim -> web):

=====================  ==================  ==========================================
Topic                  Type                Payload
=====================  ==================  ==========================================
g1/cmd_vel             g1.CmdVel           manual joystick (linear_x, linear_y, angular_z)
g1/goto_command        g1.GotoCommand      x, y (m), gait ("walk"|"run"), speed (m/s, 0 = gait default)
g1/goto_cancel         g1.Trigger          cancel an in-progress goto
g1/policy_command      g1.PolicyCommand    command: "play" / "stop"
g1/home_position       g1.Trigger          explicit reset (only way to recover from a fall)
g1/reset_sim           g1.Trigger          same as home_position, triggered by the "초기화" button
g1/stand_command       g1.Trigger          back to the standing controller (walking policy: stop, then blend)
g1/emergency_stop      g1.Trigger          drop to ``stand`` immediately (does NOT reset the sim)
g1/robot_pose          g1.RobotPose        world pose of the robot base
g1/joint_states        g1.JointState       body-joint positions/velocities
g1/status              g1.Status           JSON ``{"mode", "gait", "target", ...}``
g1/camera_frame        g1.CameraFrame      JPEG frame from the head camera (base64), see below
g1/lidar_scan          g1.LidarScan        360 deg horizontal sweep from the head lidar (meters)
=====================  ==================  ==========================================

A walled room + a few static obstacles are spawned around the robot (see ``_spawn_room``) purely so
the lidar has real geometry to scan - the registered task's own scene is otherwise just a flat
ground plane. A separate, non-conda process (``ros2_slam_bridge.py``, see its own module docstring)
turns ``g1/lidar_scan`` + ``g1/robot_pose`` into ROS 2 topics for ``slam_toolbox`` and republishes
the resulting map back onto this same DDS bus as ``g1.OccupancyMap`` - this script has no ROS 2
dependency of its own. Pass ``--no_lidar`` to skip both the room and the lidar.

Requires the ``cyclonedds`` and ``pillow`` Python packages (``pip install cyclonedds pillow``) in the
same Python environment Isaac Lab runs under - this project's own ``isaac`` conda env (Isaac Lab is pip-installed
into it, so plain ``python`` runs this script directly - no ``isaaclab.sh`` involved, see the repo's
top-level README). No ROS 2 install of any kind is needed - deliberately so: ROS 2's own Python
bindings (rclpy) ship as a C extension tied to one specific system Python (e.g. apt's ROS 2 Jazzy on
Ubuntu 24.04 targets Python 3.12), which is binary-incompatible with a conda env pinned to a
different Python for Isaac Lab (this project's ``isaac`` env). ``cyclonedds``'s PyPI wheels are built
per-Python-version, so installing it straight into ``isaac`` just works. Both this script and the web
controller must run on the same DDS domain id (``--dds_domain_id``, default 0).

.. code-block:: bash

    conda activate isaac
    pip install cyclonedds   # once

    # Terminal 1: this script (loads the robot, owns the physics loop)
    python controller/g1_isaac_sim_bridge.py

    # Terminal 2: the web relay (see unitree_g1_web_controller_complete.py)
    python controller/unitree_g1_web_controller_complete.py
    # -> http://localhost:5000
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Raw-DDS bridge exposing a live G1 to the web controller.")
parser.add_argument(
    "--task",
    type=str,
    default="Template-G1-Isaac-AMP-Dance-Direct-v0",
    help="Registered gym task providing the G1 robot and the skrl AMP dance policy.",
)
parser.add_argument(
    "--policy_checkpoint",
    type=str,
    default=None,
    help="Path to a skrl AMP checkpoint (.pt). Defaults to the latest run under logs/skrl/g1_amp_dance.",
)
_DEFAULT_STAND_POLICY_CHECKPOINT = (
    "/mnt/data/github/Robots/g1_isaac/logs/skrl/g1_stand/2026-08-13_02-36-26_ppo_torch/checkpoints/best_agent.pt"
)
parser.add_argument(
    "--stand_policy_checkpoint",
    type=str,
    default=_DEFAULT_STAND_POLICY_CHECKPOINT,
    help="Path to a G1-PPO-Direct-Stand-v0 PPO checkpoint (.pt). When set (the default - see "
    "_DEFAULT_STAND_POLICY_CHECKPOINT), 'policy' mode (g1/policy_command 'play', which now also runs "
    "automatically on load - see SharedState) runs this live standing policy instead of the AMP dance "
    "one - see _StandPolicyController's own docstring for how it's run against this same env/robot. "
    "Pass an empty string to fall back to the old AMP-dance-by-default behavior.",
)
_DEFAULT_LOCO_POLICY_DIR = (
    "/mnt/data/github/Robots/unitree_g1_rl_lab/logs/rsl_rl/finetune_yaw/2026-10-22_18-14-44/exported"
)
parser.add_argument(
    "--loco_policy_dir",
    type=str,
    default=_DEFAULT_LOCO_POLICY_DIR,
    help="unitree_rl_lab rsl_rl 'exported' dir (policy.pt; deploy.yaml read from here or ../params/). "
    "Required: joystick ('manual') and go-to-point ('goto') driving both run this velocity-tracking "
    "policy - see _LocomotionPolicyController.",
)
parser.add_argument(
    "--loco_transition_time",
    type=float,
    default=0.3,
    help="Seconds to blend joint targets/gains from the current controller's into the locomotion policy's "
    "(policy already in the loop - see _LocomotionPolicyController). Verified in sim from the Stand PPO "
    "stance: 0.3s settles upright with the least drift, 0.6-1.0s still hold but drift ~0.5m while "
    "recovering, 2.0s falls (too long with only partial policy authority).",
)
parser.add_argument(
    "--loco_handover_time",
    type=float,
    default=0.5,
    help="Seconds to blend from the locomotion policy back to the Stand PPO policy on the web UI's Stand button "
    "(after the robot has stopped). Verified in sim from forward/turning/strafing/spinning: 0.3-0.5s hold, "
    "0.5s with the least tilt; a direct switch tilts ~30 deg, 1.0s+ falls.",
)
parser.add_argument(
    "--loco_warmup_time",
    type=float,
    default=1.0,
    help="Seconds the locomotion policy runs at zero velocity before the joystick command is applied.",
)
parser.add_argument(
    "--dds_domain_id", type=int, default=0, help="CycloneDDS domain id (must match the web controller)."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--no_camera", action="store_true", default=False, help="Skip mounting the head camera (saves RTX render overhead)."
)
parser.add_argument(
    "--no_lidar",
    action="store_true",
    default=False,
    help="Skip mounting the head lidar and spawning the walled room it scans (see _spawn_room/_attach_head_lidar).",
)
parser.add_argument(
    "--camera_width", type=int, default=640, help="Head camera frame width (px). Higher = sharper but more RTX render + DDS bandwidth cost."
)
parser.add_argument(
    "--camera_height", type=int, default=480, help="Head camera frame height (px). Keep the same 4:3 ratio as width unless you also change the sensor's aperture."
)
parser.add_argument(
    "--camera_jpeg_quality",
    type=int,
    default=88,
    help="JPEG encode quality (1-100) for camera_frame DDS frames. Higher = fewer compression artifacts but bigger frames.",
)
parser.add_argument(
    "--ground_clearance",
    type=float,
    default=0.003,
    help=(
        "Target gap (m) between the lowest foot-sole point and the ground plane, enforced by "
        "measuring the robot's *actual* post-teleport foot height (forward kinematics) after every "
        "reset and shifting the root Z to match - see _align_feet_to_ground(). Exact by "
        "construction, regardless of joint pose; no more guessing a fixed spawn-height margin."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# checked before AppLauncher below so a missing policy fails fast instead of after Isaac Sim boots
if not args_cli.loco_policy_dir:
    parser.error("--loco_policy_dir is required (joystick/go-to-point have no other controller)")
# the head camera needs RTX rendering enabled even in headless mode - see AppLauncher's own
# --enable_cameras flag; harmless (a no-op) if --no_camera is also passed.
if not args_cli.no_camera:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import base64
import builtins
import io
import json
import math
import os
import random
import threading
import time

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from g1_dds_types import (
    TOPIC_CAMERA_FRAME,
    TOPIC_CMD_VEL,
    TOPIC_EMERGENCY_STOP,
    TOPIC_STAND_COMMAND,
    TOPIC_GOTO_CANCEL,
    TOPIC_GOTO_COMMAND,
    TOPIC_HOME_POSITION,
    TOPIC_JOINT_STATES,
    TOPIC_LIDAR_SCAN,
    TOPIC_POLICY_COMMAND,
    TOPIC_RESET_SIM,
    TOPIC_ROBOT_POSE,
    TOPIC_STATUS,
    CameraFrame,
    CmdVel,
    GotoCommand,
    JointState,
    LidarScan,
    PolicyCommand,
    RobotPose,
    Status,
    Trigger,
    dds_listener,
    make_participant,
)
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg, patterns
from isaaclab.utils.math import quat_apply
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from skrl.utils.runner.torch import Runner

import g1_isaac.tasks  # noqa: F401

AGENT_CFG_ENTRY_POINT = "skrl_amp_cfg_entry_point"

# ---------------------------------------------------------------------------
# Trained-standing pose/gain snapshot, applied to this env's own robot at startup (see
# _apply_trained_stand_defaults below) so "stand" mode (a zero action, i.e. a PD hold of the default pose) and every
# env.reset() hold *this* pose/gain rather than isaaclab_assets' untrained default.
#
# Source: a converged G1-PPO-Direct-Stand-v0 policy (checkpoint logs/skrl/g1_stand/
# 2026-08-13_02-36-26_ppo_torch/checkpoints/best_agent.pt - see docs/G1-PPO-Direct-Stand-v0.md),
# played back with `python scripts/skrl/play.py --task G1-PPO-Direct-Stand-v0 --algorithm PPO
# --num_envs 3 --checkpoint <that checkpoint> --record_pose_gain logs/pose_gain.csv`. env0's and
# env2's full first episodes (5999 steps each, no fall - both reached the 60s timeout) landed in
# logs/pose_gain_env0.csv / logs/pose_gain_env2.csv.
#
# Both pose and gain below are the plain per-column mean of both files pooled together - *not*
# symmetrized (an earlier version of this snapshot forced left/right pose into g1_isaac_env.py's
# _LEG_JOINT_MIRROR_PAIRS convention; reverted). env0 and env2 both independently settled on a
# visibly asymmetric stance (e.g. env0 left_knee 0.665 vs right_knee 1.033 rad, env2 0.629 vs 0.991 -
# same direction, right leg consistently more flexed than left in both) - since that's the actual,
# twice-verified-to-hold-60s stance rather than a guess, it's kept as-is: the symmetrized version was
# never played back and confirmed, and for hip_roll/hip_yaw specifically it would have meant flipping
# a sign relative to what was actually observed (raw data has *both* sides negative there, e.g. hip_roll
# left=-0.172/right=-0.253 - not just unequal magnitude but the same sign on both sides, which
# _LEG_JOINT_MIRROR_PAIRS's convention says a symmetric stance shouldn't have).
# ---------------------------------------------------------------------------
_TRAINED_STAND_POSE = {
    "left_hip_pitch_joint": -0.353413,
    "right_hip_pitch_joint": -0.514801,
    "left_hip_roll_joint": -0.172107,
    "right_hip_roll_joint": -0.253025,
    "left_hip_yaw_joint": 0.016697,
    "right_hip_yaw_joint": 0.076916,
    "left_knee_joint": 0.647028,
    "right_knee_joint": 1.011715,
    "left_ankle_pitch_joint": -0.315327,
    "right_ankle_pitch_joint": -0.524862,
    "left_ankle_roll_joint": -0.038432,
    "right_ankle_roll_joint": 0.046443,
}
# category -> (actuator group, joint names, stiffness, damping) - same left/right-tied-to-one-category
# grouping as g1_isaac_env.py's _GAIN_SEARCH_CATEGORIES (see that file's own comment for why). The
# actuator group matters for *how* each category gets written - see _apply_trained_stand_defaults.
_TRAINED_STAND_GAIN = {
    "hip_yaw": ("legs", ("left_hip_yaw_joint", "right_hip_yaw_joint"), 16.755824, 5.096802),
    "hip_roll": ("legs", ("left_hip_roll_joint", "right_hip_roll_joint"), 18.874950, 14.990407),
    "hip_pitch": ("legs", ("left_hip_pitch_joint", "right_hip_pitch_joint"), 16.680559, 3.773930),
    "knee": ("legs", ("left_knee_joint", "right_knee_joint"), 33.468639, 9.421704),
    "ankle_pitch": ("feet", ("left_ankle_pitch_joint", "right_ankle_pitch_joint"), 14.802975, 0.175884),
    "ankle_roll": ("feet", ("left_ankle_roll_joint", "right_ankle_roll_joint"), 6.728400, 0.050328),
    "waist_yaw": ("waist", ("waist_yaw_joint",), 2968.899885, 3.138571),
    "waist_roll": ("waist", ("waist_roll_joint",), 2773.845506, 4.099625),
    "waist_pitch": ("waist", ("waist_pitch_joint",), 837.902333, 1.861697),
}


def _apply_trained_stand_defaults(robot) -> None:
    """Overwrites this robot's default_joint_pos (legs/feet - _TRAINED_STAND_POSE) and live
    stiffness/damping (legs/feet/waist - _TRAINED_STAND_GAIN) with the trained snapshot above.

    Gain must be written *differently* depending on the target's actuator type, not through one
    uniform API - this env's robot_cfg (G1AmpEnvCfg/G1AsapEnvCfg, see module docstring) gives legs/feet
    isaaclab_assets' stock DCMotorCfg and waist its stock ImplicitActuatorCfg (unlike the Stand task's
    _WAIST_ACTUATOR_DCMOTOR conversion - see g1_isaac_env_stand_cfg.py's comment on why that exists):

    - legs/feet (DCMotorCfg): IdealPDActuator.compute() computes torque from `self.stiffness` /
      `self.damping` - tensors cached on the actuator *object itself* - every step, and never reads
      anything from the PhysX DOF level. Must be written directly into those tensors (same mechanism
      g1_isaac_env.py's _apply_action uses for its own live RL gain search).
    - waist (ImplicitActuatorCfg): the opposite - it performs no computation of its own and reads
      nothing from its own tensors; PD is handled by PhysX's native joint drive, only reachable via
      Articulation.write_joint_stiffness_to_sim/write_joint_damping_to_sim.

    Using the wrong one of these two for a given actuator is silently a no-op (no error, the gain just
    never actually changes) - this function got exactly that wrong for legs/feet on the first pass
    (used write_joint_stiffness_to_sim for everything), which meant legs/feet kept isaaclab_assets'
    original, much stiffer gains (stiffness 100/100/100/200 vs the ~17-34 trained here) while their
    *pose* target jumped to the trained (far more crouched) stance anyway - a stiff, fast snap into an
    unfamiliar deep-knee pose, which is exactly the kind of mismatch that makes a robot fall over.

    write_joint_stiffness_to_sim/write_joint_damping_to_sim need a GPU->CPU sync per call, too costly
    to pay every control step across thousands of parallel envs during RL training, but this runs once
    at startup on a single robot (num_envs=1 - see main()), so the cost is a non-issue here.
    """
    for name, value in _TRAINED_STAND_POSE.items():
        joint_ids, _ = robot.find_joints([name])
        robot.data.default_joint_pos[:, joint_ids[0]] = value
    for actuator_group, joint_names, stiffness, damping in _TRAINED_STAND_GAIN.values():
        if actuator_group == "waist":
            joint_ids, _ = robot.find_joints(list(joint_names))
            robot.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids)
            robot.write_joint_damping_to_sim(damping, joint_ids=joint_ids)
        else:
            actuator = robot.actuators[actuator_group]
            local_ids = torch.tensor(
                [actuator.joint_names.index(n) for n in joint_names], device=robot.device
            )
            actuator.stiffness[:, local_ids] = stiffness
            actuator.damping[:, local_ids] = damping


class _StandPolicyController:
    """Runs a trained G1-PPO-Direct-Stand-v0 PPO policy live against *this* script's own robot (the
    one G1AmpEnvCfg/G1AsapEnvCfg spawned - see module docstring), each control step, instead of the
    static _apply_trained_stand_defaults snapshot above.

    A second, separate gym.make("G1-PPO-Direct-Stand-v0", ...) isn't an option here: both that task's
    robot_cfg and this script's env spawn at the exact same prim path ("/World/envs/env_.*/Robot"), so
    a second env instance would collide with the one already running. Instead, this builds just the
    trained policy's own Gaussian model directly via skrl's model-instantiator utility
    (skrl.utils.model_instantiators.torch.gaussian_model) - bypassing skrl's Runner/PPO agent classes
    entirely, since those need a live env to infer observation/action shapes from and register the
    policy against; a plain gymnasium.spaces.Box built by hand supplies the same (48,)/(30,) shape
    info instead. Also loads the matching RunningStandardScaler observation preprocessor from the same
    checkpoint (skrl_ppo_stand_cfg.yaml's state_preprocessor) - skipping this would feed the network
    raw, unnormalized observations it was never trained on and it would behave close to randomly.

    Pose + physics stepping reuse the AMP/ASAP env's own env.step() (same decimation/telemetry/
    episode-reset-prevention pipeline main()'s loop already runs for every other mode): step() below
    only computes what 29-dim AMP-shaped action would reproduce the Stand policy's own 12-joint pose
    target under the AMP env's *own* action_scale (0.5, vs Stand's 0.25), leaving the other 17 body
    joints (waist/arms) at zero offset (their own default pose, untouched). Gain (the other half of
    the Stand policy's 30-dim action) has no equivalent in the AMP env's action space at all, so it's
    written directly into the actuators every step instead - see step()'s own comment for why
    legs/feet and waist need different write paths (same reason as _apply_trained_stand_defaults).

    Crucially, all of this policy's own pose/deviation math is anchored to the *original*
    isaaclab_assets default pose for the 12 Stand joints - passed in by the caller
    (original_default_joint_pos, see __init__) rather than read live off
    robot.data.default_joint_pos, since main() may have already overwritten that via
    _apply_trained_stand_defaults for the static "stand" mode's own use. Using the mutated tensor here
    would silently shift what this policy treats as "neutral" away from what it was actually trained
    against.
    """

    def __init__(
        self,
        robot,
        checkpoint_path: str,
        device: str,
        amp_body_joint_ids: list[int],
        original_default_joint_pos: torch.Tensor,
        amp_action_scale: float,
    ):
        """``original_default_joint_pos``: a clone of robot.data.default_joint_pos taken *before*
        _apply_trained_stand_defaults (or anything else) had a chance to overwrite it - must be
        captured by the caller (see main()) right after the robot is created, since by the time this
        constructor might run (e.g. lazily, on first switch into 'policy' mode) that tensor may already
        hold the static-snapshot values instead. See class docstring for why this distinction matters.
        """
        import gymnasium.spaces as spaces
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        from skrl.utils.model_instantiators.torch import shared_model

        from g1_isaac.tasks.direct.g1_isaac.g1_isaac_env import _BODY_JOINT_PATTERNS, _GAIN_SEARCH_CATEGORIES
        from g1_isaac.tasks.direct.g1_isaac.g1_isaac_env_stand_cfg import G1IsaacEnvCfg

        stand_cfg = G1IsaacEnvCfg()
        self.robot = robot
        self.device = device
        self.action_scale = stand_cfg.action_scale  # 0.25 - Stand's own, not the AMP env's 0.5
        self.gain_scale_max = stand_cfg.gain_scale_max
        # this env's own action_scale (0.5 for both G1AmpEnvCfg/G1AsapEnvCfg today, but read from the
        # live cfg rather than hardcoded so this can't silently drift out of sync if either changes) -
        # see step()'s own use.
        self.amp_action_scale = amp_action_scale

        # the 12 legs/feet joints Stand's own policy positions directly - same pattern list that file
        # itself resolves against, imported rather than retyped so this can never silently drift out
        # of sync with what the checkpoint was actually trained on.
        self.pose_joint_ids, _ = robot.find_joints(_BODY_JOINT_PATTERNS, preserve_order=True)
        self.n_pose = len(self.pose_joint_ids)
        # frozen, *un*-mutated reference pose - see constructor docstring above.
        self.pose_default = original_default_joint_pos[:, self.pose_joint_ids].clone()
        # where each of those 12 joints sits within the AMP env's own 29-joint action ordering, so
        # step() can place each pose value at the right index of the 29-dim action it hands to
        # env.step() - see step()'s own comment.
        self.amp_action_index = [amp_body_joint_ids.index(j) for j in self.pose_joint_ids]
        self.n_amp_joints = len(amp_body_joint_ids)  # 29 for both AMP/ASAP - see step()'s own use

        self.gain_categories = _GAIN_SEARCH_CATEGORIES
        self.n_gain_cat = len(self.gain_categories)
        # per-category (actuator group, write mode, ids, default stiffness, default damping) - same
        # DCMotor-direct-tensor vs. waist-ImplicitActuator-via-write_joint_*_to_sim split as
        # _apply_trained_stand_defaults (see that function's docstring for *why* two different write
        # paths are required - using the wrong one for a given actuator is a silent no-op).
        self._gain_targets = []
        for _cat_name, actuator_name, joint_names in self.gain_categories:
            actuator = robot.actuators[actuator_name]
            if actuator_name == "waist":
                write_ids, _ = robot.find_joints(list(joint_names))
                local_id_for_default = actuator.joint_names.index(joint_names[0])
            else:
                write_ids = torch.tensor(
                    [actuator.joint_names.index(n) for n in joint_names], device=device
                )
                local_id_for_default = write_ids[0]
            default_stiffness = float(actuator.stiffness[0, local_id_for_default].item())
            default_damping = float(actuator.damping[0, local_id_for_default].item())
            self._gain_targets.append(
                (actuator_name, actuator, write_ids, default_stiffness, default_damping)
            )

        # gain action from the *previous* step, in [-1, 1] - part of the Stand policy's own
        # observation (see g1_isaac_env.py's _get_observations: it conditions on what gains it's
        # currently running, not just the resulting dynamics), zeroed here to match that env's own
        # reset-time value (isaaclab_assets default gains, gain_action=0).
        self.prev_gain_action = torch.zeros(1, 2 * self.n_gain_cat, device=device)

        observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.n_pose * 2 + 6 + 2 * self.n_gain_cat,))
        action_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.n_pose + 2 * self.n_gain_cat,))
        net = [{"name": "net", "input": "OBSERVATIONS", "layers": [256, 128, 128], "activations": "elu"}]
        print("[g1_isaac_sim_bridge] Building Stand policy model ...")
        # skrl_ppo_stand_cfg.yaml sets models.separate: False - policy and value SHARE one trunk
        # (single_forward_pass) with two output heads, not two independent networks. Must be built with
        # shared_model (matching skrl.utils.runner.torch.Runner._generate_models's own "shared models"
        # branch exactly), not gaussian_model alone - the checkpoint's "policy" key holds this whole
        # shared object's state_dict (net_container.* trunk + both policy_layer.* and value_layer.*
        # heads, since PPO sets self.policy is self.value when separate=False - see its __init__), so a
        # policy-only GaussianModel's state_dict shape can never match it (this is what the first
        # version of this class got wrong: "Missing key(s) ... net_container.6 ... Unexpected key(s)
        # ... policy_layer/value_layer").
        self.policy = shared_model(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            structure=["GaussianMixin", "DeterministicMixin"],
            roles=["policy", "value"],
            parameters=[
                {
                    "clip_actions": False,
                    "clip_log_std": True,
                    "min_log_std": -20.0,
                    "max_log_std": 2.0,
                    "initial_log_std": -1.0,
                    "network": net,
                    "output": "ACTIONS",
                },
                {"clip_actions": False, "network": net, "output": "ONE"},
            ],
        )
        # materialize the instantiator's lazy modules before load_state_dict - see Model.init_state_dict's
        # own docstring ("always before performing any operation on model parameters"), same call
        # Runner._generate_models makes for every model role it builds.
        dummy_obs = torch.zeros(1, observation_space.shape[0], device=device)
        # both "observations" and "states" keys: the generated network code reads from whichever
        # token the yaml's network.input names ("OBSERVATIONS" here -> inputs["observations"]) -
        # passing only "states" leaves that lookup as None and crashes the lazy Linear layer's shape
        # inference (AttributeError: 'NoneType' object has no attribute 'shape') - verified directly
        # against skrl's own Model.init_state_dict, which supplies both keys in its default fallback.
        dummy_inputs = {"observations": dummy_obs, "states": dummy_obs}
        self.policy.init_state_dict(inputs=dummy_inputs, role="policy")
        self.policy.init_state_dict(inputs=dummy_inputs, role="value")

        print(f"[g1_isaac_sim_bridge] Loading Stand policy checkpoint: {checkpoint_path}")
        checkpoint = torch.load(os.path.abspath(checkpoint_path), map_location=device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy.eval()
        self.preprocessor = RunningStandardScaler(size=observation_space, device=device)
        if "observation_preprocessor" in checkpoint:
            self.preprocessor.load_state_dict(checkpoint["observation_preprocessor"])

    def _build_observation(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self.pose_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.pose_joint_ids]
        obs = torch.cat(
            (
                joint_pos - self.pose_default,
                joint_vel,
                self.robot.data.projected_gravity_b,
                self.robot.data.root_ang_vel_b,
                self.prev_gain_action,
            ),
            dim=-1,
        )
        return self.preprocessor(obs, train=False)

    def step(self) -> torch.Tensor:
        """Runs one control step of the Stand policy and returns a 29-dim action shaped for the AMP/
        ASAP env's own action space, ready to hand straight to env.step() - see class docstring for
        why reusing that (rather than stepping physics by hand here) is both correct and simpler.
        Gain is written directly into the actuators as a side effect (also see class docstring).
        """
        obs = self._build_observation()
        with torch.inference_mode():
            _, outputs = self.policy.act({"observations": obs, "states": obs}, role="policy")
        action = outputs["mean_actions"]  # deterministic (eval-mode) action, same as scripts/skrl/play.py

        pose_action = action[:, : self.n_pose]
        gain_action = torch.clamp(action[:, self.n_pose :], -1.0, 1.0)
        self.prev_gain_action = gain_action

        stiffness_scale = self.gain_scale_max ** gain_action[:, : self.n_gain_cat]
        damping_scale = self.gain_scale_max ** gain_action[:, self.n_gain_cat :]
        for cat_idx, (actuator_name, actuator, write_ids, default_stiffness, default_damping) in enumerate(
            self._gain_targets
        ):
            new_stiffness = default_stiffness * float(stiffness_scale[0, cat_idx].item())
            new_damping = default_damping * float(damping_scale[0, cat_idx].item())
            if actuator_name == "waist":
                self.robot.write_joint_stiffness_to_sim(new_stiffness, joint_ids=write_ids)
                self.robot.write_joint_damping_to_sim(new_damping, joint_ids=write_ids)
            else:
                actuator.stiffness[:, write_ids] = new_stiffness
                actuator.damping[:, write_ids] = new_damping

        # place each Stand pose value at its own joint's position in the AMP env's 29-dim action,
        # rescaled so that env's own _apply_action (target = default + action * amp_action_scale)
        # reproduces exactly Stand's own (target = pose_default + pose_action * stand_action_scale) -
        # every other position (waist/arms) stays 0, holding those joints' own default pose.
        amp_action = torch.zeros(1, self.n_amp_joints, device=self.device)
        amp_action[0, self.amp_action_index] = pose_action[0] * (self.action_scale / self.amp_action_scale)
        return amp_action

    def reset(self) -> None:
        """Zero prev_gain_action back to its reset-time value (see __init__'s own comment on why it
        starts at zero). env.reset() teleports the robot back to its default pose but has no idea
        this controller object exists, so without this the *next* step() after a reset would feed
        the policy a stale pre-reset gain action - call this whenever the env is reset (home/reset
        button, see main()'s do_reset handling)."""
        self.prev_gain_action = torch.zeros(1, 2 * self.n_gain_cat, device=self.device)

    def restore_default_gains(self) -> None:
        """Write each gain-searched actuator's stiffness/damping back to its own cached
        ``default_stiffness``/``default_damping`` (captured in ``__init__`` before this policy ever
        touched them).

        ``step()`` above overwrites legs/feet/waist stiffness+damping live, every step, as one half of
        the policy's own action - and nothing else in this process ever puts them back. Isaac Lab's PD
        actuators apply essentially zero torque at zero (or very low) gain, so leaving mode="policy"
        with a low gain still latched in means the legs go completely unresponsive to *any* later
        position command (the "stand" mode's default-pose hold included) even though the command itself is
        fine - the actuator just isn't allowed to act on it. Call this on every transition *out* of
        "policy" mode (see main()'s own mode-change handling) so manual/goto/stand always start from a
        known-good gain, the same way ``reset()`` above already restores ``prev_gain_action``.
        """
        for actuator_name, actuator, write_ids, default_stiffness, default_damping in self._gain_targets:
            if actuator_name == "waist":
                self.robot.write_joint_stiffness_to_sim(default_stiffness, joint_ids=write_ids)
                self.robot.write_joint_damping_to_sim(default_damping, joint_ids=write_ids)
            else:
                actuator.stiffness[:, write_ids] = default_stiffness
                actuator.damping[:, write_ids] = default_damping


# Unitree SDK motor order for the 29-DOF G1 - deploy.yaml's ``joint_ids_map``/``stiffness``/``damping``
# are expressed against this list (same as unitree_rl_lab's UNITREE_G1_29DOF_CFG.joint_sdk_names).
_G1_SDK_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip
# Full-deflection values the web UI's joystick sends on g1/cmd_vel (g1_web_ui_mobile_first.html's
# movePad/rotPad: linear_x = -dy * 2.0, linear_y = dx * 1.0, angular_z = dx * 1.0) - used to normalize
# them back to [-1, 1] before scaling into the locomotion policy's own trained command range.
_JOYSTICK_FULL_SCALE = (2.0, 1.0, 1.0)


class _LocomotionPolicyController:
    """Runs a unitree_rl_lab velocity-tracking policy (rsl_rl ``exported/policy.pt`` + the run's own
    ``params/deploy.yaml``) against this script's robot for joystick ('manual') driving.

    The policy was trained on a different env than the one this script runs (unitree_rl_lab's
    ManagerBasedRLEnv: 50 Hz control, implicit PD with deploy.yaml's gains) - so, as with
    _StandPolicyController, nothing about the AMP env's own obs/action pipeline is reused:

    - observation: rebuilt here term by term in deploy.yaml's ``observations`` order, each term scaled
      and stacked over ``history_length`` frames oldest-first, then concatenated term by term - the
      same layout Isaac Lab's ObservationManager produces for a group with per-term history (first
      push fills the whole history, same as CircularBuffer after a reset).
    - action: ``target = offset + scale * action`` in the policy's own joint order, converted to the
      AMP env's 29-dim action (``target = default_joint_pos + action * amp_action_scale``) for
      env.step(). The policy runs every ``round(policy_dt / env_step_dt)`` env steps and its target is
      held in between.
    - gain: legs/feet/waist/arms stiffness+damping (and armature) are switched to deploy.yaml's trained
      values while this controller drives the robot, written through the same explicit-tensor vs.
      write_joint_*_to_sim split as _apply_trained_stand_defaults (see its docstring), and restored on
      exit via restore_default_gains().

    Switching straight from a standing pose into a walking policy topples the robot (different pose,
    gains and an empty action history all change at once), so begin_transition() runs a staged
    hand-over first:

    1. ``interp``: the policy already runs (zero velocity command), but the commanded joint target
       *and* PD gains blend (smoothstep) from whatever was being commanded the instant before
       (``robot.data.joint_pos_target`` + the live gains - no torque discontinuity on the first step)
       to the policy's own output and trained gains. The policy has to be in the loop here: an
       open-loop blend of PD targets toward its default pose has nothing balancing the robot, and was
       verified in sim to tip the G1 over sideways out of the Stand PPO policy's leaning stance.
    2. ``warmup``: pure policy, still at zero velocity, so it settles into its own stepping stance
       before any motion is asked of it.
    3. ``active``: joystick velocity, rate-limited (see _CMD_ACCEL), is fed to the policy.
    """

    # max command change rate (vx/vy in m/s^2, wz in rad/s^2) - a joystick slammed to full deflection
    # becomes a ramp instead of a step the policy never saw during training.
    _CMD_ACCEL = (1.0, 1.0, 1.5)
    _JOYSTICK_DEADZONE = 0.05
    # warmup only hands over to the joystick once the robot has been this calm for _STABLE_HOLD seconds
    # (|projected gravity xy| ~ sin(tilt), base xy speed in m/s) - not just after a fixed time, so a
    # hand-over that left the robot still recovering never gets a velocity command on top of it.
    _STABLE_TILT = 0.12
    _STABLE_SPEED = 0.3
    _STABLE_HOLD = 0.5
    # request_stop(): hand over only once base xy speed stays under this (m/s) for _STABLE_HOLD seconds,
    # or _STOP_TIMEOUT seconds after the command reached zero regardless (the zero-velocity policy never fully freezes - it keeps
    # stepping in place on its gait clock)
    _STOPPED_SPEED = 0.15
    _STOP_TIMEOUT = 4.0

    def __init__(
        self,
        robot,
        policy_dir: str,
        device: str,
        amp_body_joint_ids: list[int],
        amp_action_scale: float,
        env_step_dt: float,
        transition_time: float,
        warmup_time: float,
        handover_time: float,
    ):
        import yaml

        from isaaclab.actuators import ImplicitActuator

        policy_dir = os.path.abspath(policy_dir)
        policy_path = os.path.join(policy_dir, "policy.pt")
        deploy_path = next(
            (
                p
                for p in (os.path.join(policy_dir, "deploy.yaml"), os.path.join(policy_dir, "..", "params", "deploy.yaml"))
                if os.path.isfile(p)
            ),
            None,
        )
        if deploy_path is None:
            raise FileNotFoundError(f"deploy.yaml not found next to {policy_dir} or in its ../params/")
        print(f"[g1_isaac_sim_bridge] Loading locomotion policy: {policy_path} (cfg: {os.path.abspath(deploy_path)})")
        with open(deploy_path) as f:
            deploy = yaml.safe_load(f)
        self.policy = torch.jit.load(policy_path, map_location=device).eval()

        self.robot = robot
        self.device = device
        self.transition_time = transition_time
        self.warmup_time = warmup_time
        self.env_step_dt = env_step_dt
        self.policy_dt = float(deploy["step_dt"])
        self.policy_decimation = max(1, round(self.policy_dt / env_step_dt))
        if abs(self.policy_decimation * env_step_dt - self.policy_dt) > 1e-6:
            print(
                f"[g1_isaac_sim_bridge] WARNING: policy step_dt {self.policy_dt} is not a multiple of the env "
                f"step_dt {env_step_dt} - running the policy every {self.policy_decimation} env steps"
            )

        # policy joint order (the training env's own articulation order) -> names -> this robot's ids
        sdk_ids = deploy["joint_ids_map"]
        self.joint_names = [_G1_SDK_JOINT_NAMES[i] for i in sdk_ids]
        self.joint_ids, _ = robot.find_joints(self.joint_names, preserve_order=True)
        self.amp_action_index = [amp_body_joint_ids.index(j) for j in self.joint_ids]
        self.n_amp_joints = len(amp_body_joint_ids)
        self.amp_action_scale = amp_action_scale

        def _tensor(values):
            return torch.tensor(values, dtype=torch.float32, device=device)

        self.default_pos = _tensor(deploy["default_joint_pos"])
        # stiffness/damping are stored in SDK order (see export_deploy_cfg.py), the rest in policy order
        self.trained_stiffness = _tensor([deploy["stiffness"][i] for i in sdk_ids])
        self.trained_damping = _tensor([deploy["damping"][i] for i in sdk_ids])
        # unitree_rl_lab's UNITREE_G1_29DOF_CFG gives every actuator armature=0.01 (not exported to deploy.yaml)
        self.trained_armature = torch.full_like(self.default_pos, 0.01)
        action_cfg = deploy["actions"]["JointPositionAction"]
        self.action_scale = _tensor(action_cfg["scale"])
        self.action_offset = _tensor(action_cfg["offset"])
        self.action_clip = action_cfg.get("clip")

        ranges = deploy["commands"]["base_velocity"]["ranges"]
        self.cmd_ranges = (ranges["lin_vel_x"], ranges["lin_vel_y"], ranges["ang_vel_z"])

        self.obs_terms = []  # (name, scale tensor, history_length, params)
        for name, term_cfg in deploy["observations"].items():
            if name not in self._OBS_TERM_DIMS:
                raise ValueError(f"Unsupported observation term in {deploy_path}: {name!r}")
            dim = self._obs_term_dim(name)
            scale = term_cfg.get("scale")
            scale_t = _tensor(scale) if scale is not None else torch.ones(dim, device=device)
            self.obs_terms.append((name, scale_t, int(term_cfg.get("history_length") or 1), term_cfg.get("params") or {}))
        obs_dim = sum(self._obs_term_dim(name) * hist for name, _, hist, _ in self.obs_terms)
        try:
            with torch.inference_mode():
                self.policy(torch.zeros(1, obs_dim, device=device))
        except RuntimeError as exc:
            raise ValueError(f"Locomotion obs dim {obs_dim} (from {deploy_path}) doesn't fit {policy_path}") from exc

        # per-actuator gain write targets: (actuator, is_implicit, robot joint ids, actuator-local ids, policy idx)
        self._gain_groups = []
        for actuator in robot.actuators.values():
            policy_idx = [i for i, n in enumerate(self.joint_names) if n in actuator.joint_names]
            if not policy_idx:
                continue
            self._gain_groups.append(
                (
                    actuator,
                    isinstance(actuator, ImplicitActuator),
                    [self.joint_ids[i] for i in policy_idx],
                    torch.tensor([actuator.joint_names.index(self.joint_names[i]) for i in policy_idx], device=device),
                    torch.tensor(policy_idx, device=device),
                )
            )
        # this robot's own gains/armature, captured before anything has run - see restore_default_gains()
        self.default_stiffness, self.default_damping = self._read_gains()
        self.default_armature = robot.data.joint_armature[0, self.joint_ids].clone()

        self.handover_time = handover_time
        self.phase = "idle"
        self._phase_time = 0.0
        self._stop_requested = False
        self._desired = torch.zeros(3, device=device)
        self._default_accel = torch.tensor(self._CMD_ACCEL, device=device)
        self._cmd_accel = self._default_accel.clone()
        self._cmd = torch.zeros(3, device=device)
        print(
            f"[g1_isaac_sim_bridge] Locomotion policy ready: obs {obs_dim}, 29 joints, every {self.policy_decimation} "
            f"env steps, cmd ranges vx={self.cmd_ranges[0]} vy={self.cmd_ranges[1]} wz={self.cmd_ranges[2]}"
        )

    _OBS_TERM_DIMS = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "velocity_commands": 3,
        "joint_pos_rel": None,  # number of policy joints
        "joint_vel_rel": None,
        "last_action": None,
        "gait_phase": 2,
    }

    def _obs_term_dim(self, name: str) -> int:
        return self._OBS_TERM_DIMS[name] or len(self.joint_names)

    def _read_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        stiffness = torch.zeros_like(self.default_pos)
        damping = torch.zeros_like(self.default_pos)
        for actuator, is_implicit, robot_ids, local_ids, policy_idx in self._gain_groups:
            if is_implicit:
                stiffness[policy_idx] = self.robot.data.joint_stiffness[0, robot_ids]
                damping[policy_idx] = self.robot.data.joint_damping[0, robot_ids]
            else:
                stiffness[policy_idx] = actuator.stiffness[0, local_ids]
                damping[policy_idx] = actuator.damping[0, local_ids]
        return stiffness, damping

    def _write_gains(self, stiffness: torch.Tensor, damping: torch.Tensor) -> None:
        for actuator, is_implicit, robot_ids, local_ids, policy_idx in self._gain_groups:
            if is_implicit:
                self.robot.write_joint_stiffness_to_sim(stiffness[policy_idx].unsqueeze(0), joint_ids=robot_ids)
                self.robot.write_joint_damping_to_sim(damping[policy_idx].unsqueeze(0), joint_ids=robot_ids)
            else:
                actuator.stiffness[:, local_ids] = stiffness[policy_idx]
                actuator.damping[:, local_ids] = damping[policy_idx]

    @torch.inference_mode()
    def begin_transition(self) -> None:
        """Start the staged hand-over (see class docstring) from whatever is being commanded right now.
        Must be called *before* anything else (e.g. _StandPolicyController.restore_default_gains) touches
        the live gains on this step, so the blend starts from the gains actually in effect."""
        self._start_target = self.robot.data.joint_pos_target[0, self.joint_ids].clone()
        self._start_stiffness, self._start_damping = self._read_gains()
        self.robot.write_joint_armature_to_sim(self.trained_armature.unsqueeze(0), joint_ids=self.joint_ids)
        self._target = self._start_target.clone()
        self._cmd = torch.zeros(3, device=self.device)
        self._reset_policy_state()
        self._stop_requested = False
        self.phase, self._phase_time = "interp", 0.0
        print(f"[g1_isaac_sim_bridge] Locomotion: blending into policy control over {self.transition_time:.1f}s")

    @torch.inference_mode()
    def restore_default_gains(self) -> None:
        """Put back this robot's own gains/armature (see _StandPolicyController.restore_default_gains for
        why leaving a mode without this leaves the next controller fighting the wrong gains)."""
        self._write_gains(self.default_stiffness, self.default_damping)
        self.robot.write_joint_armature_to_sim(self.default_armature.unsqueeze(0), joint_ids=self.joint_ids)
        self.phase = "idle"

    def set_joystick(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        """Raw g1/cmd_vel values -> normalized [-1, 1] stick deflection in the robot's own frame (+x
        forward, +y left, +z counter-clockwise). The UI's linear_y/angular_z are positive for a drag to
        the *right*, hence the sign flips."""
        raw = (linear_x / _JOYSTICK_FULL_SCALE[0], -linear_y / _JOYSTICK_FULL_SCALE[1], -angular_z / _JOYSTICK_FULL_SCALE[2])
        stick = [0.0 if abs(v) < self._JOYSTICK_DEADZONE else max(-1.0, min(1.0, v)) for v in raw]
        # positive deflection scales to the range's upper bound, negative to its lower bound, so an
        # asymmetric range (e.g. vx in [-0.5, 1.0]) still uses the full stick travel both ways
        cmd = [v * (hi if v > 0 else -lo) for v, (lo, hi) in zip(stick, self.cmd_ranges)]
        self._desired = torch.tensor(cmd, dtype=torch.float32, device=self.device)
        self._cmd_accel = torch.tensor(self._CMD_ACCEL, device=self.device)

    def set_velocity_command(self, vx: float, vy: float, wz: float, linear_accel: float | None = None) -> None:
        """Desired base velocity in m/s / rad/s (robot frame), e.g. from the go-to-point controller -
        used as given, *not* clipped to the trained command range (that's the caller's call, see
        _GOTO_RUN_SPEED). ``linear_accel`` (m/s^2) overrides _CMD_ACCEL's vx/vy ramp rate, so a run
        builds up speed gradually instead of at joystick rate."""
        self._desired = torch.tensor([vx, vy, wz], dtype=torch.float32, device=self.device)
        accel = list(self._CMD_ACCEL)
        if linear_accel is not None:
            accel[0] = accel[1] = linear_accel
        self._cmd_accel = torch.tensor(accel, device=self.device)

    def _desired_command(self) -> torch.Tensor:
        return self._desired

    def _reset_policy_state(self) -> None:
        self._last_action = torch.zeros_like(self.default_pos)
        self._history = {}
        self._policy_steps = 0
        self._substep = 0

    def _compute_obs_term(self, name: str, params: dict) -> torch.Tensor:
        data = self.robot.data
        if name == "base_ang_vel":
            return data.root_ang_vel_b[0]
        if name == "projected_gravity":
            return data.projected_gravity_b[0]
        if name == "velocity_commands":
            return self._cmd
        if name == "joint_pos_rel":
            return data.joint_pos[0, self.joint_ids] - self.default_pos
        if name == "joint_vel_rel":
            return data.joint_vel[0, self.joint_ids]
        if name == "last_action":
            return self._last_action
        # gait_phase: same clock as unitree_rl_lab's mdp.gait_phase (episode steps * step_dt), started
        # from 0 when the policy takes over, like a fresh episode
        period = float(params.get("period", 0.8))
        phase = (self._policy_steps * self.policy_dt) % period / period
        return torch.tensor(
            [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)], dtype=torch.float32, device=self.device
        )

    def _policy_step(self) -> None:
        if self.phase in ("active", "stopping"):
            desired = self._desired_command() if self.phase == "active" else torch.zeros(3, device=self.device)
            # a gentler set_velocity_command ramp only slows speeding *up* - slowing down always gets
            # _CMD_ACCEL, so e.g. the go-to-point controller can still brake in time for its target
            slowing = desired.abs() < self._cmd.abs()
            accel = torch.where(slowing, torch.maximum(self._cmd_accel, self._default_accel), self._cmd_accel)
            max_delta = accel * self.policy_dt
            self._cmd = self._cmd + torch.clamp(desired - self._cmd, -max_delta, max_delta)
        else:
            self._cmd = torch.zeros(3, device=self.device)

        parts = []
        for name, scale, history_length, params in self.obs_terms:
            frame = self._compute_obs_term(name, params) * scale
            hist = self._history.get(name)
            if hist is None:
                hist = frame.unsqueeze(0).repeat(history_length, 1)
            else:
                hist = torch.cat((hist[1:], frame.unsqueeze(0)), dim=0)
            self._history[name] = hist
            parts.append(hist.reshape(-1))
        obs = torch.cat(parts).unsqueeze(0)

        action = self.policy(obs)[0]
        if self.action_clip is not None:
            action = torch.clamp(action, -float(self.action_clip), float(self.action_clip))
        self._last_action = action
        self._policy_target = self.action_offset + self.action_scale * action
        self._policy_steps += 1

    def _update_stable_time(self, max_speed: float) -> None:
        gravity_xy = self.robot.data.projected_gravity_b[0, :2]
        speed_xy = self.robot.data.root_lin_vel_b[0, :2]
        stable = (
            torch.linalg.norm(gravity_xy).item() < self._STABLE_TILT
            and torch.linalg.norm(speed_xy).item() < max_speed
        )
        self._stable_time = self._stable_time + self.env_step_dt if stable else 0.0

    def request_stop(self) -> None:
        """Begin bringing the robot to rest for a hand-over to another (standing) controller: the
        command ramps to zero and the policy keeps balancing until the robot has been calm for a while
        (see ``ready_for_handover``). Requested during the walk-in blend, it starts once that's done.
        Braking always uses _CMD_ACCEL, even after a gentler set_velocity_command ramp - a run slowed at
        its own 0.3 m/s^2 build-up rate was still moving when _STOP_TIMEOUT ran out and fell over."""
        self._stop_requested = True
        self._cmd_accel = torch.tensor(self._CMD_ACCEL, device=self.device)

    @property
    def ready_for_handover(self) -> bool:
        return self.phase == "stopped"

    def step(self) -> torch.Tensor:
        """Advance one env step; returns the 29-dim AMP-shaped action for env.step()."""
        self._phase_time += self.env_step_dt
        if self._stop_requested and self.phase in ("warmup", "active"):
            self._stop_requested = False
            self.phase, self._phase_time, self._stable_time = "stopping", 0.0, 0.0
            print("[g1_isaac_sim_bridge] Locomotion: stopping for hand-over to the standing controller")
        if self.phase == "warmup":
            self._update_stable_time(self._STABLE_SPEED)
            if self._phase_time >= self.warmup_time and self._stable_time >= self._STABLE_HOLD:
                self.phase = "active"
                print(f"[g1_isaac_sim_bridge] Locomotion: joystick active (after {self._phase_time:.1f}s warmup)")
        elif self.phase == "stopping":
            # stricter speed bound than warmup's: hand over only once the robot has actually stopped
            self._update_stable_time(self._STOPPED_SPEED)
            if torch.any(self._cmd != 0):
                self._phase_time = 0.0  # the timeout only counts once the command has ramped to zero
            if self._stable_time >= self._STABLE_HOLD or self._phase_time >= self._STOP_TIMEOUT:
                self.phase = "stopped"
                print(f"[g1_isaac_sim_bridge] Locomotion: stopped ({self._phase_time:.1f}s after the command reached zero)")
        if self._substep % self.policy_decimation == 0:
            self._policy_step()
        self._substep += 1

        if self.phase == "interp":
            t = min(1.0, self._phase_time / max(self.transition_time, 1e-6))
            alpha = t * t * (3.0 - 2.0 * t)  # smoothstep: zero blend rate at both ends
            self._target = self._start_target + alpha * (self._policy_target - self._start_target)
            self._write_gains(
                self._start_stiffness + alpha * (self.trained_stiffness - self._start_stiffness),
                self._start_damping + alpha * (self.trained_damping - self._start_damping),
            )
            if t >= 1.0:
                self.phase, self._phase_time, self._stable_time = "warmup", 0.0, 0.0
                print("[g1_isaac_sim_bridge] Locomotion: policy in full control (zero-velocity warmup)")
        else:
            self._target = self._policy_target

        return self._to_amp_action(self._target)

    def _to_amp_action(self, target: torch.Tensor) -> torch.Tensor:
        amp_action = torch.zeros(1, self.n_amp_joints, device=self.device)
        amp_action[0, self.amp_action_index] = (
            target - self.robot.data.default_joint_pos[0, self.joint_ids]
        ) / self.amp_action_scale
        return amp_action

    @torch.inference_mode()
    def begin_handover(self) -> None:
        """Start blending out to the next controller (see handover_step) from this policy's current
        target/gains. Call once ``ready_for_handover``."""
        self._from_target = self._target.clone()
        self._from_stiffness, self._from_damping = self._read_gains()
        self.phase, self._phase_time = "handover", 0.0
        print(f"[g1_isaac_sim_bridge] Locomotion: handing over to the standing controller over {self.handover_time:.1f}s")

    def handover_step(self, next_controller_step) -> torch.Tensor:
        """One env step of the hand-over: joint targets and gains blend (smoothstep) from this policy's
        last ones to the next controller's, which keeps running closed-loop the whole time. Switching
        straight over was verified in sim to throw the G1 to ~25-30 deg of tilt.

        ``next_controller_step``: callable returning that controller's 29-dim AMP-shaped action for
        this step. It may write its own gains (as _StandPolicyController.step does) - those, plus this
        robot's defaults for any joint it doesn't write, are what the gains blend toward. Once
        ``handover_done``, the caller switches mode and calls restore_default_gains()."""
        self._phase_time += self.env_step_dt
        t = min(1.0, self._phase_time / max(self.handover_time, 1e-6))
        alpha = t * t * (3.0 - 2.0 * t)
        self._write_gains(self.default_stiffness, self.default_damping)
        next_action = next_controller_step()
        next_stiffness, next_damping = self._read_gains()
        self._write_gains(
            self._from_stiffness + alpha * (next_stiffness - self._from_stiffness),
            self._from_damping + alpha * (next_damping - self._from_damping),
        )
        next_target = (
            self.robot.data.default_joint_pos[0, self.joint_ids]
            + next_action[0, self.amp_action_index] * self.amp_action_scale
        )
        self._target = self._from_target + alpha * (next_target - self._from_target)
        if t >= 1.0:
            self.phase = "handover_done"
        # joints outside the policy's 29 (none today) keep the next controller's own action
        action = next_action.clone()
        action[0, self.amp_action_index] = self._to_amp_action(self._target)[0, self.amp_action_index]
        return action

    @property
    def handover_done(self) -> bool:
        return self.phase == "handover_done"

    def status(self) -> dict:
        return {"phase": self.phase, "cmd": [round(v, 3) for v in self._cmd.tolist()]}


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> float:
    """Yaw (rad) from an Isaac Lab (w, x, y, z) quaternion."""
    w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# Go-to-point on the locomotion policy (web UI Walk/Run + target x, y [+ velocity]). Run speed is a
# *command* value, past the policy's trained 1.0 m/s - it tracks ~1.02 m/s actual there; verified in
# sim with a 0.3 m/s^2 ramp, a 0.5 rad/s turn at speed and a stop -> Stand PPO hand-over: 1.0-1.4 never
# fell (tilt <= 0.12), 1.6 fell 1 of 2 with ~0.2 tilt and only ~0.04 m/s more actual speed.
_GOTO_WALK_SPEED = 0.5  # m/s, Walk when no velocity is entered
_GOTO_RUN_SPEED = 1.4  # m/s command - also the cap for an entered velocity
_GOTO_MIN_SPEED = 0.1
_GOTO_ACCEL = 0.3  # m/s^2 - speed builds up gradually (braking still uses the policy's _CMD_ACCEL)
_GOTO_ARRIVE_RADIUS = 0.2  # m
_GOTO_SLOWDOWN_GAIN = 0.8  # 1/s: speed limit = gain * distance, so it eases into the target


def _goto_velocity_command(
    robot_xy: tuple[float, float], yaw: float, target_xy: tuple[float, float], speed: float, max_yaw_rate: float
) -> tuple[float, float, float, bool]:
    """Robot-frame (vx, wz) toward a target for the locomotion policy. The bearing to the target vs
    the robot's own heading gives the heading error: yaw rate steers it out (capped at the policy's
    trained range), forward speed fades out with it (zero past 60 deg - turn in place first) and with
    the distance left. Returns ``(vx, wz, distance, arrived)``."""
    dx, dy = target_xy[0] - robot_xy[0], target_xy[1] - robot_xy[1]
    distance = math.hypot(dx, dy)
    if distance < _GOTO_ARRIVE_RADIUS:
        return 0.0, 0.0, distance, True
    heading_err = math.atan2(dy, dx) - yaw
    heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))
    wz = max(-max_yaw_rate, min(max_yaw_rate, 1.5 * heading_err))
    align = max(0.0, (math.cos(heading_err) - 0.5) / 0.5)
    vx = min(speed, _GOTO_SLOWDOWN_GAIN * distance) * align
    return vx, wz, distance, False


def _agent_act(agent, obs_tensor: torch.Tensor):
    """Deterministic action from a loaded skrl agent, tolerant of the exact ``Agent.act`` signature
    (skrl versions differ on whether a second "shared states" positional arg is accepted)."""
    try:
        outputs = agent.act(obs_tensor, None, timestep=0, timesteps=0)
    except TypeError:
        outputs = agent.act(obs_tensor, timestep=0, timesteps=0)
    return outputs[-1].get("mean_actions", outputs[0])


# ---------------------------------------------------------------------------
# Head camera: mounted on the robot after the env is created (never part of the registered task's
# own cfg, so training/eval scripts sharing that task are unaffected). Constructing an Isaac Lab
# Camera object flips on the sim-wide "/isaaclab/render/rtx_sensors" setting as a side effect, which
# is what makes DirectRLEnv.step()'s own decimation loop start actually calling sim.render() each
# step - no changes needed there.
# ---------------------------------------------------------------------------
def _raise_if_sensor_init_failed(sensor_name: str) -> None:
    """Surface a manually-triggered ``_initialize_callback(None)`` failure immediately.

    ``SensorBase._initialize_callback`` (isaaclab.sensors.sensor_base) catches *any* exception
    raised inside ``_initialize_impl()``, stashes it in ``builtins.ISAACLAB_CALLBACK_EXCEPTION``,
    and still marks the sensor ``_is_initialized = True`` regardless - this is normally fine since
    the callback runs off the timeline's PLAY event and a later ``RuntimeError`` there shouldn't
    crash the whole app. But ``_attach_head_camera``/``_attach_head_lidar`` invoke that same
    callback directly, well after PLAY already fired, specifically to init a sensor added post-hoc
    - here the swallowed exception means the sensor is left half-initialized (e.g. a RayCaster
    with no ``self.drift``/rays set up), and the only symptom is a confusing, unrelated-looking
    crash on the *next* ``env.reset()`` (``AttributeError: ... has no attribute 'drift'``) instead
    of the real error at the point it actually happened. Re-raise here so the real cause (e.g. a
    mount prim that isn't a rigid body / not Xformable) surfaces right away with a clear traceback.
    """
    exc = getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None)
    if exc is not None:
        builtins.ISAACLAB_CALLBACK_EXCEPTION = None
        raise RuntimeError(f"{sensor_name} sensor failed to initialize (see chained exception below)") from exc


def _robot_body_prim_path(body_name: str, robot_prim_path: str = "/World/envs/env_0/Robot") -> str:
    """Stage path of one of the robot's rigid-body links, found by searching under the robot prim.

    Must not be built as ``f"{robot_prim_path}/{body_name}"``: G1_LOCAL_USD_PATH nests every link one
    level deeper (``.../Robot/g1_full_collision/torso_link``), so that guessed path doesn't exist - and
    spawning a sensor there silently creates a brand-new, static Xform instead, i.e. a camera/lidar
    fixed in the world rather than riding on the robot. Matching on RigidBodyAPI also skips same-named
    collision sub-prims (e.g. ``torso_link/collisions/head_link``).
    """
    from pxr import Usd, UsdPhysics

    from isaaclab.sim.utils.stage import get_current_stage

    for prim in Usd.PrimRange(get_current_stage().GetPrimAtPath(robot_prim_path)):
        if prim.GetName() == body_name and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
    raise RuntimeError(f"No rigid body named {body_name!r} under {robot_prim_path}")


def _find_camera_mount_body(body_names: list[str]) -> str:
    """Pick a body to mount the camera on: prefer a head-ish link, then torso/pelvis, else whatever
    the first body is. Exact link names vary by G1 USD build, hence the defensive fallback chain
    (mirrors the same pattern used for ``G1AsapEnvCfg.keypoint_body_names``)."""
    lowered = [n.lower() for n in body_names]
    for pattern in ("head", "torso", "pelvis"):
        for name, low in zip(body_names, lowered):
            if pattern in low:
                return name
    return body_names[0]


# Neck camera mount, in torso_link's own frame (+X forward, +Z up). Measured off G1_LOCAL_USD_PATH:
# the head's collision mesh spans z=0.28..0.49 m above torso_link (so the neck - the bottom of the
# head - is at ~0.29 m) and reaches x=+0.079 m at the front, so x=0.09 puts the lens just clear of
# the face. (The asset's own d435_link sits higher, z=0.43, and is pitched 47.6 deg down at the floor.)
_NECK_CAMERA_POS = (0.09, 0.0, 0.29)
_NECK_CAMERA_PITCH_DEG = 10.0  # slight downward tilt so the floor right in front of the feet is in view


def _attach_head_camera(robot, scene, width: int, height: int) -> Camera:
    """Spawn a forward-looking RGB camera on the robot's neck (the (single, num_envs=1) robot's
    ``torso_link``, see _NECK_CAMERA_POS) and return the live sensor object.

    The camera prim is a child of the torso link's own prim (found via _robot_body_prim_path), so it
    moves and turns with the robot.
    """
    torso_path = _robot_body_prim_path("torso_link")
    half_pitch = math.radians(_NECK_CAMERA_PITCH_DEG) / 2.0
    # convention="world": forward=+X/up=+Z in the parent link's frame; +pitch about +Y tilts +X down
    offset = CameraCfg.OffsetCfg(
        pos=_NECK_CAMERA_POS, rot=(math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0), convention="world"
    )
    mount_desc = f"{torso_path} neck (pos {_NECK_CAMERA_POS}, pitch {_NECK_CAMERA_PITCH_DEG:.0f} deg down)"

    camera_cfg = CameraCfg(
        prim_path=f"{torso_path}/head_cam",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.05, 20.0)),
        offset=offset,
    )
    camera = Camera(cfg=camera_cfg)
    # SensorBase (see isaaclab.sensors.sensor_base) only calls _initialize_impl() - which sets up
    # the buffers _update_outdated_buffers() needs (e.g. _is_outdated) - in response to the
    # timeline's PLAY event. gym.make() above already started playback before this camera existed,
    # so that event already fired and this sensor would otherwise stay uninitialized forever
    # (AttributeError: 'Camera' object has no attribute '_is_outdated' the first time .data is
    # read). Firing the same callback manually is the only way to initialize a sensor added after
    # the sim is already playing.
    camera._initialize_callback(None)
    _raise_if_sensor_init_failed("head_cam")
    # scene.update(dt) (called every physics substep inside env.step()'s decimation loop) is what
    # actually calls camera.update(dt) to mark its buffers dirty/refresh them - it only iterates
    # scene.sensors, so a camera that's never registered there gets its "outdated" flag consumed
    # exactly once on the first .data read and never refreshed again (looks like the feed is frozen
    # on the first frame forever, which is exactly what was happening before this line existed).
    scene.sensors["head_cam"] = camera
    print(f"[g1_isaac_sim_bridge] Head camera mounted on {mount_desc} ({width}x{height} RGB)")
    return camera


def _encode_camera_frame(camera: Camera, jpeg_quality: int) -> tuple[int, int, str] | None:
    """Grab the camera's latest RGB frame and JPEG-encode it as a base64 string, or None if the
    camera hasn't produced a frame yet."""
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.shape[0] == 0:
        return None
    frame = rgb[0, ..., :3].detach().cpu().numpy().astype("uint8")
    height, width = frame.shape[0], frame.shape[1]
    buf = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(buf, format="JPEG", quality=jpeg_quality)
    return width, height, base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Walled room + head lidar: like the head camera above, spawned/attached after gym.make() so
# training/eval scripts sharing the registered task are unaffected - the task's own _setup_scene()
# spawns nothing but a flat /World/ground plane, which gives a lidar nothing to hit. This gives the
# lidar (and the SLAM stack consuming it, see ros2_slam_bridge.py) real geometry to map, and gives
# the robot room to walk/run around in.
# ---------------------------------------------------------------------------
ROOM_HALF_SIZE = 7.0  # interior half-width/depth (m) -> ~14x14m room
WALL_HEIGHT = 2.4  # comfortably taller than head height, so the horizontal lidar ring always hits it
WALL_THICKNESS = 0.3
LIDAR_MAX_RANGE = 20.0  # m - room's diagonal is ~20m, plenty of margin
LIDAR_RANGE_MIN = 0.1

_OBSTACLE_SPAWN_CLEARANCE = 1.5  # min distance (m) from the robot's own spawn point (0, 0 in room-local
# offsets) - keeps a randomly placed obstacle from landing on top of the robot.
_OBSTACLE_MIN_SEPARATION = 1.2  # min center-to-center distance (m) between two obstacles - largest
# obstacle half-extent here is the 0.6x0.6 box's ~0.42m diagonal, so this still leaves clearance
# between any two even at their closest.
_OBSTACLE_WALL_MARGIN = 1.0  # min distance (m) an obstacle center must stay from the room's interior
# walls (ROOM_HALF_SIZE), so it never clips into a wall.
_OBSTACLE_LAYOUT_SEED = 42  # fixed so the obstacle layout is the same on every run of this script -
# still randomized in *shape* (Latin hypercube, see _sample_obstacle_offsets) rather than the old
# recognizable grid, just not re-randomized on every launch.


def _sample_obstacle_offsets(count: int) -> list[tuple[float, float]]:
    """Latin-hypercube-samples `count` (dx, dy) offsets from the room center.

    Plain independent-uniform sampling (the first version of this function) can, purely by chance,
    still cluster several obstacles into a similar x or y band - e.g. three obstacles all landing
    near the same row even though each individual draw was random. Latin hypercube sampling avoids
    that structurally: the valid x-range and y-range are each split into `count` equal strata,
    independently shuffled, then paired up 1:1, so every obstacle gets its own x-stratum *and* its
    own y-stratum - no two obstacles can ever share a row or column of that grid, regardless of how
    the random draws land. The exact point within its paired stratum is still uniform-random (and
    re-rolled a few times if unlucky enough to land too close to the robot's spawn point or, near a
    stratum boundary, another obstacle) - see the _OBSTACLE_* constants above.

    Uses its own seeded `random.Random(_OBSTACLE_LAYOUT_SEED)` instance rather than the shared
    top-level `random` module - so the layout comes out identical on every run (that's the point,
    see _OBSTACLE_LAYOUT_SEED), regardless of whatever else in this process may or may not have
    touched the module-global random state first.
    """
    rng = random.Random(_OBSTACLE_LAYOUT_SEED)
    bound = ROOM_HALF_SIZE - _OBSTACLE_WALL_MARGIN
    stratum_size = (2.0 * bound) / count
    x_strata = list(range(count))
    y_strata = list(range(count))
    rng.shuffle(x_strata)
    rng.shuffle(y_strata)

    offsets: list[tuple[float, float]] = []
    for i in range(count):
        x_lo = -bound + x_strata[i] * stratum_size
        y_lo = -bound + y_strata[i] * stratum_size
        dx, dy = 0.0, 0.0
        for _attempt in range(20):
            dx = rng.uniform(x_lo, x_lo + stratum_size)
            dy = rng.uniform(y_lo, y_lo + stratum_size)
            if math.hypot(dx, dy) < _OBSTACLE_SPAWN_CLEARANCE:
                continue
            if any(math.hypot(dx - px, dy - py) < _OBSTACLE_MIN_SEPARATION for px, py in offsets):
                continue
            break
        # falls back to the last-sampled (possibly too-close) point if every attempt collided -
        # very unlikely once each obstacle already has its own stratum, but keeps this from ever
        # raising instead of just occasionally placing two obstacles a bit closer than ideal.
        offsets.append((dx, dy))
    return offsets


def _spawn_room(env_origin_xy: tuple[float, float]) -> None:
    """Spawn a static walled room + a few obstacles around the robot's spawn point.

    Every prim here is collision-only (``collision_props`` set, ``rigid_props``/``mass_props`` left
    ``None``) - the same static-collider pattern the ground plane itself already uses
    (``spawn_ground_plane`` in ``g1_isaac_amp_env.py``). That makes them immovable by construction,
    not just "heavy": there's no RigidBodyAPI on them at all for the robot to shove, regardless of
    impact force.
    """
    ox, oy = env_origin_xy
    collision = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    span = 2.0 * ROOM_HALF_SIZE

    walls = [
        # (name, size(x, y, z), translation(x, y, z))
        ("wall_north", (span + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT), (ox, oy + ROOM_HALF_SIZE, WALL_HEIGHT / 2)),
        ("wall_south", (span + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT), (ox, oy - ROOM_HALF_SIZE, WALL_HEIGHT / 2)),
        ("wall_east", (WALL_THICKNESS, span, WALL_HEIGHT), (ox + ROOM_HALF_SIZE, oy, WALL_HEIGHT / 2)),
        ("wall_west", (WALL_THICKNESS, span, WALL_HEIGHT), (ox - ROOM_HALF_SIZE, oy, WALL_HEIGHT / 2)),
    ]
    for name, size, translation in walls:
        sim_utils.spawn_cuboid(
            f"/World/room/{name}", sim_utils.CuboidCfg(size=size, collision_props=collision), translation=translation
        )

    # a few pillars scattered off-center so they don't block the spawn point - randomized (rejection
    # sampling, see _sample_obstacle_offsets) rather than fixed corners, so the layout isn't a
    # recognizable grid every time this room gets (re)spawned.
    offsets = _sample_obstacle_offsets(4)
    for i, (dx, dy) in enumerate(offsets[:2]):
        sim_utils.spawn_cuboid(
            f"/World/room/obstacle_box_{i}",
            sim_utils.CuboidCfg(size=(0.6, 0.6, WALL_HEIGHT), collision_props=collision),
            translation=(ox + dx, oy + dy, WALL_HEIGHT / 2),
        )
    for i, (dx, dy) in enumerate(offsets[2:]):
        sim_utils.spawn_cylinder(
            f"/World/room/obstacle_cyl_{i}",
            sim_utils.CylinderCfg(radius=0.4, height=WALL_HEIGHT, collision_props=collision),
            translation=(ox + dx, oy + dy, WALL_HEIGHT / 2),
        )
    print(f"[g1_isaac_sim_bridge] Spawned {span:.0f}x{span:.0f}m walled room + 4 static obstacles around ({ox:.1f}, {oy:.1f})")


def _attach_head_lidar(robot, scene) -> MultiMeshRayCaster:
    """Spawn a head-mounted 360-degree horizontal lidar (single-ring 2D scan, matching the real
    spinning lidars used for 2D SLAM) and return the live sensor object. Must be called after
    ``_spawn_room`` - the raycast mesh targets below need to already exist on the stage.

    Mounted on the asset's ``d435_link`` (a fixed sub-frame of the torso at head height - not a body,
    so it never shows up in ``robot.body_names``) when present, else on the head/torso/pelvis body
    itself, and uses the same post-play-start init workaround as ``_attach_head_camera`` (RayCaster
    sensors need the identical ``_initialize_callback(None)`` + ``scene.sensors[...]`` registration
    trick as Camera).

    Casts only against ``/World/ground`` and ``/World/room/*`` - the robot's own body is
    deliberately never a raycast target, so unlike the real sensor there's no self-occlusion to
    simulate here; the head mount is used anyway to keep the sim setup representative of where this
    would actually sit on real hardware.
    """
    from isaaclab.sim.utils.stage import get_current_stage

    mount_body = _find_camera_mount_body(robot.body_names)
    base_path = _robot_body_prim_path(mount_body)
    d435_path = f"{base_path}/d435_link"

    if get_current_stage().GetPrimAtPath(d435_path).IsValid():
        parent_path = d435_path
        mount_desc = f"{mount_body}/d435_link"
    else:
        parent_path = base_path
        mount_desc = mount_body

    # Attach the ray-caster to a freshly-spawned Xform child rather than `parent_path` directly.
    # `parent_path` is a live, physics-driven robot link - once the sim is playing, IsaacLab's
    # RayCaster._obtain_trackable_prim_view() ends up handing that (existing) prim to XFormPrimView,
    # which validates it via `prim.IsA(UsdGeom.Xformable)` straight off the plain USD stage. Under
    # the GPU/Fabric physics pipeline that query can come back empty (`GetTypeName() == ''`) for a
    # prim actively owned by physics, even though it's a perfectly normal Xform - raising "Prim ...
    # is not an xformable" and crashing the ray-caster's init. `_attach_head_camera` above never hits
    # this because it always spawns its own fresh Camera prim under `parent_path` rather than
    # tracking the link prim itself; mirror that here with a plain identity-offset Xform mount so the
    # ray-caster tracks a prim that was never claimed by physics/Fabric.
    mount_path = f"{parent_path}/lidar_mount"
    sim_utils.create_prim(mount_path, "Xform", translation=(0.0, 0.0, 0.0))

    lidar_cfg = MultiMeshRayCasterCfg(
        prim_path=mount_path,
        mesh_prim_paths=["/World/ground", "/World/room/.*"],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=0.5,
        ),
        ray_alignment="yaw",  # scan plane stays level regardless of torso pitch/roll during gait
        max_distance=LIDAR_MAX_RANGE,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
    )
    lidar = MultiMeshRayCaster(cfg=lidar_cfg)
    # see _attach_head_camera's identical comment above: a sensor added after gym.make() already
    # started playback misses the PLAY-event callback that normally initializes it.
    lidar._initialize_callback(None)
    _raise_if_sensor_init_failed("head_lidar")
    scene.sensors["head_lidar"] = lidar
    print(f"[g1_isaac_sim_bridge] Head lidar mounted on {mount_desc} (360 deg, {LIDAR_MAX_RANGE:.0f}m range)")
    return lidar


def _encode_lidar_scan(lidar: MultiMeshRayCaster) -> LidarScan | None:
    """Convert the lidar's latest ray-hit buffer into a LaserScan-shaped ``LidarScan``. Misses (no
    mesh within ``LIDAR_MAX_RANGE``) come back as ``inf`` in ``ray_hits_w`` - reported here as
    ``range_max + 0.01``, the standard "no return" convention any LaserScan consumer expects."""
    hits = lidar.data.ray_hits_w
    if hits is None or hits.shape[1] == 0:
        return None
    origin = lidar.data.pos_w[0]
    dists = torch.linalg.norm(hits[0] - origin, dim=-1)
    no_return = LIDAR_MAX_RANGE + 0.01
    dists = torch.nan_to_num(dists, nan=no_return, posinf=no_return, neginf=no_return)
    dists = torch.clamp(dists, min=LIDAR_RANGE_MIN, max=no_return)
    num_rays = dists.shape[0]
    angle_increment = (2.0 * math.pi) / num_rays
    return LidarScan(
        angle_min=-math.pi,
        angle_max=math.pi - angle_increment,
        angle_increment=angle_increment,
        range_min=LIDAR_RANGE_MIN,
        range_max=LIDAR_MAX_RANGE,
        ranges=dists.tolist(),
    )


# ---------------------------------------------------------------------------
# Ground alignment: measure the *actual* post-teleport foot height via the robot's own forward
# kinematics (robot.data.body_pos_w/body_quat_w, already computed for us by Isaac Lab - no need to
# hand-derive the leg chain) and shift the root Z to close the gap exactly, every reset. Replaces an
# earlier fixed "+3cm spawn height" guess that wasn't reliably enough (or too much) depending on the
# joint pose actually used at reset.
# ---------------------------------------------------------------------------
# Local Z offset (m) from left/right_ankle_roll_link's own origin down to the bottom of its foot-sole
# collision spheres - measured directly from assets/g1_full_collision.usd (4x radius-0.005m spheres
# per foot, all at local z=-0.03, so sole = -0.03 - 0.005 = -0.035). Static: doesn't depend on the
# leg's joint angles, only on this asset's own (fixed) foot collision geometry.
FOOT_SOLE_LOCAL_Z_OFFSET = -0.035


def _align_feet_to_ground(robot, foot_body_ids: list[int], ground_z: float, clearance: float) -> None:
    """Shift env 0's root Z so the lower of the two feet' sole points sits ``clearance`` meters above
    ``ground_z``. Must be called right after a pose has been written to the robot (so
    ``body_pos_w``/``body_quat_w`` reflect it) and from inside ``torch.inference_mode()``, matching
    the context ``env.reset()`` itself runs under."""
    local_offset = torch.tensor([0.0, 0.0, FOOT_SOLE_LOCAL_Z_OFFSET], device=robot.device)
    sole_world_z = []
    for body_id in foot_body_ids:
        rotated = quat_apply(robot.data.body_quat_w[0, body_id], local_offset)
        sole_world_z.append(robot.data.body_pos_w[0, body_id, 2].item() + rotated[2].item())

    shift = (ground_z + clearance) - min(sole_world_z)

    root_pose = torch.cat([robot.data.root_pos_w[0:1].clone(), robot.data.root_quat_w[0:1].clone()], dim=-1)
    print(
        f"[g1_isaac_sim_bridge] _align_feet_to_ground: root xy going in = "
        f"({root_pose[0, 0].item():.2f}, {root_pose[0, 1].item():.2f}), lowest sole z = "
        f"{min(sole_world_z):.4f}, height shift applied = {shift:.4f}"
    )
    root_pose[0, 2] += shift
    env_ids = torch.tensor([0], device=robot.device)
    robot.write_root_pose_to_sim(root_pose, env_ids)


# ---------------------------------------------------------------------------
# Shared state between the DDS callback thread and the main sim-stepping thread.
# ---------------------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "stand"  # "stand" | "manual" | "goto" | "policy"
        self.gait = "stand"  # "stand" | "walk" | "run"
        # (linear_x, linear_y, angular_z) as sent by the web UI's joystick, normalized by
        # _LocomotionPolicyController.set_joystick
        self.cmd_vel = (0.0, 0.0, 0.0)
        self.goto_target: tuple[float, float] | None = None
        self.goto_speed = 0.0  # m/s from the web UI's velocity field, 0 = the gait's default
        self.home_requested = False
        self.reset_requested = False
        self.estop_requested = False
        self.stand_requested = False

    def snapshot(self):
        with self.lock:
            return self.mode, self.gait, self.cmd_vel, self.goto_target


class G1SimBridgeDds:
    """Raw-DDS (CycloneDDS) endpoint the web controller talks to (see module docstring for the
    topic table)."""

    def __init__(self, domain_id: int, state: SharedState, joint_names: list[str]):
        self.state = state
        self.joint_names = joint_names
        self.participant = make_participant(domain_id)

        self._cmd_vel_reader = DataReader(
            self.participant, Topic(self.participant, TOPIC_CMD_VEL, CmdVel), listener=dds_listener(self._on_cmd_vel)
        )
        self._goto_reader = DataReader(
            self.participant,
            Topic(self.participant, TOPIC_GOTO_COMMAND, GotoCommand),
            listener=dds_listener(self._on_goto_command),
        )
        self._goto_cancel_reader = DataReader(
            self.participant,
            Topic(self.participant, TOPIC_GOTO_CANCEL, Trigger),
            listener=dds_listener(self._on_goto_cancel),
        )
        self._policy_reader = DataReader(
            self.participant,
            Topic(self.participant, TOPIC_POLICY_COMMAND, PolicyCommand),
            listener=dds_listener(self._on_policy_command),
        )
        self._home_reader = DataReader(
            self.participant, Topic(self.participant, TOPIC_HOME_POSITION, Trigger), listener=dds_listener(self._on_home)
        )
        self._reset_reader = DataReader(
            self.participant, Topic(self.participant, TOPIC_RESET_SIM, Trigger), listener=dds_listener(self._on_reset)
        )
        self._estop_reader = DataReader(
            self.participant, Topic(self.participant, TOPIC_EMERGENCY_STOP, Trigger), listener=dds_listener(self._on_estop)
        )
        self._stand_reader = DataReader(
            self.participant, Topic(self.participant, TOPIC_STAND_COMMAND, Trigger), listener=dds_listener(self._on_stand)
        )

        self._pose_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_ROBOT_POSE, RobotPose))
        self._joint_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_JOINT_STATES, JointState))
        self._status_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_STATUS, Status))
        self._camera_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_CAMERA_FRAME, CameraFrame))
        self._lidar_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_LIDAR_SCAN, LidarScan))

        print(
            f"[g1_isaac_sim_bridge] DDS ready on domain {domain_id} - listening on "
            f"{TOPIC_CMD_VEL}, {TOPIC_GOTO_COMMAND}, {TOPIC_POLICY_COMMAND}"
        )

    def _on_cmd_vel(self, msg: CmdVel) -> None:
        with self.state.lock:
            self.state.cmd_vel = (msg.linear_x, msg.linear_y, msg.angular_z)
            if max(abs(msg.linear_x), abs(msg.linear_y), abs(msg.angular_z)) > 0.05:
                self.state.mode = "manual"
                if self.state.gait == "stand":
                    self.state.gait = "walk"
            # releasing the joystick keeps 'manual' mode - the walking policy at zero velocity

    def _on_goto_command(self, msg: GotoCommand) -> None:
        with self.state.lock:
            self.state.mode = "goto"
            self.state.gait = msg.gait if msg.gait in ("walk", "run") else "walk"
            self.state.goto_target = (msg.x, msg.y)
            self.state.goto_speed = max(0.0, msg.speed)
        print(f"[g1_isaac_sim_bridge] DDS received g1/goto_command: ({msg.x:.2f}, {msg.y:.2f}) {msg.gait} speed={msg.speed:.2f}")

    def _on_goto_cancel(self, _msg: Trigger) -> None:
        with self.state.lock:
            if self.state.mode == "goto":
                # walking policy: same stop-then-blend as the Stand button, not a hard switch
                self.state.stand_requested = True
            self.state.goto_target = None

    def _on_policy_command(self, msg: PolicyCommand) -> None:
        print(f"[g1_isaac_sim_bridge] DDS received g1/policy_command: {msg.command!r}")
        with self.state.lock:
            if msg.command == "play":
                self.state.mode = "policy"
            elif msg.command == "stop" and self.state.mode == "policy":
                self.state.mode = "stand"

    def _on_home(self, _msg: Trigger) -> None:
        print("[g1_isaac_sim_bridge] DDS received g1/home_position -> will reset next loop iteration")
        with self.state.lock:
            self.state.home_requested = True

    def _on_reset(self, _msg: Trigger) -> None:
        print("[g1_isaac_sim_bridge] DDS received g1/reset_sim -> will reset next loop iteration")
        with self.state.lock:
            self.state.reset_requested = True

    def _on_stand(self, _msg: Trigger) -> None:
        print("[g1_isaac_sim_bridge] DDS received g1/stand_command")
        with self.state.lock:
            self.state.stand_requested = True

    def _on_estop(self, _msg: Trigger) -> None:
        print("[g1_isaac_sim_bridge] DDS received g1/emergency_stop")
        with self.state.lock:
            self.state.estop_requested = True

    def publish_telemetry(self, root_xy, root_z, quat_wxyz, joint_pos, joint_vel, status: dict) -> None:
        self._pose_writer.write(
            RobotPose(
                x=root_xy[0],
                y=root_xy[1],
                z=root_z,
                qw=quat_wxyz[0],
                qx=quat_wxyz[1],
                qy=quat_wxyz[2],
                qz=quat_wxyz[3],
            )
        )
        self._joint_writer.write(JointState(name=self.joint_names, position=joint_pos, velocity=joint_vel))
        self._status_writer.write(Status(payload_json=json.dumps(status)))

    def publish_camera_frame(self, width: int, height: int, jpeg_base64: str) -> None:
        self._camera_writer.write(CameraFrame(width=width, height=height, jpeg_base64=jpeg_base64))

    def publish_lidar_scan(self, scan: LidarScan) -> None:
        self._lidar_writer.write(scan)


@hydra_task_config(args_cli.task, AGENT_CFG_ENTRY_POINT)
def main(env_cfg: DirectRLEnvCfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # self-collisions OFF - same as training (scripts/skrl/train.py never enables it). Turning it on
    # here (as scripts/skrl/play.py does, purely for a trained policy's visual realism) is what was
    # actually causing the reset-time "bounce": at the default reset pose (arms at sides, all
    # shoulder/elbow angles at 0) the arm collision cylinders very plausibly rest against/inside the
    # torso mesh, and PhysX's depenetration launches the whole articulation apart to resolve it the
    # instant self-collision is checked. Ground penetration and a stale PD target were both ruled
    # out (see _align_feet_to_ground and do_reset_and_align) before landing on this as the cause.
    env_cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = False
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else env_cfg.seed
    # no automatic reset-on-fall: a fallen robot stays fallen in sim until the web UI's home/reset
    # button explicitly calls env.reset() (see the episode_length_buf zeroing below for the other
    # half - neutralizing the fixed episode_length_s timeout reset)
    env_cfg.early_termination = False
    # reset to the robot's own USD/articulation default pose (a neutral standing pose, default_root_state
    # + default_joint_pos - the same pose you'd see right after loading the robot, before any motion is
    # applied) instead of G1AmpEnvCfg's own default ("random-start": frame 0 of the dance clip). That
    # frame has both arms swept in close to the torso (large shoulder roll/yaw), and since the retargeted
    # motion only tracks the 29 body joints - fingers always snap to their own default pose regardless -
    # the bulky 5-finger hands end up visibly clipping through each other/the torso right at reset.
    env_cfg.reset_strategy = "default"

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped
    device = unwrapped.device
    robot = unwrapped.robot
    # snapshot before _apply_trained_stand_defaults below could mutate default_joint_pos -
    # _StandPolicyController needs the untouched isaaclab_assets original for its own pose/deviation
    # math regardless (see its constructor docstring), even though (see next comment) that function
    # is now skipped entirely whenever a live Stand policy is in play.
    original_default_joint_pos = robot.data.default_joint_pos.clone()
    if args_cli.stand_policy_checkpoint:
        # *Don't* apply the static pose/gain snapshot when a live Stand policy is requested - it would
        # corrupt two things _StandPolicyController depends on being the pristine isaaclab_assets
        # original: (1) AMP/ASAP's own _apply_action computes joint targets as
        # `robot.data.default_joint_pos + action * amp_action_scale` - _StandPolicyController.step()
        # converts the Stand policy's pose_action into an equivalent `action` on the assumption that
        # this default is untouched; if _apply_trained_stand_defaults already overwrote it, every
        # commanded target ends up offset by (snapshot_value - original_value), up to ~0.25 rad per
        # joint - and (2) the legs/feet actuators' own stiffness/damping tensors, which
        # _StandPolicyController reads once at construction as its "default gain" baseline for
        # gain_scale_max**action scaling - if _apply_trained_stand_defaults already overwrote *those*
        # too, that baseline is the snapshot's gain, not isaaclab_assets', throwing off every gain the
        # policy computes. Both together are exactly why an earlier version of this integration looked
        # visibly wrong/fell over despite the policy loading correctly - the values reaching the robot
        # were quietly offset from what the policy actually intended at every single joint.
        print(
            "[g1_isaac_sim_bridge] --stand_policy_checkpoint set - skipping the static "
            "_apply_trained_stand_defaults snapshot so the live policy sees isaaclab_assets' original "
            "pose/gain as its reference (see main()'s own comment for why mixing the two corrupts both)."
        )
    else:
        # "stand" mode / every reset now hold the pose+gain a trained G1-PPO-Direct-Stand-v0 policy
        # actually settled on, rather than isaaclab_assets' untrained default - see
        # _apply_trained_stand_defaults's own docstring for where these values came from.
        _apply_trained_stand_defaults(robot)
    body_joint_ids = unwrapped.body_joint_ids
    joint_names = [robot.joint_names[i] for i in body_joint_ids]
    action_scale = unwrapped.cfg.action_scale
    step_dt = env.unwrapped.step_dt
    foot_body_ids, _ = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"])
    env_origin_z = unwrapped.scene.env_origins[0, 2].item()
    env_origin = unwrapped.scene.env_origins[0, :2].tolist()

    if not args_cli.no_lidar:
        _spawn_room(env_origin)

    def do_reset_and_align():
        """env.reset() followed by an exact, measured foot-to-ground alignment (see
        _align_feet_to_ground) - must run inside inference_mode for the same reason the reset
        itself does (env.step() below already runs under it, see the do_reset handling)."""
        with torch.inference_mode():
            reset_obs, _ = env.reset()
            _align_feet_to_ground(robot, foot_body_ids, env_origin_z, args_cli.ground_clearance)
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.forward()
        return reset_obs

    state = SharedState()
    # the standing controller startup, reset/home and the Stand button all return to: the live Stand PPO
    # policy when one is loaded (a plain PD hold of the default pose topples this robot), else 'stand'
    # (that PD hold, of the _apply_trained_stand_defaults snapshot)
    standing_mode = "policy" if args_cli.stand_policy_checkpoint else "stand"
    if args_cli.stand_policy_checkpoint:
        # skip the "stand" (static-snapshot pose) default and go straight into 'policy' mode
        # (the Stand PPO policy, see _StandPolicyController) as soon as the robot loads - no need to
        # press the web UI's policy-play button first. Only when a stand policy was actually requested;
        # without --stand_policy_checkpoint the default stays "stand" (unchanged behavior).
        state.mode = "policy"

    camera = None
    # reported in every g1/status so the web UI's CAM FEED panel can say *why* there's no picture
    # instead of a bare "not connected"
    camera_error = "disabled (--no_camera)" if args_cli.no_camera else None
    if not args_cli.no_camera:
        # the camera is a nice-to-have (see the per-step encode below) - a failure here shouldn't take
        # the whole sim down with it, the robot is still drivable without a picture
        try:
            camera = _attach_head_camera(robot, unwrapped.scene, args_cli.camera_width, args_cli.camera_height)
        except RuntimeError as exc:
            cause = exc.__cause__ or exc
            camera_error = f"{type(cause).__name__}: {cause}"
            if int(np.__version__.split(".")[0]) >= 2:
                # Isaac Sim 5.x's Replicator (omni.syntheticdata) is built against numpy 1.x - under
                # numpy 2 every camera annotator fails to attach with "Unable to write from unknown dtype"
                camera_error += (
                    f" - numpy {np.__version__} is installed but Isaac Sim needs numpy 1.x "
                    "(pip install 'numpy==1.26.4' 'opencv-python<4.11')"
                )
            print(f"[g1_isaac_sim_bridge] WARNING: head camera unavailable, continuing without it - {camera_error}")
    lidar = None if args_cli.no_lidar else _attach_head_lidar(robot, unwrapped.scene)

    node = G1SimBridgeDds(args_cli.dds_domain_id, state, joint_names)

    skrl_agent = None

    def ensure_policy_loaded():
        nonlocal skrl_agent
        if skrl_agent is not None:
            return skrl_agent
        print("[g1_isaac_sim_bridge] Loading skrl AMP dance policy checkpoint ...")
        wrapper_env = SkrlVecEnvWrapper(env, ml_framework="torch")
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(wrapper_env, experiment_cfg)
        if args_cli.policy_checkpoint:
            resume_path = os.path.abspath(args_cli.policy_checkpoint)
        else:
            log_root_path = os.path.abspath(
                os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
            )
            resume_path = get_checkpoint_path(log_root_path, run_dir=".*_amp_torch", other_dirs=["checkpoints"])
        print(f"[g1_isaac_sim_bridge] Loading checkpoint: {resume_path}")
        runner.agent.load(resume_path)
        runner.agent.enable_training_mode(False, apply_to_models=True)
        skrl_agent = runner.agent
        return skrl_agent

    # separate from ensure_policy_loaded() above (AMP dance, an skrl Agent/Runner) - this is a plain
    # PPO Gaussian model for G1-PPO-Direct-Stand-v0, loaded and run standalone (see
    # _StandPolicyController's own docstring for why it can't reuse the Runner/env.make path the AMP
    # one does). Only relevant when --stand_policy_checkpoint is set; 'policy' mode falls back to the
    # AMP dance policy above otherwise.
    stand_policy_controller = None

    def ensure_stand_policy_loaded():
        nonlocal stand_policy_controller
        if stand_policy_controller is not None:
            return stand_policy_controller
        stand_policy_controller = _StandPolicyController(
            robot, args_cli.stand_policy_checkpoint, device, body_joint_ids, original_default_joint_pos, action_scale
        )
        return stand_policy_controller

    # built eagerly (unlike the two policies above) so it captures this robot's own gains/armature
    # before any controller has touched them - see _LocomotionPolicyController.restore_default_gains.
    loco_controller = _LocomotionPolicyController(
        robot,
        args_cli.loco_policy_dir,
        device,
        body_joint_ids,
        action_scale,
        step_dt,
        args_cli.loco_transition_time,
        args_cli.loco_warmup_time,
        args_cli.loco_handover_time,
    )

    def drive_loco_to_target(goto_target, gait) -> float | None:
        """Feed the locomotion policy this step's go-to-point velocity; on arrival, hand the robot to
        'manual' at zero velocity - the walking policy keeps stepping in place, ready for the joystick
        or the next Walk/Run (the Stand button still returns it to standing). Returns the distance
        left (None once arrived)."""
        if goto_target is None:
            loco_controller.set_velocity_command(0.0, 0.0, 0.0)
            return None
        root_xy = (robot.data.root_pos_w[0, 0].item() - env_origin[0], robot.data.root_pos_w[0, 1].item() - env_origin[1])
        yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[0])
        with state.lock:
            requested = state.goto_speed
        speed = requested if requested > 0 else (_GOTO_RUN_SPEED if gait == "run" else _GOTO_WALK_SPEED)
        speed = max(_GOTO_MIN_SPEED, min(_GOTO_RUN_SPEED, speed))
        max_yaw_rate = loco_controller.cmd_ranges[2][1]
        vx, wz, distance, arrived = _goto_velocity_command(root_xy, yaw, goto_target, speed, max_yaw_rate)
        loco_controller.set_velocity_command(vx, 0.0, wz, linear_accel=_GOTO_ACCEL)
        if arrived:
            print(f"[g1_isaac_sim_bridge] Go-to: arrived at {goto_target} ({distance:.2f}m off) - stepping in place")
            with state.lock:
                state.goto_target = None
                state.mode, state.cmd_vel = "manual", (0.0, 0.0, 0.0)
        return distance

    obs = do_reset_and_align()

    step_counter = 0
    # tracked so a "policy" -> anything-else mode change can restore normal actuator gain (see
    # _StandPolicyController.restore_default_gains's own docstring for why this is needed: that
    # policy overwrites legs/feet/waist stiffness/damping live every step it runs, and nothing else
    # ever puts them back - without this, switching to manual/goto leaves those joints running on
    # whatever (possibly very low) gain the policy last picked, so commanded targets barely move them
    # even though the command itself is fine.
    previous_mode = state.mode
    try:
        while simulation_app.is_running():
            loop_start = time.time()

            with state.lock:
                do_estop = state.estop_requested
                state.estop_requested = False
                if do_estop:
                    state.mode, state.gait, state.goto_target = "stand", "stand", None
                    state.cmd_vel = (0.0, 0.0, 0.0)
                do_stand = state.stand_requested
                state.stand_requested = False
                # the walking policy can't just be switched off mid-stride (see
                # _LocomotionPolicyController.request_stop/handover_step) - it stays in 'manual' until
                # its hand-over finishes; any other mode goes straight to the standing controller
                stand_via_loco = do_stand and state.mode in ("manual", "goto")
                if do_stand:
                    state.cmd_vel, state.goto_target = (0.0, 0.0, 0.0), None
                    if not stand_via_loco:
                        state.mode, state.gait = standing_mode, "stand"
                do_reset = state.home_requested or state.reset_requested
                state.home_requested = False
                state.reset_requested = False
                if do_reset:
                    # home/reset both fully stop whatever was driving the robot, so the CPG/dance
                    # policy doesn't immediately walk/dance it away again from the freshly-reset pose.
                    # Exception: a live Stand PPO policy (--stand_policy_checkpoint) doesn't "walk
                    # away" - it just balances in place - and _apply_trained_stand_defaults is
                    # deliberately skipped whenever that flag is set (see main()'s own comment), so
                    # falling back to static "stand" here would hold isaaclab_assets' untrained
                    # open-loop pose/gain instead, which just topples over. Mirrors the same
                    # conditional main() already uses for the very first reset at startup.
                    state.mode, state.gait, state.goto_target = standing_mode, "stand", None
                    state.cmd_vel = (0.0, 0.0, 0.0)
            if stand_via_loco:
                if args_cli.stand_policy_checkpoint:
                    loco_controller.request_stop()
                else:
                    # no Stand PPO to hand over to, and a plain PD hold of the default pose falls over
                    # (verified in sim) - the walking policy at zero velocity *is* the safe stand here
                    print("[g1_isaac_sim_bridge] Stand: no --stand_policy_checkpoint, holding the walking policy at zero velocity")
            if do_reset:
                # bracket the actual physics reset with position printouts so a "reset looks like
                # it did nothing" report can be immediately split into "the sim didn't move the
                # robot" (a real backend bug) vs. "it moved, then something drove it away again /
                # the web UI didn't reflect it" (a client-side or telemetry issue) - just look at
                # this terminal right after clicking Reset.
                before_xy = (robot.data.root_pos_w[0, 0].item(), robot.data.root_pos_w[0, 1].item())
                print(
                    f"[g1_isaac_sim_bridge] Resetting environment now (env.reset(), sim.is_playing()="
                    f"{unwrapped.sim.is_playing()}) - position before: ({before_xy[0]:.2f}, {before_xy[1]:.2f})"
                )
                obs = do_reset_and_align()
                after_xy = (robot.data.root_pos_w[0, 0].item(), robot.data.root_pos_w[0, 1].item())
                print(f"[g1_isaac_sim_bridge] Reset complete - position after: ({after_xy[0]:.2f}, {after_xy[1]:.2f})")
                if stand_policy_controller is not None:
                    stand_policy_controller.reset()

            mode, gait, cmd_vel, goto_target = state.snapshot()
            # joystick ('manual') and go-to-point ('goto') both drive the same locomotion policy -
            # switching between the two doesn't restart its walk-in
            uses_loco = mode in ("manual", "goto")
            # entering first: begin_transition() blends from the gains in effect *right now*, before
            # the Stand policy's restore below overwrites them (its blend then rewrites them anyway)
            if uses_loco and (previous_mode not in ("manual", "goto") or loco_controller.phase == "idle"):
                loco_controller.begin_transition()
            if previous_mode == "policy" and mode != "policy" and stand_policy_controller is not None:
                print("[g1_isaac_sim_bridge] Leaving 'policy' mode - restoring legs/feet/waist actuator gain to default")
                stand_policy_controller.restore_default_gains()
            if not uses_loco and loco_controller.phase != "idle":
                print("[g1_isaac_sim_bridge] Leaving locomotion policy - restoring actuator gain/armature to default")
                loco_controller.restore_default_gains()
            previous_mode = mode
            distance_to_target = None

            with torch.inference_mode():
                if mode == "policy" and args_cli.stand_policy_checkpoint:
                    # PPO standing policy (G1-PPO-Direct-Stand-v0), not the AMP dance one - see
                    # _StandPolicyController's own docstring. Builds the 29-dim AMP-shaped action
                    # itself (pose) and writes gain directly into the actuators as a side effect.
                    action = ensure_stand_policy_loaded().step()
                elif mode == "policy":
                    action = _agent_act(ensure_policy_loaded(), obs["policy"])
                elif uses_loco:
                    if loco_controller.ready_for_handover:
                        ensure_stand_policy_loaded().reset()
                        loco_controller.begin_handover()
                    if loco_controller.phase == "handover":
                        action = loco_controller.handover_step(ensure_stand_policy_loaded().step)
                        if loco_controller.handover_done:
                            with state.lock:
                                if state.mode in ("manual", "goto"):
                                    state.mode, state.gait, state.goto_target = "policy", "stand", None
                    else:
                        if mode == "goto":
                            distance_to_target = drive_loco_to_target(goto_target, gait)
                        else:
                            loco_controller.set_joystick(*cmd_vel)
                        action = loco_controller.step()
                else:
                    # 'stand': zero action = PD hold of the default pose
                    action = torch.zeros(1, len(body_joint_ids), device=device)

                obs, _, _, _, _ = env.step(action)
                # never let the AMP task's fixed episode_length_s timeout auto-reset the robot - the
                # only resets are explicit, via the web UI's home/reset buttons (see do_reset above)
                unwrapped.episode_length_buf[:] = 0

            step_counter += 1
            if step_counter % 5 == 0:  # ~20 Hz telemetry, regardless of physics step rate
                root_pos = robot.data.root_pos_w[0]
                status = {
                    "mode": mode,
                    "gait": gait,
                    "target": list(goto_target) if goto_target else None,
                    "distance_to_target": distance_to_target,
                    "policy_loaded": skrl_agent is not None,
                    # 'policy' mode runs the Stand PPO policy (not the AMP dance) when this is set - the
                    # web UI shows that as Standing
                    "stand_policy": bool(args_cli.stand_policy_checkpoint),
                    "loco": loco_controller.status(),
                    "goto_speed": state.goto_speed,
                    "camera_error": camera_error,
                }
                node.publish_telemetry(
                    (root_pos[0].item() - env_origin[0], root_pos[1].item() - env_origin[1]),
                    root_pos[2].item(),
                    robot.data.root_quat_w[0].tolist(),
                    robot.data.joint_pos[0, body_joint_ids].tolist(),
                    robot.data.joint_vel[0, body_joint_ids].tolist(),
                    status,
                )

            if camera is not None and step_counter % 1 == 0:  # every step (~100 Hz) - raise the modulo to throttle if bandwidth/CPU becomes an issue
                # camera streaming is a nice-to-have on top of the actual robot control loop - never
                # let a hiccup here (render product still warming up, transient encode failure, ...)
                # take down the whole process the way the last two camera-related bugs did.
                try:
                    encoded = _encode_camera_frame(camera, args_cli.camera_jpeg_quality)
                    if encoded is not None:
                        node.publish_camera_frame(*encoded)
                except Exception as exc:
                    print(f"[g1_isaac_sim_bridge] Camera frame skipped ({exc!r})")

            if lidar is not None and step_counter % 10 == 0:  # ~10 Hz - plenty for SLAM, far less DDS/CPU cost than the per-step camera
                try:
                    scan = _encode_lidar_scan(lidar)
                    if scan is not None:
                        node.publish_lidar_scan(scan)
                except Exception as exc:
                    print(f"[g1_isaac_sim_bridge] Lidar scan skipped ({exc!r})")

            sleep_time = step_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
