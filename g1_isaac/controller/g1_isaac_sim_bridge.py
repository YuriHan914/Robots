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
* ``manual`` - joystick ``cmd_vel`` drives a simple open-loop sinusoidal (CPG) walking gait. No
  trained locomotion policy exists in this repo for arbitrary commanded velocities - only in-place
  motion imitation (AMP dance / ASAP motion-tracking) - so manual/goto driving falls back to a
  hand-tuned gait. Treat the amplitudes/signs in :class:`CpgGait` as a starting point: verify against
  this asset's actual joint-axis conventions in sim and retune before relying on it.
* ``goto``   - given a target ``(x, y)`` and a gait (``walk``/``run``), a proportional controller
  steers heading then forward speed toward the point using the same CPG gait, and drops back to
  ``stand`` on arrival.
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
g1/cmd_vel             g1.CmdVel           manual joystick (linear_x, angular_z used)
g1/goto_command        g1.GotoCommand      x, y (m), gait ("walk"|"run")
g1/goto_cancel         g1.Trigger          cancel an in-progress goto
g1/policy_command      g1.PolicyCommand    command: "play" / "stop"
g1/home_position       g1.Trigger          explicit reset (only way to recover from a fall)
g1/reset_sim           g1.Trigger          same as home_position, triggered by the "초기화" button
g1/emergency_stop      g1.Trigger          drop to ``stand`` immediately (does NOT reset the sim)
g1/robot_pose          g1.RobotPose        world pose of the robot base
g1/joint_states        g1.JointState       body-joint positions/velocities
g1/status              g1.Status           JSON ``{"mode", "gait", "target", ...}``
g1/camera_frame        g1.CameraFrame      JPEG frame from the head camera (base64), see below
=====================  ==================  ==========================================

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
parser.add_argument(
    "--dds_domain_id", type=int, default=0, help="CycloneDDS domain id (must match the web controller)."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--no_camera", action="store_true", default=False, help="Skip mounting the head camera (saves RTX render overhead)."
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
import io
import json
import math
import os
import threading
import time

import gymnasium as gym
import torch
from PIL import Image

from g1_dds_types import (
    TOPIC_CAMERA_FRAME,
    TOPIC_CMD_VEL,
    TOPIC_EMERGENCY_STOP,
    TOPIC_GOTO_CANCEL,
    TOPIC_GOTO_COMMAND,
    TOPIC_HOME_POSITION,
    TOPIC_JOINT_STATES,
    TOPIC_POLICY_COMMAND,
    TOPIC_RESET_SIM,
    TOPIC_ROBOT_POSE,
    TOPIC_STATUS,
    CameraFrame,
    CmdVel,
    GotoCommand,
    JointState,
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
from isaaclab.utils.math import quat_apply
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from skrl.utils.runner.torch import Runner

import g1_isaac.tasks  # noqa: F401

AGENT_CFG_ENTRY_POINT = "skrl_amp_cfg_entry_point"


# ---------------------------------------------------------------------------
# Heuristic CPG walking gait (see module docstring: no trained velocity-command locomotion policy
# exists in this repo, only in-place motion imitation - manual/goto driving needs its own controller).
# ---------------------------------------------------------------------------
class CpgGait:
    """Open-loop sinusoidal gait producing per-joint position offsets (radians) from the default pose."""

    def __init__(self, joint_names: list[str]):
        self._index = {name: i for i, name in enumerate(joint_names)}
        self.num_joints = len(joint_names)
        self.phase = 0.0

    def _set(self, offsets: list[float], name: str, value: float) -> None:
        idx = self._index.get(name)
        if idx is not None:
            offsets[idx] = value

    def step(self, dt: float, forward: float, turn: float, gait: str) -> list[float]:
        """Advance the gait phase by ``dt`` and return this step's joint offsets."""
        offsets = [0.0] * self.num_joints
        speed = abs(forward)
        if gait == "stand" or speed < 0.05:
            self.phase = 0.0
            return offsets

        frequency = (1.0 + 0.6 * speed) if gait == "walk" else (1.6 + 0.8 * speed)
        amplitude = 0.35 if gait == "walk" else 0.55
        self.phase = (self.phase + 2.0 * math.pi * frequency * dt) % (2.0 * math.pi)
        direction = 1.0 if forward >= 0.0 else -1.0
        turn = max(-1.0, min(1.0, turn))
        s_left = math.sin(self.phase)
        s_right = math.sin(self.phase + math.pi)

        # legs: hip pitch swings the leg fwd/back, knee flexes during swing, ankle keeps the foot level
        self._set(offsets, "left_hip_pitch_joint", direction * amplitude * s_left)
        self._set(offsets, "right_hip_pitch_joint", direction * amplitude * s_right)
        self._set(offsets, "left_knee_joint", 0.15 + amplitude * 1.3 * max(0.0, -s_left))
        self._set(offsets, "right_knee_joint", 0.15 + amplitude * 1.3 * max(0.0, -s_right))
        self._set(offsets, "left_ankle_pitch_joint", -direction * amplitude * 0.5 * s_left)
        self._set(offsets, "right_ankle_pitch_joint", -direction * amplitude * 0.5 * s_right)
        # arms swing opposite the same-side leg for balance
        self._set(offsets, "left_shoulder_pitch_joint", -direction * amplitude * 0.6 * s_left)
        self._set(offsets, "right_shoulder_pitch_joint", -direction * amplitude * 0.6 * s_right)
        # turning: small waist/hip-roll bias, no dedicated strafing gait (lateral cmd_vel is ignored)
        self._set(offsets, "waist_yaw_joint", 0.15 * turn)
        self._set(offsets, "left_hip_roll_joint", 0.05 * turn)
        self._set(offsets, "right_hip_roll_joint", -0.05 * turn)
        return offsets


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> float:
    """Yaw (rad) from an Isaac Lab (w, x, y, z) quaternion."""
    w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _goto_command(
    robot_xy: tuple[float, float],
    yaw: float,
    target_xy: tuple[float, float],
    max_speed: float,
    arrive_radius: float = 0.15,
) -> tuple[float, float, float, bool]:
    """Proportional go-to-point steering: face the target, then walk toward it. Returns
    ``(forward, turn, distance, arrived)``."""
    dx, dy = target_xy[0] - robot_xy[0], target_xy[1] - robot_xy[1]
    distance = math.hypot(dx, dy)
    if distance < arrive_radius:
        return 0.0, 0.0, distance, True
    heading_err = math.atan2(dy, dx) - yaw
    heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))
    turn = max(-1.0, min(1.0, 2.0 * heading_err))
    slow_for_turn = max(0.0, 1.0 - abs(heading_err) / (math.pi / 2.0))
    slow_for_arrival = min(1.0, distance / 0.5)
    forward = max_speed * slow_for_turn * slow_for_arrival
    return forward, turn, distance, False


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


def _attach_head_camera(robot, scene, width: int, height: int) -> Camera:
    """Spawn a forward-looking RGB camera on the (single, num_envs=1) robot instance and return the
    live sensor object.

    This G1 USD ships a ``d435_link`` prim - the mount frame for the real robot's head-mounted Intel
    RealSense D435, at head height with a baked-in ~48-degree pitch already tuned by Unitree so its
    local +X axis looks forward out of the robot's face. It's a fixed sub-frame of the torso body
    (no joint of its own), so it never shows up in ``robot.body_names`` - have to check for it
    directly on the USD stage. Falls back to a guessed offset on a head/torso/pelvis body if this
    G1 build doesn't have it.
    """
    from isaaclab.sim.utils.stage import get_current_stage

    mount_body = _find_camera_mount_body(robot.body_names)
    base_path = f"/World/envs/env_0/Robot/{mount_body}"
    d435_path = f"{base_path}/d435_link"

    if get_current_stage().GetPrimAtPath(d435_path).IsValid():
        parent_path = d435_path
        offset = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world")
        mount_desc = f"{mount_body}/d435_link (real RealSense D435 mount)"
    else:
        parent_path = base_path
        # NOTE: guessed offset - assumes this body's local frame points "forward" along +X like the
        # robot root does (see _yaw_from_quat_wxyz). Verify the view actually looks forward once
        # running; if not, adjust `rot` here (convention="world" means forward=+X/up=+Z in the
        # parent body's own local frame).
        offset = CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0), convention="world")
        mount_desc = f"{mount_body} (no d435_link on this G1 build - using a guessed offset)"

    camera_cfg = CameraCfg(
        prim_path=f"{parent_path}/head_cam",
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
_FOOT_SOLE_LOCAL_Z_OFFSET = -0.035


def _align_feet_to_ground(robot, foot_body_ids: list[int], ground_z: float, clearance: float) -> None:
    """Shift env 0's root Z so the lower of the two feet' sole points sits ``clearance`` meters above
    ``ground_z``. Must be called right after a pose has been written to the robot (so
    ``body_pos_w``/``body_quat_w`` reflect it) and from inside ``torch.inference_mode()``, matching
    the context ``env.reset()`` itself runs under."""
    local_offset = torch.tensor([0.0, 0.0, _FOOT_SOLE_LOCAL_Z_OFFSET], device=robot.device)
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
        self.cmd_vel = (0.0, 0.0)  # (forward, turn) - heuristic units, see CpgGait
        self.goto_target: tuple[float, float] | None = None
        self.home_requested = False
        self.reset_requested = False
        self.estop_requested = False

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

        self._pose_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_ROBOT_POSE, RobotPose))
        self._joint_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_JOINT_STATES, JointState))
        self._status_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_STATUS, Status))
        self._camera_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_CAMERA_FRAME, CameraFrame))

        print(
            f"[g1_isaac_sim_bridge] DDS ready on domain {domain_id} - listening on "
            f"{TOPIC_CMD_VEL}, {TOPIC_GOTO_COMMAND}, {TOPIC_POLICY_COMMAND}"
        )

    def _on_cmd_vel(self, msg: CmdVel) -> None:
        with self.state.lock:
            self.state.cmd_vel = (msg.linear_x, msg.angular_z)
            if abs(msg.linear_x) > 0.05 or abs(msg.angular_z) > 0.05:
                self.state.mode = "manual"
                if self.state.gait == "stand":
                    self.state.gait = "walk"
            elif self.state.mode == "manual":
                self.state.mode = "stand"

    def _on_goto_command(self, msg: GotoCommand) -> None:
        with self.state.lock:
            self.state.mode = "goto"
            self.state.gait = msg.gait if msg.gait in ("walk", "run") else "walk"
            self.state.goto_target = (msg.x, msg.y)

    def _on_goto_cancel(self, _msg: Trigger) -> None:
        with self.state.lock:
            if self.state.mode == "goto":
                self.state.mode = "stand"
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
    body_joint_ids = unwrapped.body_joint_ids
    joint_names = [robot.joint_names[i] for i in body_joint_ids]
    action_scale = unwrapped.cfg.action_scale
    step_dt = env.unwrapped.step_dt
    foot_body_ids, _ = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"])
    env_origin_z = unwrapped.scene.env_origins[0, 2].item()

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
    cpg = CpgGait(joint_names)

    camera = (
        None
        if args_cli.no_camera
        else _attach_head_camera(robot, unwrapped.scene, args_cli.camera_width, args_cli.camera_height)
    )

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

    obs = do_reset_and_align()
    env_origin = unwrapped.scene.env_origins[0, :2].tolist()

    step_counter = 0
    try:
        while simulation_app.is_running():
            loop_start = time.time()

            with state.lock:
                do_estop = state.estop_requested
                state.estop_requested = False
                if do_estop:
                    state.mode, state.gait, state.goto_target = "stand", "stand", None
                    state.cmd_vel = (0.0, 0.0)
                do_reset = state.home_requested or state.reset_requested
                state.home_requested = False
                state.reset_requested = False
                if do_reset:
                    # home/reset both fully stop whatever was driving the robot, so the CPG/policy
                    # doesn't immediately walk it away again from the freshly-reset pose
                    state.mode, state.gait, state.goto_target = "stand", "stand", None
                    state.cmd_vel = (0.0, 0.0)
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
                cpg.phase = 0.0

            mode, gait, cmd_vel, goto_target = state.snapshot()
            distance_to_target = None

            with torch.inference_mode():
                if mode == "policy":
                    action = _agent_act(ensure_policy_loaded(), obs["policy"])
                else:
                    if mode == "goto" and goto_target is not None:
                        root_xy = (
                            robot.data.root_pos_w[0, 0].item() - env_origin[0],
                            robot.data.root_pos_w[0, 1].item() - env_origin[1],
                        )
                        yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[0])
                        max_speed = 0.6 if gait == "walk" else 1.2
                        forward, turn, distance_to_target, arrived = _goto_command(
                            root_xy, yaw, goto_target, max_speed
                        )
                        if arrived:
                            with state.lock:
                                state.mode, state.goto_target = "stand", None
                            gait_for_cpg = "stand"
                        else:
                            gait_for_cpg = gait
                    elif mode == "manual":
                        forward, turn = cmd_vel
                        gait_for_cpg = gait
                    else:
                        forward, turn, gait_for_cpg = 0.0, 0.0, "stand"

                    offsets = cpg.step(step_dt, forward, turn, gait_for_cpg)
                    action = torch.tensor(
                        [[offset / action_scale for offset in offsets]], device=device, dtype=torch.float32
                    )

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
