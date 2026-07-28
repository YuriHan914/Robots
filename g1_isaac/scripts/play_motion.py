# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play back a retargeted G1 motion capture clip (CSV) on the Unitree G1 robot.

The CSV is expected to have the columns produced by the G1 motion retargeting pipeline:
``Frame``, ``root_translateX/Y/Z`` (cm), ``root_rotateX/Y/Z`` (deg, extrinsic XYZ euler) and one
``<joint_name>_dof`` column (deg) per articulated G1 joint, e.g. ``left_hip_pitch_joint_dof``.

By default the clip is replayed kinematically: every frame the recorded root pose and joint angles
are written directly into the simulation, so the robot reproduces the capture exactly regardless of
balance/contacts. Pass ``--dynamic`` to instead only set the joint angles as PD position targets
(after teleporting once to the clip's starting pose) and let gravity/contacts/actuator dynamics
actually drive the robot - it may fall over if the clip isn't dynamically feasible for G1.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/play_motion.py --motion_file data/dance1_retarget_g1.csv
    ./isaaclab.sh -p scripts/play_motion.py --motion_file data/dance1_retarget_g1.csv --dynamic

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

_DEFAULT_MOTION_FILE = Path(__file__).parent.parent / "data" / "dance1_retarget_g1.csv"

# add argparse arguments
parser = argparse.ArgumentParser(description="Play back a retargeted G1 motion CSV on the Unitree G1 robot.")
parser.add_argument(
    "--motion_file",
    type=str,
    default=str(_DEFAULT_MOTION_FILE),
    help="Path to the retargeted motion CSV file.",
)
parser.add_argument("--fps", type=float, default=30.0, help="Playback frame rate of the motion clip (source capture rate).")
parser.add_argument("--loop", action="store_true", default=False, help="Loop the motion clip instead of holding the last frame.")
parser.add_argument(
    "--dynamic",
    action="store_true",
    default=False,
    help=(
        "Drive the robot with physics instead of teleporting the exact recorded pose every frame: only the "
        "starting pose is set directly, then joint angles are applied as PD position targets and gravity/"
        "contacts/actuator dynamics determine the outcome. The robot may fall over if the clip isn't "
        "dynamically feasible for G1."
    ),
)
parser.add_argument(
    "--physics_hz",
    type=float,
    default=200.0,
    help="Physics simulation rate in Hz, used only in --dynamic mode (motion targets still update at --fps).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import time

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

# G1 with the 5-finger Inspire ("FTP" variant) hand: 29 body joints (matching the retargeting
# target ../GEM-X/third_party/soma-retargeter uses, g1_29dof_rev_1_0) plus 24 finger joints
# (12/hand: index/middle/ring/little x2, thumb x4). Note soma-retargeter itself retargets onto the
# hand-less g1_29dof_rev_1_0 skeleton (a bare wrist with a static rubber-hand mesh, no finger
# joints), so this motion clip never has finger data - see load_motion()'s handling of unmatched
# joints below.
#
# Built from newton-assets' g1_29dof_rev_1_0_with_inspire_hand_FTP URDF, converted to USD via
# IsaacLab's convert_urdf.py and then given full-body collision by
# scripts/tools/add_g1_full_body_collision.py (see that script's docstring for the exact commands
# and provenance). Not committed to git (this repo's .gitignore excludes *.usd); re-run that
# pipeline to regenerate it.
G1_USD_PATH = str(Path(__file__).parent.parent / "assets" / "g1_full_collision.usd")

# G1_29DOF_CFG's "hands" actuator only covers the 3-finger simplified hand (index/middle/thumb);
# G1_USD_PATH's 5-finger Inspire hand adds ring and little fingers, so extend the joint regex or
# Isaac Lab errors at spawn time over joints with no actuator.
_HANDS_ACTUATOR_5FINGER = G1_29DOF_CFG.actuators["hands"].replace(
    joint_names_expr=[".*_index_.*", ".*_middle_.*", ".*_thumb_.*", ".*_ring_.*", ".*_little_.*"],
)


def load_motion(motion_file: str, joint_names: list[str], default_joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load the retargeted motion CSV and reorder joint columns to match the robot's joint order.

    The motion clip only covers the 29 body joints. Any other joints on the robot (e.g. finger
    joints on a hand variant of G1) are held at ``default_joint_pos`` for every frame.

    Returns:
        root_pose: (num_frames, 7) array of (x, y, z, qw, qx, qy, qz) in sim units/radians.
        joint_pos: (num_frames, num_joints) array of joint angles in radians, ordered like ``joint_names``.
    """
    with open(motion_file) as f:
        rows = list(csv.reader(f))
    header, data_rows = rows[0], rows[1:]
    data = np.array(data_rows, dtype=np.float64)
    num_frames = data.shape[0]

    def col(name: str) -> np.ndarray:
        return data[:, header.index(name)]

    # root position: cm -> m
    root_pos = np.stack([col("root_translateX"), col("root_translateY"), col("root_translateZ")], axis=-1) / 100.0
    # root orientation: extrinsic XYZ euler, deg -> rad
    roll = np.deg2rad(col("root_rotateX"))
    pitch = np.deg2rad(col("root_rotateY"))
    yaw = np.deg2rad(col("root_rotateZ"))
    root_quat = quat_from_euler_xyz(torch.from_numpy(roll), torch.from_numpy(pitch), torch.from_numpy(yaw)).numpy()
    root_pose = np.concatenate([root_pos, root_quat], axis=-1)

    # joint angles: match each "<joint_name>_dof" CSV column to the robot's joint order; joints without
    # a matching column (e.g. fingers) are held at their default pose for the whole clip
    dof_columns = {c[: -len("_dof")]: c for c in header if c.endswith("_dof")}
    unmatched = [name for name in joint_names if name not in dof_columns]
    if unmatched:
        print(f"[WARN]: Motion file has no column for joints (holding at default pose): {unmatched}")
    joint_pos = np.tile(default_joint_pos, (num_frames, 1)).astype(np.float64)
    for i, name in enumerate(joint_names):
        if name in dof_columns:
            joint_pos[:, i] = np.deg2rad(col(dof_columns[name]))

    return root_pose.astype(np.float32), joint_pos.astype(np.float32)


def design_scene() -> Articulation:
    """Set up the ground plane, lighting, and G1 robot."""
    # ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    # light
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    # robot - self-collisions disabled by default on G1_29DOF_CFG; enable so the full-body collision
    # geometry in G1_USD_PATH (g1_full_collision.usd) actually stops self-penetration (e.g. hands
    # passing through legs) in --dynamic mode
    robot_cfg = G1_29DOF_CFG.replace(
        prim_path="/World/Robot",
        spawn=G1_29DOF_CFG.spawn.replace(
            usd_path=G1_USD_PATH,
            articulation_props=G1_29DOF_CFG.spawn.articulation_props.replace(enabled_self_collisions=True),
        ),
        actuators={**G1_29DOF_CFG.actuators, "hands": _HANDS_ACTUATOR_5FINGER},
    )
    return Articulation(cfg=robot_cfg)


def run_simulator(
    sim: SimulationContext,
    robot: Articulation,
    root_pose: np.ndarray,
    joint_pos: np.ndarray,
    fps: float,
    loop: bool,
    dynamic: bool,
):
    """Replay the motion clip on the robot.

    In kinematic mode (default) the recorded root pose and joint angles are written directly into
    the simulation every frame, so playback always matches the clip exactly. In dynamic mode only
    the starting pose is teleported; afterwards only PD position targets are updated and physics
    (gravity, contacts, actuator dynamics) determines what the robot actually does.
    """
    num_frames = root_pose.shape[0]
    root_pose_t = torch.from_numpy(root_pose).to(sim.device)
    joint_pos_t = torch.from_numpy(joint_pos).to(sim.device)
    zero_root_vel = torch.zeros((1, 6), device=sim.device)
    zero_joint_vel = torch.zeros((1, joint_pos.shape[1]), device=sim.device)

    frame_dt = 1.0 / fps
    sim_dt = sim.get_physics_dt()
    # number of physics steps to take per motion-clip frame (only matters in dynamic mode)
    substeps = max(1, round(frame_dt / sim_dt))

    def teleport_to_frame(idx: int):
        frame_joint_pos = joint_pos_t[idx : idx + 1]
        robot.write_root_pose_to_sim(root_pose_t[idx : idx + 1])
        robot.write_root_velocity_to_sim(zero_root_vel)
        robot.write_joint_state_to_sim(frame_joint_pos, zero_joint_vel)
        robot.set_joint_position_target(frame_joint_pos)
        robot.write_data_to_sim()

    # in dynamic mode, only the starting pose is set directly; physics takes over from there
    if dynamic:
        teleport_to_frame(0)

    frame_idx = 0
    done = False
    print(f"[INFO]: Playing back {num_frames} frames at {fps} fps (loop={loop}, dynamic={dynamic})...")
    while simulation_app.is_running():
        start_time = time.time()

        if dynamic:
            # only command the PD targets; gravity/contacts/inertia decide the actual motion
            robot.set_joint_position_target(joint_pos_t[frame_idx : frame_idx + 1])
            robot.write_data_to_sim()
            for _ in range(substeps):
                sim.step()
                robot.update(sim_dt)
        else:
            # also update the actuator drive targets to match, otherwise the PD drives fight the
            # kinematic pose we just wrote (their target would still be the stale default pose)
            teleport_to_frame(frame_idx)
            sim.step()
            robot.update(sim_dt)

        if not done:
            frame_idx += 1
            if frame_idx >= num_frames:
                if loop:
                    frame_idx = 0
                    if dynamic:
                        teleport_to_frame(0)
                else:
                    frame_idx = num_frames - 1
                    done = True
                    print("[INFO]: Motion finished, holding last frame. Pass --loop to repeat.")

        sleep_time = frame_dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    """Play back the motion clip on the G1 robot."""
    # in dynamic mode physics needs a fine step size for stable contacts/PD control; in kinematic
    # mode the physics step is irrelevant (state is overwritten every frame) so it just matches --fps
    sim_dt = 1.0 / args_cli.physics_hz if args_cli.dynamic else 1.0 / args_cli.fps
    # default gpu_collision_stack_size (2**26 ~= 67.1M) is too small for G1's full-body self-collision
    # mesh (see design_scene()'s enabled_self_collisions) and overflows under PhysX, dropping contacts;
    # bump it well past the ~69M PhysX reports needing.
    physx_cfg = sim_utils.PhysxCfg(gpu_collision_stack_size=2**27)
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, physx=physx_cfg)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.8])

    robot = design_scene()
    sim.reset()

    default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy()
    root_pose, joint_pos = load_motion(args_cli.motion_file, robot.joint_names, default_joint_pos)

    run_simulator(sim, robot, root_pose, joint_pos, args_cli.fps, args_cli.loop, args_cli.dynamic)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
