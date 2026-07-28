# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# G1 asset served directly from the Isaac Sim asset library (not the IsaacLab-specific nucleus copy)
G1_USD_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/g1.usd"

# repo root: source/g1_isaac/g1_isaac/tasks/direct/g1_isaac/g1_isaac_env_cfg.py -> parents[6]
_REPO_ROOT = Path(__file__).resolve().parents[6]
# same locally-generated, full-body-collision G1 asset used by scripts/play_motion.py (see that script's
# comments and scripts/tools/add_g1_full_body_collision.py); not committed to git (*.usd is gitignored)
G1_LOCAL_USD_PATH = str(_REPO_ROOT / "assets" / "g1_full_collision.usd")
# retargeted mocap clip played back by scripts/play_motion.py, reused here as the AMP reference motion
DEFAULT_MOTION_FILE = str(_REPO_ROOT / "data" / "dance1_retarget_g1.csv")

# G1_29DOF_CFG's "hands" actuator only covers the 3-finger simplified hand (index/middle/thumb).
# G1_LOCAL_USD_PATH's asset has the 5-finger Inspire ("FTP" variant) hand instead, which adds
# ring and little/pinky fingers - extend the joint regex so those joints get an actuator too
# (Isaac Lab errors at spawn time if any joint on the articulation isn't covered by one).
_HANDS_ACTUATOR_5FINGER = G1_29DOF_CFG.actuators["hands"].replace(
    joint_names_expr=[".*_index_.*", ".*_middle_.*", ".*_thumb_.*", ".*_ring_.*", ".*_little_.*"],
)


@configclass
class G1IsaacEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    # 29 DOF: legs (2x4) + feet (2x2) + waist (3) + arms (2x7)
    action_space = 29
    # joint pos (29) + joint vel (29) + projected gravity (3)
    observation_space = 61
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=G1_29DOF_CFG.spawn.replace(usd_path=G1_USD_PATH),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - action scale (joint position target offset from default pose, in radians)
    action_scale = 0.5
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_upright = -1.0
    rew_scale_joint_deviation = -0.01
    # - reset states/conditions
    min_base_height = 0.5  # terminate if the base falls below this height [m]


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    """G1 imitation-learning (AMP) environment: style reward drives the policy to mimic a mocap clip."""

    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    # 29 DOF: legs (2x4) + feet (2x2) + waist (3) + arms (2x7)
    action_space = 29
    # AMP/policy observation: dof pos (29) + dof vel (29) + root height (1) + root orientation
    # tangent/normal (6) + root linear vel (3) + root angular vel (3)
    observation_space = 71
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 71

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # robot(s) - same locally-generated, full-body-collision asset used by scripts/play_motion.py
    # G1_29DOF_CFG ships with enabled_self_collisions=False. scripts/skrl/train.py and play.py
    # override this explicitly (False and True respectively) after this cfg is loaded, so the
    # default here is left untouched.
    robot_cfg: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=G1_29DOF_CFG.spawn.replace(usd_path=G1_LOCAL_USD_PATH),
        actuators={**G1_29DOF_CFG.actuators, "hands": _HANDS_ACTUATOR_5FINGER},
        # G1_29DOF_CFG's default spawn height (0.75) was tuned against the 3-finger hand asset's
        # sparser foot collision (a few hand-placed contact spheres). The 5-finger asset's feet use
        # a full SDF mesh collider (see scripts/tools/add_g1_full_body_collision.py) that extends
        # ~2.8cm lower at this same default pose (right_ankle_roll_link measured at world Z
        # -0.0276 with the old 0.75 spawn height), so the robot used to spawn with its feet already
        # penetrating the ground. Raised by 0.04 for ~1.2cm of clearance margin.
        init_state=G1_29DOF_CFG.init_state.replace(pos=(0.0, 0.0, 0.79)),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - action scale (joint position target offset from default pose, in radians)
    action_scale = 0.5
    # - reference motion (retargeted mocap clip, see scripts/play_motion.py)
    motion_file: str = DEFAULT_MOTION_FILE
    motion_fps: float = 30.0
    # - reset states/conditions
    early_termination = True
    termination_height = 0.5  # terminate if the base falls below this height [m]
    reset_strategy = "random-start"  # default, random, random-start
    """Strategy to be followed when resetting each environment (G1's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling the reference motion at random, uniform times.
    * random-start: pose and joint states are set by sampling the reference motion at the start (time zero).

    Defaults to "random-start" (the same clip frame ``scripts/play_motion.py`` starts from, verified
    to sit above the ground) rather than "random": since the AMP policy only tracks the reference
    motion loosely (via the discriminator's style reward, not direct PD tracking of the clip), resetting
    into an arbitrary mid-clip pose right before the untrained policy's near-default action snaps the
    joint targets back to ``default_joint_pos`` can produce a large, fast joint-angle jump that drives
    the feet through the (box-collider-approximated) ground on the very first control step.
    """
    reset_root_height_offset = 0.02  # lift applied to the reference motion's root height on reset [m]
    """Extra vertical clearance added to the reference motion's root height when teleporting the robot
    into a reset pose, to avoid spawning with (box-collider-approximated) geometry already penetrating
    the ground plane - see ``reset_strategy``.
    """


# G1 body joints tracked by the retargeted mocap clip(s) (see MotionLoader/MotionLibrary) - same count as
# G1AmpEnvCfg.action_space; finger joints are excluded from the action/tracking space and always held at
# their default pose (see G1AsapMotionEnv._apply_action).
_ASAP_NUM_BODY_JOINTS = 29
_ASAP_HISTORY_LENGTH = 4
# actor ("policy") observation, one frame: base ang vel (3) + projected gravity (3) + dof pos (N) +
# dof vel (N) + last action (N) + reference-motion phase (1) - see G1AsapMotionEnv._compute_actor_feat.
# No base linear velocity and no tracking-error terms (sim2real-friendly, matches ASAP's actor obs design).
_ASAP_ACTOR_FEAT_DIM = 3 + 3 + 3 * _ASAP_NUM_BODY_JOINTS + 1
# stacked with `_ASAP_HISTORY_LENGTH` past frames (see G1AsapMotionEnv's CircularBuffer-backed history)
_ASAP_ACTOR_OBS_DIM = _ASAP_ACTOR_FEAT_DIM * (1 + _ASAP_HISTORY_LENGTH)


@configclass
class G1AsapEnvCfg(DirectRLEnvCfg):
    """G1 motion-tracking environment, ported from ASAP (https://github.com/LeCAR-Lab/ASAP)'s
    ``humanoidverse.envs.motion_tracking`` task onto Isaac Lab.

    Unlike ``G1AmpEnvCfg`` (style reward from a learned discriminator), this task uses hand-crafted
    DeepMimic-style tracking rewards computed directly against the reference motion (see
    ``G1AsapMotionEnv``), plus an asymmetric actor/critic observation split, a reward-penalty curriculum,
    and (optionally) multiple reference clips - reproducing the core pieces of ASAP's motion-tracking
    formulation on top of Isaac Lab's own ``Articulation``/``DirectRLEnv`` primitives (no port of ASAP's
    own MJCF/skeleton-based simulator/motion-library abstraction).
    """

    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition (see `_ASAP_*` derivations above)
    action_space = _ASAP_NUM_BODY_JOINTS
    observation_space = _ASAP_ACTOR_OBS_DIM
    state_space = 0  # critic ("privileged") observation is NOT declared here - see G1AsapMotionEnv,
    # which instead exposes it through `self.extras["observations"]["critic"]` (same idiom G1AmpEnv
    # already uses for `amp_obs`), since its size depends on the robot USD's body count (only known once
    # the asset is spawned in sim, unlike `observation_space`/`action_space` which must be static ints).
    history_length = _ASAP_HISTORY_LENGTH

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # robot(s) - same locally-generated, full-body-collision asset as G1AmpEnvCfg
    robot_cfg: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=G1_29DOF_CFG.spawn.replace(usd_path=G1_LOCAL_USD_PATH),
        actuators={**G1_29DOF_CFG.actuators, "hands": _HANDS_ACTUATOR_5FINGER},
        init_state=G1_29DOF_CFG.init_state.replace(pos=(0.0, 0.0, 0.79)),  # see G1AmpEnvCfg's comment
    )
    # a second, non-simulated instance of the same asset used purely to compute forward kinematics for
    # the reference motion (its joints/root are teleported to the interpolated reference-motion state
    # every step via write_joint_state_to_sim/write_root_pose_to_sim + SimulationContext.forward(), which
    # runs Isaac Sim's own kinematic-only articulation update - no dynamics step - so body_pos_w/body_quat_w/
    # body_lin_vel_w/body_ang_vel_w on this asset give the reference body/keypoint state without porting
    # ASAP's own MJCF-based FK engine). Gravity and collisions are disabled so it never interacts physically
    # with the real robot, the ground, or (in the GPU pipeline) other environments' reference robots.
    ref_robot_cfg: ArticulationCfg = robot_cfg.replace(
        prim_path="/World/envs/env_.*/RobotRef",
        spawn=robot_cfg.spawn.replace(
            rigid_props=robot_cfg.spawn.rigid_props.replace(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    # scene - fewer parallel envs than G1AmpEnvCfg by default: each env now hosts two articulations
    # (policy robot + reference robot), roughly doubling per-env memory/compute cost.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - action scale (joint position target offset from default pose, in radians)
    action_scale = 0.5
    # - reference motion: a single CSV file, or a directory of CSVs for multi-clip training (see
    #   motions.resolve_motion_files/MotionLibrary). Defaults to the single clip used by the AMP task.
    motion_file: str = DEFAULT_MOTION_FILE
    motion_fps: float = 30.0
    # whether each env is assigned a reference clip by weighted-random sampling (training) or
    # deterministic round-robin (eval) - see G1AsapMotionEnv.__init__ / MotionLibrary.load_motions.
    deterministic_motion_assignment: bool = False
    # how often (in seconds of sim time) each env's assigned clip is re-sampled during training, matching
    # ASAP's `resample_time_interval_s` (kept large: mainly matters once >1 clip is present in motion_file).
    motion_resample_interval_s: float = 2000.0
    # body names (regex) tracked separately as "feet" for the tighter-sigma feet-position reward term
    feet_body_names: list[str] = ["(left|right)_ankle_roll_link"]
    # body names (regex) tracked separately as ASAP's "3-point" VR/keypoint reward (hands + head), with
    # its own tight sigma/higher weight - approximated with the real end-effector-ish links closest to
    # ASAP's virtual hand/head markers (no `extend_config`-style fixed-offset virtual bodies are added).
    # Empty/unmatched entries are dropped with a startup warning instead of erroring (the term then just
    # contributes 0), since exact G1 USD body names vary by hand variant/build - verify against
    # `env.unwrapped.robot.body_names` and adjust if these don't match your asset. This asset is the
    # Inspire 5-finger ("FTP") hand build (no `rubber_hand_link`); `middle_1` (the middle finger's
    # proximal link, rigidly close to the palm) stands in for the missing palm/hand link.
    keypoint_body_names: list[str] = ["(left|right)_wrist_yaw_link", "(left|right)_middle_1"]

    # - reward weights/sigmas (DeepMimic-style exp(-error / sigma) tracking rewards, proportions carried
    #   over from ASAP's reward_motion_tracking_dm_2real.yaml; penalty terms are ASAP's generic
    #   legged_robot_base penalties, trimmed to the ones that don't need foot contact-force sensing)
    reward_body_pos_weight = 1.0
    reward_body_pos_sigma = 0.05
    reward_feet_pos_weight = 1.0
    reward_feet_pos_sigma = 0.03
    reward_keypoint_pos_weight = 1.6
    reward_keypoint_pos_sigma = 0.03
    reward_body_rot_weight = 0.5
    reward_body_rot_sigma = 1.0
    reward_body_lin_vel_weight = 0.5
    reward_body_lin_vel_sigma = 1.0
    reward_body_ang_vel_weight = 0.5
    reward_body_ang_vel_sigma = 1.0
    reward_joint_pos_weight = 0.75
    reward_joint_pos_sigma = 1.0
    reward_joint_vel_weight = 0.5
    reward_joint_vel_sigma = 1.0
    # - penalty terms (all scaled by the reward-penalty curriculum, see below), ASAP's
    #   legged_robot_base.py generic penalties (`_reward_penalty_torques`/`_reward_limits_*`)
    reward_action_rate_weight = -0.5
    reward_torque_weight = -1.0e-6
    reward_dof_pos_limit_weight = -10.0
    reward_dof_vel_limit_weight = -5.0
    reward_torque_limit_weight = -5.0
    # fraction of the hard joint velocity/torque limits below which no penalty is applied (ASAP's 0.9/0.825)
    dof_vel_limit_margin = 0.9
    torque_limit_margin = 0.825
    # not curriculum-scaled (safety-critical, always fully applied)
    reward_termination_weight = -20.0

    # - reset states/conditions
    early_termination = True
    termination_height = 0.5  # terminate if the base falls below this height [m]
    # terminate if any tracked body strays more than this far from the reference motion [m]. Curriculum'd:
    # starts loose (so an untrained policy can survive long enough to learn) and tightens toward `_end` as
    # `_curriculum_factor` -> 1, matching ASAP's `terminate_when_motion_far` curriculum.
    termination_motion_far_threshold_start = 1.0
    termination_motion_far_threshold_end = 0.25
    reset_strategy = "random"  # default, random, random-start (see G1AmpEnvCfg for the semantics)
    reset_root_height_offset = 0.02

    # - reward-penalty curriculum: a single scalar factor in [0, 1] that scales the action-rate penalty
    #   (and tightens `termination_motion_far_threshold`), ramped up/down based on a running average of
    #   episode length vs. a target fraction of the max episode length - a trimmed version of ASAP's
    #   `_update_reward_penalty_curriculum` (legged_robot_base.py).
    curriculum_initial_factor = 0.0
    curriculum_min_factor = 0.0
    curriculum_increment = 0.02
    curriculum_target_episode_length_frac = 0.8
    curriculum_episode_length_ema_alpha = 0.05