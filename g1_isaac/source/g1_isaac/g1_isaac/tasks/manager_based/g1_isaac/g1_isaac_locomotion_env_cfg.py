# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1-PPO-Locomotion-v0: manager-based, whole-body (29 DOF) velocity + target-point locomotion.

Built from ``tasks/direct/g1_isaac/locomotion_example/velocity_env_cfg.py`` - IsaacLab's own generic
manager-based velocity-locomotion template - rather than from any of this repo's existing Direct-workflow
G1 tasks (``tasks/direct/g1_isaac/g1_isaac_env_stand_cfg.py``/``g1_isaac_env_lowerbody_v3_cfg.py`` etc.):
this task is independent of that lineage, imports nothing from it, and is registered under
``tasks/manager_based/g1_isaac/`` instead.

Two deliberate scope choices, relative to that Direct lineage (documented here for the same reason
``g1_isaac_env_base_cfg.py`` documents its own relationship to Stand-v0 - so the choice reads as
intentional, not an oversight):

* **Whole-body (29 DOF) action space** - legs+feet (12) + waist (3) + arms (14), fingers excluded -
  unlike the LowerBody-v0..v3 line (12 DOF, legs+feet only). Matches Stand-v0's joint *coverage*, but
  the PD gains are **fixed** (`isaaclab_assets`' own `G1_29DOF_CFG` defaults, safety-capped knee/ankle
  velocity limits only) rather than Stand's learned pose+gain search - this task's difficulty axis is
  velocity/target tracking, not gain search, same rationale ``g1_isaac_env_lowerbody_cfg.py`` gives for
  its own fixed-gain choice.
* **One continuous velocity command covers stand/walk/run** (`GoalConditionedVelocityCommand`, see
  ``mdp/commands.py``) rather than a discrete gait/phase curriculum. This is a direct, deliberate response
  to a real failure mode recorded in this repo's own ``train_docs/humanoid_stand_walk_run_ppo_design_extended.md``
  (Section 6): a phase-gated stand->walk curriculum collapsed (episode length 37.7s -> 0.41s) the moment
  the whole env population crossed into a velocity range it had never sampled before during the phase-0
  stretch. Sampling the *entire* stand..run range from episode 0 - the same `rel_standing_envs` idiom
  IsaacLab's own `UniformVelocityCommand` already uses for "stand" - sidesteps that failure mode
  structurally rather than tuning a transition window.

`rel_goal_envs=0.0` during training (see `CommandsCfg.base_velocity` below): target-point tracking is a
fixed distance/bearing -> velocity formula (`GoalConditionedVelocityCommand._update_command`), not
something a network needs to learn, so it's applied only at inference time by feeding that formula's
output into the same `velocity_commands` observation the policy was trained on - training itself uses
100% random-velocity envs so stand/walk/run coverage isn't diluted by target-seeking episodes.
"""

from __future__ import annotations

import math
from pathlib import Path

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

_REPO_ROOT = Path(__file__).resolve().parents[6]
G1_LOCAL_USD_PATH = str("/mnt/data/github/Robots/g1_isaac/assets/g1_full_collision.usd")
_ROBOT_BODY_PRIM_PREFIX = "g1_full_collision"
_HANDS_ACTUATOR_5FINGER = G1_29DOF_CFG.actuators["hands"].replace(
    joint_names_expr=[".*_index_.*", ".*_middle_.*", ".*_thumb_.*", ".*_ring_.*", ".*_little_.*"],
)
_LEGS_ACTUATOR_CAPPED_KNEE_VEL = G1_29DOF_CFG.actuators["legs"].replace(
    velocity_limit={**G1_29DOF_CFG.actuators["legs"].velocity_limit, ".*_knee_joint": math.radians(200.0)},
)
_FEET_ACTUATOR_CAPPED_ANKLE_VEL = G1_29DOF_CFG.actuators["feet"].replace(
    velocity_limit={".*_ankle_pitch_joint": math.radians(150.0), ".*_ankle_roll_joint": math.radians(150.0)},
)

# default standing pose for hip/knee/ankle (12 joints), replacing G1_29DOF_CFG's own straight-leg default
# (hip_pitch/knee/ankle_pitch only, -0.10/0.30/-0.20) - same symmetrized, Stand-v0-rollout-derived bent
# stance as g1_isaac_env_lowerbody_v3_cfg.py's identically-named constant (see that file's own comment for
# the full derivation/symmetrization method). Every other joint (waist, arms, fingers) stays at its
# G1_29DOF_CFG default (0.0 rad, unset here). Used as robot_cfg's own init_state.joint_pos below, so this
# is both the pose the robot spawns in AND what every reset scales around (EventCfg.reset_robot_joints ->
# mdp.reset_joints_by_scale reads robot.data.default_joint_pos, which init_state.joint_pos initializes).
_DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.4337,
    "right_hip_pitch_joint": -0.4337,
    "left_hip_roll_joint": 0.0388,
    "right_hip_roll_joint": -0.0388,
    "left_hip_yaw_joint": -0.0310,
    "right_hip_yaw_joint": 0.0310,
    "left_knee_joint": 0.8300,
    "right_knee_joint": 0.8300,
    "left_ankle_pitch_joint": -0.4211,
    "right_ankle_pitch_joint": -0.4211,
    "left_ankle_roll_joint": -0.0421,
    "right_ankle_roll_joint": 0.0421,

    'left_shoulder_pitch_joint': 0.279368,
    'left_shoulder_roll_joint': 0.279225,
    'left_shoulder_yaw_joint': 0.0,
    'left_elbow_joint': 0.0,
    'left_wrist_roll_joint': 0.0,
    'left_wrist_pitch_joint': 0.0,
    'left_wrist_yaw_joint': 0.0,
    
    'right_shoulder_pitch_joint': -0.279368,
    'right_shoulder_roll_joint': -0.279225,
    'right_shoulder_yaw_joint': 0.0,
    'right_elbow_joint': 0.0,
    'right_wrist_roll_joint': 0.0,
    'right_wrist_pitch_joint': 0.0,
    'right_wrist_yaw_joint': 0.0,
}

_BODY_JOINT_PATTERNS = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "waist_.*_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]


##
# Scene definition
##


@configclass
class G1LocomotionSceneCfg(InteractiveSceneCfg):
    """Rough terrain + G1 humanoid, following ``locomotion_example/velocity_env_cfg.py``'s ``MySceneCfg``."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=G1_29DOF_CFG.spawn.replace(
            usd_path=G1_LOCAL_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=G1_29DOF_CFG.spawn.rigid_props.replace(max_depenetration_velocity=0.2),
        ),
        actuators={
            **G1_29DOF_CFG.actuators,
            "hands": _HANDS_ACTUATOR_5FINGER,
            "legs": _LEGS_ACTUATOR_CAPPED_KNEE_VEL,
            "feet": _FEET_ACTUATOR_CAPPED_ANKLE_VEL,
        },
        init_state=G1_29DOF_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.79), joint_pos=_DEFAULT_POSE
        ),
    )

    height_scanner = RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{_ROBOT_BODY_PRIM_PREFIX}/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # one sensor over every body (locomotion_example's convention) - termination/reward terms below select
    # "pelvis" / ".*_ankle_roll_link" out of it via each term's own `SceneEntityCfg(body_names=...)`.
    contact_forces = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{_ROBOT_BODY_PRIM_PREFIX}/.*", history_length=3, track_air_time=True
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""


    base_velocity = mdp.GoalConditionedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        # target-point tracking is not learned here (see mdp/commands.py's distance/bearing -> velocity
        # math - it's a fixed formula, not a network) - it's applied at inference time by feeding the
        # same formula's output into the policy's velocity_commands observation. Training instead uses
        # 100% random-velocity envs so the full stand/walk/run range gets uniform coverage.
        rel_goal_envs=0.0,
        goal_pos_range=(2.0, 8.0),
        goal_decel_gain=2.0,
        goal_arrival_radius=0.3,
        debug_vis=True,
        ranges=mdp.GoalConditionedVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 3.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-2.0, 2.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=_BODY_JOINT_PATTERNS, scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # restricted to the same 29 body joints as ActionsCfg.joint_pos (SceneEntityCfg with no
        # joint_names filter would default to *every* joint on the asset, including the 24 unactuated
        # finger joints held permanently at default - wasted, always-near-constant observation width).
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_BODY_JOINT_PATTERNS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_BODY_JOINT_PATTERNS)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Domain randomization + periodic pushes - same structure as
    ``locomotion_example``/this repo's own Stand/LowerBody event configs.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.5, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Reward terms - reference example's velocity-tracking set, unmodified, plus `feet_slide`.

    `mdp.goal_reached_bonus` still exists in ``mdp/rewards.py`` but isn't wired in here: with
    `rel_goal_envs=0.0` (see ``CommandsCfg.base_velocity``), `is_goal_env` is never True during
    training, so the term would always evaluate to zero - dead weight in the reward sum.
    """

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # scoped to the 29 body joints (see ObservationsCfg.joint_pos's comment) - fingers are never
    # policy-controlled, so their torque/accel shouldn't be part of what the policy is penalized for.
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_BODY_JOINT_PATTERNS)},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_BODY_JOINT_PATTERNS)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee_link"), "threshold": 1.0},
    )
    # penalizes feet moving horizontally while in contact with the ground (skating/dragging) - added
    # after observing left-foot sliding in trained rollouts. Weight matches IsaacLab's reference G1
    # rough_env_cfg.
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # penalizes every step where neither foot touches the ground (both feet airborne at once -
    # hopping/bunny-hopping rather than a proper single-stance walking gait). weight is the penalty's
    # max (applied at zero commanded speed); it's ramped down to zero by the command term's own top
    # speed (ranges.lin_vel_x below) since a real running gait has a legitimate flight phase - see
    # both_feet_airborne's own docstring.
    both_feet_airborne = RewTerm(
        func=mdp.both_feet_airborne,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 1.0,
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis"), "threshold": 1.0},
    # )
    collapsed = DoneTerm(func=mdp.root_height_below_terrain, params={"minimum_height": 0.352})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class G1IsaacLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """See this module's own docstring for the full design rationale."""

    scene: G1LocomotionSceneCfg = G1LocomotionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 60.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**23
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
