# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-G1-Isaac-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_isaac_env_cfg:G1IsaacEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# manager-based G1 locomotion: whole-body (29 DOF) stand/walk/run velocity tracking + target-point
# ("go to (x, y)") navigation, built from tasks/direct/g1_isaac/locomotion_example (IsaacLab's own generic
# manager-based velocity-locomotion template) - independent of this repo's Direct-workflow G1 task lineage
# (tasks/direct/g1_isaac/*). See g1_isaac_locomotion_env_cfg.py's module docstring for the full design.
gym.register(
    id="G1-PPO-Locomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_isaac_locomotion_env_cfg:G1IsaacLocomotionEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_locomotion_cfg.yaml",
    },
)

# Fine-tune 환경들 (yaw turning 특화)
gym.register(
    id="Unitree-G1-29dof-Velocity-Finetune",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": velocity_env_cfg_finetune.RobotEnvCfgFinetune,
        "play_env_cfg_entry_point": velocity_env_cfg_finetune.RobotPlayEnvCfgFinetune,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_finetune:FinetuneBasePPORunnerCfg",
    },
    disable_env_checker=True,
)
