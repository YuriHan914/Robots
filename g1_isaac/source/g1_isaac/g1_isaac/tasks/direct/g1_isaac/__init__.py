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
    id="Template-G1-Isaac-Direct-v0",
    entry_point=f"{__name__}.g1_isaac_env:G1IsaacEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_isaac_env_cfg:G1IsaacEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-G1-Isaac-AMP-Dance-Direct-v0",
    entry_point=f"{__name__}.g1_isaac_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_isaac_env_cfg:G1AmpEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)

gym.register(
    id="Template-G1-Isaac-ASAP-Motion-Direct-v0",
    entry_point=f"{__name__}.g1_isaac_asap_env:G1AsapMotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_isaac_env_cfg:G1AsapEnvCfg",
        # consumed directly (as a plain dict, via isaaclab_tasks' hydra_task_config) by
        # scripts/asap/train.py and play.py's own PPO implementation - not skrl's.
        "asap_cfg_entry_point": f"{agents.__name__}:asap_motion_cfg.yaml",
    },
)
