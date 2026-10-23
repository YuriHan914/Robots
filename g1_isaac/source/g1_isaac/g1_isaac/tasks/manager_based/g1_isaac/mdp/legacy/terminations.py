# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the G1 locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_terrain(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate once the asset's root sinks below ``minimum_height`` above the *local* terrain.

    Same idea as ``isaaclab.envs.mdp.root_height_below_minimum``, except that one compares
    ``root_pos_w[:, 2]`` directly against a world-frame constant, which is only correct on flat
    terrain (see its own docstring). This subtracts each env's terrain origin height
    (``env.scene.env_origins[:, 2]``, the same per-env ground reference the ``terrain_levels_vel``
    curriculum in this package already relies on) first, so it stays correct on the rough terrain
    this task trains on.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_above_terrain = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return height_above_terrain < minimum_height
