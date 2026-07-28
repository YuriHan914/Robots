# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play back a checkpoint trained by ``scripts/asap/train.py`` on the ASAP-style G1
motion-tracking task (``Template-G1-Isaac-ASAP-Motion-Direct-v0``)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of the G1 ASAP-style motion-tracking policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-G1-Isaac-ASAP-Motion-Direct-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import g1_isaac.tasks  # noqa: F401

from algo.actor_critic import ActorCritic

AGENT_CFG_ENTRY_POINT = "asap_cfg_entry_point"


@hydra_task_config(args_cli.task, AGENT_CFG_ENTRY_POINT)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """Play with the ASAP-style motion-tracking policy."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # self-collisions on for evaluation, so the full-body collision geometry actually stops
    # self-penetration - see scripts/asap/train.py, which keeps them off during training.
    env_cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = True
    # deterministic round-robin clip assignment (instead of duration-weighted random sampling) and
    # start every env at time zero, so playback is reproducible - see MotionLibrary.load_motions.
    env_cfg.deterministic_motion_assignment = True
    env_cfg.reset_strategy = "random-start"

    env_cfg.seed = args_cli.seed if args_cli.seed is not None else env_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "asap", agent_cfg["runner"]["experiment_name"]))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, other_dirs=["checkpoints"])
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    device = env.unwrapped.device
    num_actions = env.unwrapped.num_body_joints

    obs, extras = env.reset()
    num_actor_obs = obs["policy"].shape[-1]
    num_critic_obs = extras["observations"]["critic"].shape[-1]

    net_cfg = agent_cfg["network"]
    actor_critic = ActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        hidden_dims=net_cfg["hidden_dims"],
        activation=net_cfg["activation"],
        init_noise_std=net_cfg["init_noise_std"],
    ).to(device)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    state = torch.load(resume_path, map_location=device)
    actor_critic.load_state_dict(state["model_state_dict"])
    actor_critic.eval()

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = actor_critic.act_inference(obs["policy"])
            obs, _, _, _, extras = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
