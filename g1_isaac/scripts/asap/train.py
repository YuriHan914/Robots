# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train the G1 ASAP-style motion-tracking task with a self-contained PPO implementation.

Unlike ``scripts/skrl/train.py`` (skrl's AMP agent, discriminator-driven style reward), this trains
``Template-G1-Isaac-ASAP-Motion-Direct-v0`` (``g1_isaac_asap_env.G1AsapMotionEnv``) with hand-crafted
motion-tracking rewards and the PPO algorithm in ``scripts/asap/algo`` - ported from ASAP
(https://github.com/LeCAR-Lab/ASAP)'s ``humanoidverse.agents.ppo.ppo.PPO``, not skrl's PPO/AMP agent.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train the G1 ASAP-style motion-tracking task with PPO.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-G1-Isaac-ASAP-Motion-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and PPO.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time
from datetime import datetime

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import g1_isaac.tasks  # noqa: F401

from algo.actor_critic import ActorCritic
from algo.ppo import PPO

AGENT_CFG_ENTRY_POINT = "asap_cfg_entry_point"


@hydra_task_config(args_cli.task, AGENT_CFG_ENTRY_POINT)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """Train the ASAP-style motion-tracking policy."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # self-collisions off during training: cheaper, and not needed by the tracking reward - see
    # scripts/asap/play.py, which turns them back on for evaluation (same convention as scripts/skrl/*).
    env_cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = False

    if args_cli.max_iterations:
        agent_cfg["runner"]["num_iterations"] = args_cli.max_iterations

    # randomly sample a seed if seed == -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    agent_cfg["seed"] = seed
    env_cfg.seed = seed
    random.seed(seed)
    torch.manual_seed(seed)

    # specify directory for logging experiments: logs/asap/<experiment_name>/<time-stamp>
    log_root_path = os.path.abspath(os.path.join("logs", "asap", agent_cfg["runner"]["experiment_name"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir_path = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir_path

    # dump the configuration into the log directory
    dump_yaml(os.path.join(log_dir_path, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir_path, "params", "agent.yaml"), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir_path, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    num_actions = env.unwrapped.num_body_joints

    # probe actual actor/critic observation dims from the env itself (see G1AsapMotionEnv._get_observations)
    obs, extras = env.reset()
    num_actor_obs = obs["policy"].shape[-1]
    num_critic_obs = extras["observations"]["critic"].shape[-1]
    print(f"[INFO] actor obs dim: {num_actor_obs}, critic obs dim: {num_critic_obs}, actions: {num_actions}")

    net_cfg = agent_cfg["network"]
    algo_cfg = agent_cfg["algorithm"]
    runner_cfg = agent_cfg["runner"]

    actor_critic = ActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        hidden_dims=net_cfg["hidden_dims"],
        activation=net_cfg["activation"],
        init_noise_std=net_cfg["init_noise_std"],
    )
    ppo = PPO(
        actor_critic=actor_critic,
        num_envs=num_envs,
        num_steps_per_env=algo_cfg["num_steps_per_env"],
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        device=device,
        num_learning_epochs=algo_cfg["num_learning_epochs"],
        num_mini_batches=algo_cfg["num_mini_batches"],
        clip_param=algo_cfg["clip_param"],
        gamma=algo_cfg["gamma"],
        lam=algo_cfg["lam"],
        value_loss_coef=algo_cfg["value_loss_coef"],
        entropy_coef=algo_cfg["entropy_coef"],
        actor_learning_rate=algo_cfg["actor_learning_rate"],
        critic_learning_rate=algo_cfg["critic_learning_rate"],
        max_grad_norm=algo_cfg["max_grad_norm"],
        use_clipped_value_loss=algo_cfg["use_clipped_value_loss"],
        schedule=algo_cfg["schedule"],
        desired_kl=algo_cfg["desired_kl"],
    )

    start_iteration = 0
    if args_cli.checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args_cli.checkpoint}")
        state = ppo.load(args_cli.checkpoint)
        start_iteration = state.get("iteration", 0)

    writer = SummaryWriter(log_dir=log_dir_path)
    num_steps_per_env = algo_cfg["num_steps_per_env"]
    num_iterations = runner_cfg["num_iterations"]
    save_interval = runner_cfg["save_interval"]

    actor_obs, critic_obs = obs["policy"], extras["observations"]["critic"]
    start_time = time.time()
    for it in range(start_iteration, num_iterations):
        episode_returns = torch.zeros(num_envs, device=device)
        completed_returns = []
        # per-cause termination counts for this rollout (see G1AsapMotionEnv._get_dones's
        # extras["termination"]) - helps distinguish "falling", "drifted too far from the reference
        # motion", and "survived to the episode timeout" when diagnosing why reward/curriculum stall.
        fell_count = 0
        motion_far_count = 0
        timeout_count = 0
        with torch.inference_mode():
            for _ in range(num_steps_per_env):
                actions = ppo.act(actor_obs, critic_obs)
                obs, rewards, terminated, timeouts, extras = env.step(actions)
                dones = terminated | timeouts
                episode_returns += rewards
                if torch.any(dones):
                    completed_returns.append(episode_returns[dones].clone())
                    episode_returns[dones] = 0.0
                term_info = extras.get("termination")
                if term_info is not None:
                    fell_count += int(term_info["fell"].sum().item())
                    motion_far_count += int(term_info["motion_far"].sum().item())
                timeout_count += int(timeouts.sum().item())
                ppo.process_env_step(rewards, dones, timeouts)
                actor_obs, critic_obs = obs["policy"], extras["observations"]["critic"]
            ppo.compute_returns(critic_obs)

        stats = ppo.update()

        writer.add_scalar("loss/surrogate", stats["surrogate_loss"], it)
        writer.add_scalar("loss/value", stats["value_loss"], it)
        writer.add_scalar("policy/entropy", stats["entropy"], it)
        writer.add_scalar("policy/actor_lr", stats["actor_lr"], it)
        writer.add_scalar("curriculum/factor", env.unwrapped._curriculum_factor, it)
        writer.add_scalar("termination/fell", fell_count, it)
        writer.add_scalar("termination/motion_far", motion_far_count, it)
        writer.add_scalar("termination/timeout", timeout_count, it)
        if completed_returns:
            mean_return = torch.cat(completed_returns).mean().item()
            writer.add_scalar("reward/mean_episode_return", mean_return, it)
        else:
            mean_return = float("nan")

        elapsed = time.time() - start_time
        print(
            f"[Iter {it:5d}/{num_iterations}] surrogate={stats['surrogate_loss']:.4f} "
            f"value={stats['value_loss']:.4f} entropy={stats['entropy']:.4f} "
            f"return={mean_return:.3f} curriculum={env.unwrapped._curriculum_factor:.2f} "
            f"elapsed={elapsed:.0f}s"
        )

        if it % save_interval == 0 or it == num_iterations - 1:
            ckpt_dir = os.path.join(log_dir_path, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"model_{it}.pt")
            ppo.save(ckpt_path, extra={"iteration": it})
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    writer.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
