# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FinetuneBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Conservative fine-tuning configuration for yaw turning capability."""
    
    # REDUCED ITERATIONS: Fine-tuning from existing checkpoint
    num_steps_per_env = 24
    max_iterations = 5000  # Much smaller than original 50000 for fine-tuning
    save_interval = 50     # More frequent saves for monitoring
    experiment_name = "finetune_yaw"
    empirical_normalization = False
    
    # CONSERVATIVE LEARNING: Lower learning rate and reduced network changes
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Reduced from 1.0 for more stable fine-tuning
        actor_hidden_dims=[512, 256, 128],  # Same architecture as original
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,        # Slightly more conservative than 0.2
        entropy_coef=0.005,     # Reduced from 0.01 for more deterministic behavior
        num_learning_epochs=3,  # Reduced from 5 for gentler updates
        num_mini_batches=4,
        learning_rate=5.0e-4,   # Half the original learning rate for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,       # Slightly lower than 0.01 for conservative updates
        max_grad_norm=1.0,
    )