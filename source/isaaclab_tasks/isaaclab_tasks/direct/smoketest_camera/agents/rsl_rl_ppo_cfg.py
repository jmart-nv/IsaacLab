# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configuration for the Smoketest Camera environment.

Uses an MLP policy on flattened 32x32 RGB images (3072-dim input) for fast
camera pipeline verification. No CNN — backprop cost is negligible.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class SmoketestCameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 8
    max_iterations = 50
    save_interval = 50
    experiment_name = "smoketest_camera"
    run_name = ""
    logger = "tensorboard"
    resume = False

    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    actor = RslRlMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[128, 64],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )

    critic = RslRlMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )
