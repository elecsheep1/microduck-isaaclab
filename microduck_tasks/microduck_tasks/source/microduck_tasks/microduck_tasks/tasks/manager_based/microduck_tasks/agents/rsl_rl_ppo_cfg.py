# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg, 
    RslRlOnPolicyRunnerCfg, 
    RslRlPpoAlgorithmCfg, 
    RslRlSymmetryCfg,
)

from ..mdp import symmetry


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 2000
    save_interval = 100
    experiment_name = "microduck_stand"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,

        symmetry_cfg=RslRlSymmetryCfg(
            # 启用数据增强：每个 PPO mini-batch 增加一份左右镜像样本。
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=symmetry.compute_symmetry_states,
        ),
    )


@configclass
class PPODeployableWalkRunnerCfg(PPORunnerCfg):
    # actor 只读取真机可以构造的 policy 观测。
    # critic 在训练时额外读取仿真真实的机身线速度。
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy", "privileged"],
    }