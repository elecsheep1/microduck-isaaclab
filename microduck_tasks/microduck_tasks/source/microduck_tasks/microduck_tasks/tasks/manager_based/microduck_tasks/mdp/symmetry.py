"""Microduck 左右镜像的数据增强映射。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from ..microduck_walk_params import (
    STANDING_LINEAR_SPEED_THRESHOLD,
    STANDING_YAW_RATE_THRESHOLD,
)


@torch.no_grad()
def compute_symmetry_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """生成原始样本及其左右镜像样本。

    返回的第一个 batch 是原始数据，第二个 batch 是镜像数据。
    """

    if obs is not None:
        # obs 是一批原始观测。
        # 例如：
        #   obs.batch_size = [N]
        #   N 表示并行环境数量，也就是当前 batch 中有 N 条样本。
        batch_size = obs.batch_size[0]

        # repeat(2) 会把 batch 维度扩大为 2N。
        # 前 N 条保存原始样本，后 N 条保存镜像样本。
        # 注意：这里复制的是 TensorDict 中的张量数据，不是简单复制一个引用。
        obs_aug = obs.repeat(2)

        # policy 是 actor 使用的观测，形状通常为 [N, 49]。
        # 前半部分放原始观测，后半部分放左右镜像后的观测。
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = _mirror_policy_obs(obs["policy"])

        # privileged 是 critic 使用的额外观测。
        # 当前配置中它只有 base_lin_vel，形状通常为 [N, 3]。
        # 如果当前训练没有 privileged 观测，就不处理这一项。
        if "privileged" in obs:
            obs_aug["privileged"][:batch_size] = obs["privileged"]
            obs_aug["privileged"][batch_size:] = _mirror_privileged_obs(obs["privileged"])
    else:
        obs_aug = None

    if actions is not None:
        # actions 通常形状为 [N, 10]。
        # cat(dim=0) 沿 batch 维拼接：
        #   前 N 行：原始动作
        #   后 N 行：镜像动作
        actions_aug = torch.cat(
            (actions, _mirror_actions(actions)),
            dim=0,
        )
    else:
        actions_aug = None

    return obs_aug, actions_aug

def _mirror_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    """镜像 49 维 actor 观测。

    当前 policy 观测布局：

    [0:3]   base_ang_vel
    [3:6]   velocity_command
    [6:8]   gait_phase = [sin(phi), cos(phi)]
    [8:11]  projected_gravity
    [11:25] joint_pos (14)
    [25:39] joint_vel (14)
    [39:49] last_action (10)
    """

    # clone() 创建一份独立副本，避免修改原始观测 obs。
    # obs 的形状通常是 [N, 49]。
    mirrored = obs.clone()
    device = obs.device
    dtype = obs.dtype

    # 镜像 y-z 平面，相当于把左右方向 y 反过来。
    # 角速度是轴向量，不同坐标分量在镜像后遵循：
    #   wx 变号，wy 保持，wz 变号
    # 因此乘以 [-1, 1, -1]。
    mirrored[:, 0:3] = obs[:, 0:3] * torch.tensor(
        [-1.0, 1.0, -1.0],
        device=device,
        dtype = dtype,
    )

    # 速度指令的顺序是 [vx, vy, yaw_rate]：
    #   vx：前后方向，镜像后保持不变
    #   vy：左右方向，镜像后变号
    #   yaw_rate：左转/右转方向，镜像后变号
    mirrored[:, 3:6] = obs[:, 3:6] * torch.tensor(
        [1.0, -1.0, -1.0],
        device=device,
        dtype=dtype,
    )

    # 左右镜像时，左脚和右脚的角色互换。
    # 因此步态相位需要平移半个周期：phi -> phi + pi。
    # 对相位编码 [sin(phi), cos(phi)] 来说：
    #   sin(phi + pi) = -sin(phi)
    #   cos(phi + pi) = -cos(phi)
    # 所以两个分量都取负号。

    # 行走时，左右镜像对应相位平移半周期。
    mirrored_phase = -obs[:, 6:8]

    # standing 时相位固定为 [0, 1]，左右镜像后也应保持不变。
    #
    # 观测内 [3:6] 是 [vx, vy, yaw_rate]，与环境里的 standing 判定保持一致。
    command_speed = torch.linalg.norm(obs[:, 3:5], dim=1)
    standing = (
        (command_speed <= STANDING_LINEAR_SPEED_THRESHOLD)
        & (torch.abs(obs[:, 5]) <= STANDING_YAW_RATE_THRESHOLD)
    )
    mirrored_phase = torch.where(
        standing.unsqueeze(1),
        obs[:, 6:8],
        mirrored_phase,
    )
    mirrored[:, 6:8] = mirrored_phase

    # projected_gravity 是重力方向这个极向量在机器人坐标系中的表示。
    # 左右镜像只改变 y 方向，因此 [gx, gy, gz] 变成 [gx, -gy, gz]。
    mirrored[:, 8:11] = obs[:, 8:11] * torch.tensor(
        [1.0, -1.0, 1.0],
        device=device,
        dtype=dtype,
    )

    # 11:25 是 14 个关节位置。
    # 25:39 是 14 个关节速度。
    # 两者使用相同的左右交换和符号变化规则。
    mirrored[:, 11:25] = _mirror_joint_obs(obs[:, 11:25])
    mirrored[:, 25:39] = _mirror_joint_obs(obs[:, 25:39])

    # 39:49 是上一时刻动作 last_action。
    # 它的排列和当前 action 相同，因此复用动作镜像函数。
    mirrored[:, 39:49] = _mirror_actions(obs[:, 39:49])

    return mirrored

def _mirror_privileged_obs(obs: torch.Tensor) -> torch.Tensor:
    """镜像 critic 专用的 base_lin_vel = [vx, vy, vz]。"""
    # obs 形状通常是 [N, 3]，列顺序为 [vx, vy, vz]。
    # 复制一份，避免直接修改原始 privileged observation。
    mirrored = obs.clone()

    # 左右镜像只改变 y 方向速度，x 和 z 保持不变。
    mirrored[:, 1] *= -1.0
    return mirrored

def _mirror_joint_obs(joint_data: torch.Tensor) -> torch.Tensor:
    """镜像 14 维 joint_pos 或 joint_vel。

    当前仿真关节索引顺序由日志中的 action joint ids 与头部关节位置确定：

    0 left_hip_yaw      1 neck_pitch
    2 right_hip_yaw     3 left_hip_roll
    4 head_pitch        5 right_hip_roll
    6 left_hip_pitch    7 head_yaw
    8 right_hip_pitch   9 left_knee
    10 head_roll        11 right_knee
    12 left_ankle       13 right_ankle
    """

    # joint_data 通常形状为 [N, 14]：
    #   N  = 并行环境数量
    #   14 = 14 个关节
    # empty_like 只创建相同形状和类型的空张量，下面会逐列填入结果。
    mirrored = torch.empty_like(joint_data)

    # 左右腿需要交换位置：左关节变成右关节，右关节变成左关节。
    # 同时，腿部关节坐标在左右镜像下需要取反。
    # 例如：镜像后的 left_yaw = -原来的 right_yaw。
    mirrored[:, 0] = -joint_data[:, 2]
    mirrored[:, 2] = -joint_data[:, 0]

    mirrored[:, 3] = -joint_data[:, 5]
    mirrored[:, 5] = -joint_data[:, 3]

    mirrored[:, 6] = -joint_data[:, 8]
    mirrored[:, 8] = -joint_data[:, 6]

    mirrored[:, 9] = -joint_data[:, 11]
    mirrored[:, 11] = -joint_data[:, 9]

    mirrored[:, 12] = -joint_data[:, 13]
    mirrored[:, 13] = -joint_data[:, 12]

    # 头部不需要左右交换，因为头部关节不是成对的左右腿关节。
    #   neck_pitch、head_pitch：前后俯仰，保持不变
    #   head_yaw、head_roll：左右相关，镜像后取反
    mirrored[:, 1] = joint_data[:, 1]
    mirrored[:, 4] = joint_data[:, 4]
    mirrored[:, 7] = -joint_data[:, 7]
    mirrored[:, 10] = -joint_data[:, 10]

    return mirrored

def _mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """镜像 10 维策略动作 / last_action。

    Action Manager 实际顺序由训练日志确认：

    [left_yaw, right_yaw, left_roll, right_roll, left_pitch,
     right_pitch, left_knee, right_knee, left_ankle, right_ankle]
    """
    # actions 通常形状为 [N, 10]：
    #   N  = 并行环境数量
    #   10 = 10 个动作关节
    # 创建输出张量，下面逐个填写镜像后的动作。
    mirrored = torch.empty_like(actions)

    # yaw：交换左右，并取反。
    mirrored[:, 0] = -actions[:, 1]
    mirrored[:, 1] = -actions[:, 0]

    # hip_roll：交换左右，并取反。
    mirrored[:, 2] = -actions[:, 3]
    mirrored[:, 3] = -actions[:, 2]

    # hip_pitch：交换左右，并取反。
    mirrored[:, 4] = -actions[:, 5]
    mirrored[:, 5] = -actions[:, 4]

    # knee：交换左右，并取反。
    mirrored[:, 6] = -actions[:, 7]
    mirrored[:, 7] = -actions[:, 6]

    # ankle：交换左右，并取反。
    mirrored[:, 8] = -actions[:, 9]
    mirrored[:, 9] = -actions[:, 8]

    return mirrored


        
