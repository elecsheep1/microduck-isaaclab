# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

from ..microduck_walk_params import (
    STANDING_LINEAR_SPEED_THRESHOLD,
    STANDING_YAW_RATE_THRESHOLD,
)


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚关节位置偏离目标值。

    这个 reward 会比较每个关节当前角度和目标角度之间的偏差，
    让机器人尽量保持在某个期望关节配置附近，常用于站立、姿态保持或对齐任务。
    """
    # 取出当前机器人资产，便于访问关节状态
    asset: Articulation = env.scene[asset_cfg.name]
    # 将关节角度规整到 (-pi, pi) 区间，避免跨 2π 的跳变导致误差突增
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # 计算关节误差平方和；误差越小，惩罚越小
    return torch.sum(torch.square(joint_pos - target), dim=1)


def base_lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """惩罚机器人基座在 XY 平面上的线速度。

    如果希望机器人尽量“站稳”或“不漂移”，
    就可以惩罚根节点在平面内的线速度，降低横向和前后方向的滑移。
    """
    # 取出机器人资产
    asset: Articulation = env.scene[asset_cfg.name]
    # 根节点线速度在机体坐标系中，取 x/y 方向对应 XY 平面速度
    root_lin_vel_b = asset.data.root_lin_vel_b.torch
    # 计算平方和后得到每个环境的平面速度损失
    return torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1)

def base_vertical_velocity_l2(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚机身竖直速度，抑制行走时双脚同步蹬地产生的跳跃。"""
    asset: Articulation = env.scene[asset_cfg.name]

    vertical_velocity = asset.data.root_lin_vel_b.torch[:, 2]
    return torch.square(vertical_velocity)

def stand_vertical_velocity_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        command_threshold: float,
        yaw_threshold: float,
        std: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励零速度命令下机身竖直速度接近零。

    该项只在“没有平面移动命令、也没有转向命令”时启用。
    它直接抑制蹲起、上下弹跳等站立抖动；行走和转弯时返回 0，
    因此不会干扰正常摆腿、起落脚或速度跟踪。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)

    linear_speed = torch.linalg.norm(command[:, :2], dim=1)

    standing = (
        (linear_speed <= command_threshold)
        & (torch.abs(command[:, 2]) <= yaw_threshold)
    )

    # root_lin_vel_b 是机体坐标系中的基座线速度。
    # 第 2 列是竖直方向速度；理想站立时应接近 0。
    vertical_velocity = asset.data.root_lin_vel_b.torch[:, 2]

    # 速度为 0 时奖励为 1；速度达到 std 时约降为 exp(-1)。
    reward = torch.exp(-torch.square(vertical_velocity / std))

    # 非站立状态不施加该项，避免影响正常行走。
    return reward * standing.float()

class JointMechanicalWorkPenalty(ManagerTermBase):
    """惩罚每个控制间隔内腿部关节的绝对机械功。

    对每个关节计算 |applied_torque * (q_t - q_{t-1})|，
    再对全部指定关节求和。
    力矩 X 角位移 约等于 机械功

    使用逐关节绝对值求和，避免不同关节的正功和负功互相抵消。
    """

    def __init__(self, cfg: RewardTermCfg, env:ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._actuator_name: str = cfg.params["actuator_name"]

        asset: Articulation = env.scene[self._asset_cfg.name]
        actuator = asset.actuators[self._actuator_name]

        # articulation 的 joint_pos 按全局关节编号排列；
        # actuator 的 applied_effort 按 actuator.joint_names 排列。
        # 因此按名称建立两者之间的列映射。
        # 找出奖励配置中的关节名称
        selected_joint_names = [
            asset.joint_names[joint_id]
            for joint_id in self._asset_cfg.joint_ids
        ]

        # 检查 actuator 是否控制这些关节
        missing_joint_names = [
            name for name in selected_joint_names
            if name not in actuator.joint_names
        ]
        if missing_joint_names:
            raise ValueError(
                f"Actuator'{self._actuator_name}' does not control: "
                f"{missing_joint_names}"
            )

        # 转换成 GPU 上的整数 Tensor
        self._actuator_joint_ids = torch.tensor(
            [actuator.joint_names.index(name) for name in selected_joint_names],
            device=env.device,
            dtype=torch.long,
        )

        # 每个并行环境独立保存上一控制步的关节位置。
        self._previous_joint_pos = torch.zeros(
            env.num_envs,
            len(selected_joint_names),
            device=env.device,
        )

        self._has_previous_pos = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """reset 后清空历史，避免把上一回合的末态算成机械功。"""
        if env_ids is None:
            self._previous_joint_pos.zero_()
            self._has_previous_pos.zero_()
        else:
            self._previous_joint_pos[env_ids] = 0.0
            self._has_previous_pos[env_ids] = False

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            actuator_name: str,
            asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """返回每个环境当前控制间隔的绝对关节机械功近似值。"""
        # 两个参数由 RewardTermCfg 传入；映射已在 __init__ 完成。
        # 解绑变量名
        del actuator_name, asset_cfg

        asset: Articulation = env.scene[self._asset_cfg.name]
        actuator = asset.actuators[self._actuator_name]

        current_joint_pos = asset.data.joint_pos.torch[
            :, self._asset_cfg.joint_ids
        ]

        applied_torque = actuator.applied_effort[
            :, self._actuator_joint_ids
        ]

        # Nm * rad，近似当前控制间隔的机械功。
        joint_displacement = current_joint_pos - self._previous_joint_pos
        work = torch.sum(
            torch.abs(applied_torque * joint_displacement),
            dim=1,
        )

        # 回合第一帧只建立历史，不施加惩罚。
        work = work * self._has_previous_pos.float()
        self._previous_joint_pos.copy_(current_joint_pos)
        self._has_previous_pos.fill_(True)

        return work

def biped_air_time(env: ManagerBasedRLEnv, command_name: str, threshold: float, command_threshold: float, sensor_cfg: SceneEntityCfg,) -> torch.Tensor:
    """奖励双足行走时的单脚支撑与另一脚摆动。

    这个 reward 的核心思想是：
    - 在正常步态中，通常只有一只脚接触地面，另一只脚在摆动
    - 这样更接近自然步态，能帮助机器人学习稳定行走
    """
    # 读取足端接触传感器
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # 当前脚的空气时间和接触时间，形状一般为 [num_envs, num_bodies]
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]

    # 判断每只脚当前是否处于接触状态
    in_contact = contact_time > 0.0
    # 若在接触，则取接触时间；否则取空中时间
    mode_time = torch.where(in_contact, contact_time, air_time)

    # 当前接触脚数量：0 = 腾空，1 = 单脚支撑，2 = 双支撑。
    contact_count = torch.sum(in_contact.int(), dim=1)

    # 只有恰好一只脚接触地面时，才认为是合理的单脚支撑状态
    single_stance = contact_count == 1

    single_reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), mode_time, 0.0),
        dim=1,
    )[0]
    single_reward = torch.clamp(single_reward, max=threshold)

    # 双支撑允许在换脚瞬间短暂出现，因此只给较弱惩罚。
    double_support_penalty = 0.01 * (contact_count == 2).float()

    # 两脚腾空正是跳跃模式；惩罚设置得比双支撑更强。
    flight_penalty = 0.04 * (contact_count == 0).float()

    # reward = single_reward - double_support_penalty - flight_penalty
    reward = single_reward

    # 只有当前命令速度高于阈值时才给奖励，避免静止时也获得步态奖励
    command_speed = torch.linalg.norm(
        env.command_manager.get_command(command_name)[:, :2],
        dim=1,
    )
    active = command_speed > command_threshold

    return reward * active


def feet_slide(
        env: ManagerBasedRLEnv,
        force_threshold: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """惩罚足端在接触地面时的水平滑移。

    若脚在地面上仍有明显的水平速度，说明它正在滑动，
    这通常会降低稳定性。该 reward 会在接触状态下对滑移进行惩罚。
    """
    # 读取接触传感器
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # 仅在脚处于接触状态时计算滑移惩罚；
    # 这里用接触力大小判断脚是否落地，超过阈值即认为有接触
    contacts = (
        contact_sensor.data.net_forces_w_history.torch[
            :, :, sensor_cfg.body_ids, :
            ]
            .norm(dim=-1)
            .max(dim=1)[0]
            > force_threshold
    )

    # 取出机器人脚部在世界坐标下的线速度，保留 XY 水平分量
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]

    # 对脚部水平速度做范数，并仅在接触时累计到惩罚中
    return torch.sum(foot_vel_xy.norm(dim=-1) * contacts, dim=1)

def adaptive_gait_phase(
        env: ManagerBasedRLEnv,
        command_name: str,
        period_s: float,
        slow_period_s: float,
        slow_speed: float,
        fast_speed: float,
        command_threshold: float = STANDING_LINEAR_SPEED_THRESHOLD,
) -> torch.Tensor:
    """根据移动速度调整步态周期，并为每个并行环境维护自己的相位。

    返回的 phase 是一个形状为 [环境数量] 的张量，每个值都在 [0, 1) 内：
    0 表示周期刚开始，接近 1 表示快要进入下一个周期。
    """
    # 取得名为 command_name 的速度命令。
    # 通常每行对应一个并行环境，前三列分别是 x、y 方向速度和转向速度。
    command = env.command_manager.get_command(command_name)

    # 只计算 x、y 平面内的移动速度，不把转向速度算进去。
    # dim=1 表示沿每行的两个速度分量计算长度，结果每个环境得到一个速度值。
    speed = torch.linalg.norm(command[:, :2], dim=1)

    # 把速度映射到 0 到 1 之间：
    # speed <= slow_speed 时结果为 0；speed >= fast_speed 时结果为 1。
    # 中间速度则按比例取值。clamp 用来限制结果，避免超出 [0, 1]。
    # 注意：这里的变量名 speed_ratio 原样保留；它表示速度对应的比例。
    speed_ratio = ((speed - slow_speed) / (fast_speed - slow_speed)).clamp(0.0, 1.0)

    # 根据速度比例在 slow_period_s 和 period_s 之间插值：
    # 慢速时周期接近 slow_period_s，速度较快时周期接近 period_s。
    # 周期越短，后面相位每秒推进得越快。
    period = slow_period_s + (period_s - slow_period_s) * speed_ratio

    # 如果没有明显的平面移动命令，就固定使用 period_s，
    # 避免站立或原地转弯时套用慢走周期。
    period = torch.where(
        speed <= command_threshold,
        torch.full_like(period, period_s),
        period,
    )

    # episode_length_buf 保存每个环境在当前 episode（一次运行回合）中的步数。
    current_step = env.episode_length_buf

    # 第一次调用时，在 env 上创建持续保存的相位缓存。
    # zeros_like(speed) 会创建形状、设备和数据类型都与 speed 相同的全零张量，
    # 因此每个并行环境都有自己的初始相位 0。
    if not hasattr(env, "_microduck_gait_phase"):
        env._microduck_gait_phase = torch.zeros_like(speed)
        # 同时记下当前步数，供下一次调用计算经过了多少步。
        env._microduck_phase_last_step = current_step.clone()

    # 环境 reset 后，episode 步数会从较大值回到较小值（通常是 0）。
    # 逐环境比较，因此并行环境可以在不同时间分别 reset。
    restarted = current_step < env._microduck_phase_last_step

    # 计算自上次更新后经过的步数。
    # 如果刚 reset，就按 0 步处理，不把上一个 episode 的步数算进来。
    elapsed_steps = torch.where(
        restarted,
        torch.zeros_like(current_step),
        (current_step - env._microduck_phase_last_step).clamp(min=0),
    )

    # 把经过的步数换算成时间，再除以周期，得到相位应该增加的比例。
    # 例如经过 1 秒、周期为 2 秒，相位就增加 0.5（半个周期）。
    # remainder(..., 1.0) 让相位超过 1 时从 0 重新循环。
    phase = torch.remainder(
        env._microduck_gait_phase + elapsed_steps.float() * env.step_dt / period,
        1.0,
    )

    # 仅在环境 reset 时让相位从 0 重新开始。
    phase = torch.where(
        restarted,
        torch.zeros_like(phase),
        phase,
    )

    # 保存本次结果和当前步数，下次调用时从这里继续计算。
    # 如果同一个控制步内再次调用，步数差为 0，因此不会重复推进相位。
    env._microduck_gait_phase = phase
    env._microduck_phase_last_step = current_step.clone()
    return phase


def gait_phase_sin_cos(
        env: ManagerBasedRLEnv, 
        command_name: str,
        period_s: float,
        slow_period_s: float,
        slow_speed: float,
        fast_speed: float,
        command_threshold: float,
    ) -> torch.Tensor:
    """把自适应步态相位编码为正弦和余弦，作为策略的观测。

    adaptive_gait_phase 会根据每个环境的移动命令调整周期，并持续更新该环境的相位；
    返回的 phase 形状为 [num_envs]，每个值都在 [0, 1) 内。
    本函数将它转换为 [sin(angle), cos(angle)]，最终形状为 [num_envs, 2]。

    使用正弦和余弦可以避免直接输入 phase 时在周期边界处从接近 1 跳回 0，
    让周期首尾对应的观测也保持接近，便于策略学习连续的步态变化。
    """
    # 委托自适应相位函数处理速度对应的周期、时间推进和 episode reset。
    phase = adaptive_gait_phase(
        env,
        command_name=command_name,
        period_s=period_s,
        slow_period_s=slow_period_s,
        slow_speed=slow_speed,
        fast_speed=fast_speed,
        command_threshold=command_threshold,
    )
    # 共享相位仍持续推进，但 policy 在 standing 时不需要步态时钟
    command = env.command_manager.get_command(command_name)
    linear_speed = torch.linalg.norm(command[:, :2], dim=1)

    standing = (
        (linear_speed <= command_threshold)
        & (torch.abs(command[:, 2]) <= STANDING_YAW_RATE_THRESHOLD)
    )

    # 仅覆盖给 policy 的观测相位：
    # standing -> phase 0 -> [sin(0), cos(0)] = [0, 1]。
    # 行走 / 原地转向 -> 保留连续共享相位。
    phase_for_observation = torch.where(
        standing,
        torch.zeros_like(phase),
        phase,
    )

    angle = 2.0 * torch.pi * phase_for_observation

    # 每个环境生成两个特征：[sin(angle), cos(angle)]。
    return torch.stack(
        (torch.sin(angle), torch.cos(angle)),
        dim=1,
    )

def phase_foot_contact(
        env: ManagerBasedRLEnv, 
        command_name:str, 
        period_s:float, 
        slow_period_s:float,
        slow_speed:float,
        fast_speed:float,
        command_threshold: float, 
        slow_stance_fraction: float,
        # 低速时单脚的支撑占空比。
        # 例如 0.62 表示每只脚在一个完整周期中约 62% 时间处于支撑。
        # 双足相差半周期时，双支撑比例约为 2 * 0.62 - 1 = 24%。
        fast_stance_fraction: float,
        # 高速时的支撑占空比。
        # 例如 0.52，意味着双支撑约为 4%，更接近快速交替迈步。
        transition_fraction: float,
        # 接触目标在离地、落地边界附近的平滑宽度，以“周期比例”为单位。
        # 例如 0.04 表示每个切换边界约有 4% 周期的软过渡区
        force_threshold: float, 
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
    """按步态相位奖励左右脚交替支撑。

    设计逻辑：
    - 前半周期：左脚应该接触地面，右脚离地
    - 后半周期：右脚应该接触地面，左脚离地
    - 真实接触状态与期望状态越一致，奖励越高
    - 仅当命令速度超过阈值时才给予该奖励
    """
    # 获取接触传感器
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # 计算各脚的接触力大小：
    # net_forces_w_history 的最后一维是三维力向量 [Fx, Fy, Fz]
    # 对该维做范数，相当于计算 |F| = sqrt(Fx^2 + Fy^2 + Fz^2)
    # 再沿历史时间维度取最大值，得到该脚在当前时刻的代表性接触力（[0]即max(dim=1)的0维）
    # contacts = [[True, False], ...]
    contacts = (
        contact_sensor.data.net_forces_w_history.torch[
            :, :, sensor_cfg.body_ids, :
        ]
        .norm(dim=-1)
        .max(dim=1)[0]     # 沿历史时间维度取最大值
        > force_threshold
    )
    # 当前步态相位，范围在 [0,1)
    # phase = torch.remainder(env.episode_length_buf.float() * env.step_dt / period_s, 1.0)
    phase = adaptive_gait_phase(
        env,
        command_name=command_name,
        period_s=period_s,
        slow_period_s=slow_period_s,
        slow_speed=slow_speed,
        fast_speed=fast_speed,
        command_threshold=command_threshold,
    )
    # 取当前平面速度
    command_speed = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2],dim=1)
    # 把速度映射成速度比例
    # 低于 slow_speed 时为 0，对应慢速支撑占空比；
    # 高于 fast_speed 时为 1，对应快速支撑占空比
    speed_ratio = (
        (command_speed - slow_speed) / (fast_speed - slow_speed)
    ).clamp(0.0, 1.0)

    # 根据速度插值支撑比例
    # 低速时支撑更久、允许更多双支撑；
    # 高速时逐渐缩短支撑，恢复接近交替单脚支撑。
    stance_fraction = (
        slow_stance_fraction
        + (fast_stance_fraction - slow_stance_fraction) * speed_ratio
    )

    # 左右腿相位固定相差半周期。
    #
    # phase = 0 时左腿刚进入支撑段；
    # 右腿比左腿晚半个周期，故加 0.5 后取模。
    # torch.stack((左, 右), dim=1) —— 拼成两腿的相位张量
    leg_phase = torch.stack((phase, torch.remainder(phase + 0.5, 1.0)), dim=1)

    # 将每条腿的支撑区间设为 [0, stance_fraction)。
    #
    # 为了正确处理 phase=0 的周期边界，先把相位转换为
    # 相对于支撑段中心的环形距离，而不是直接比较 phase 是否小于阈值
    stance_center = 0.5 * stance_fraction.unsqueeze(1)
    # 求环上最短距离
    # 设 x = leg_phase - stance_center，这个组合是求环上最短距离的标准技巧。三步：
    # 第 1 步 x + 0.5：把距离范围偏移
    # 第 2 步 remainder(..., 1.0)：绕回 [0, 1)
    # 第 3 步 - 0.5 再 abs：把结果映射回 [0, 0.5]
    # 最终得到的就是"在 0~1 的环上，两点间最短距离"，范围恒为 [0, 0.5]（因为环上两点最远也就差半圈）。
    phase_distance = torch.abs(
        torch.remainder(leg_phase - stance_center + 0.5, 1.0) - 0.5
    )

    # 产生 [0, 1] 的连续期望接触概率：
    #
    # - 支撑区间中央：接近 1，明确要求接触；
    # - 摆动区间中央：接近 0，明确要求离地；
    # - 起落脚边界：接近 0.5，允许自然的双支撑或短暂接触切换。
    desired_contact_prob = torch.sigmoid(
        (0.5 * stance_fraction.unsqueeze(1) - phase_distance) / transition_fraction
    )

    # 实际接触为 0/1，期望接触为连续概率。
    #
    # 完全符合期望时得 1；
    # 完全相反时得 0；
    # 在平滑边界处，双支撑或离地不会因硬阈值而突然受到大惩罚。
    contact_match = (
        1.0 - torch.abs(contacts.float() - desired_contact_prob)
    ).mean(dim=1)

    # 保持原奖励范围 [-0.5, +0.5]，避免突然改变总奖励量级。
    reward = contact_match - 0.5

    return reward * (command_speed > command_threshold)

def phase_swing_contact_penalty(
        env:ManagerBasedRLEnv,
        command_name: str,
        period_s:float,
        slow_period_s:float,
        slow_speed:float,
        fast_speed:float,
        command_threshold:float,
        slow_stance_fraction:float,
        fast_stance_fraction:float,
        transition_fraction:float,
        yaw_scale: float,
        force_threshold:float,
        sensor_cfg:SceneEntityCfg,
    ) -> torch.Tensor:
    """惩罚脚在高置信摆动阶段仍接触地面。

    使用与 phase_foot_contact() 完全相同的速度自适应相位、
    支撑占空比和软接触目标。

    仅当某脚明确应该离地时施加明显惩罚；
    在落脚、离脚过渡区，惩罚会自动减弱，允许合理双支撑。
    """

    # 读取左右脚接触状态。
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history.torch[
            :, :, sensor_cfg.body_ids, :
        ]
        .norm(dim=-1)
        .max(dim=1)[0]
        > force_threshold
    )

    # 使用同一个累计自适应相位
    phase = adaptive_gait_phase(
        env,
        command_name=command_name,
        period_s=period_s,
        slow_period_s=slow_period_s,
        slow_speed=slow_speed,
        fast_speed=fast_speed,
        command_threshold=command_threshold,
    )

    # 计算当前平面速度和速度插值比例
    command_speed = torch.linalg.norm(
        env.command_manager.get_command(command_name)[:, :2], dim=1
    )

    yaw_gate = torch.exp(
        -torch.square(env.command_manager.get_command(command_name)[:, 2] / yaw_scale)
    )

    # 直行走路时启用；静止、原地转向时不约束。
    active = (command_speed > command_threshold) * yaw_gate

    speed_ratio = (
        (command_speed - slow_speed) / (fast_speed - slow_speed)
    ).clamp(0.0, 1.0)

    # 低速支撑时间更长，高速更接近交替支撑。
    stance_fraction = (
        slow_stance_fraction + (fast_stance_fraction - slow_stance_fraction) * speed_ratio
    )

    # 左右脚相差半周期。先左脚再右脚
    leg_phase = torch.stack(
        (
            phase,
            torch.remainder(phase + 0.5, 1.0),
        ),
        dim=1,
    )

    # 环形相位距离
    stance_center = 0.5 * stance_fraction.unsqueeze(1)
    phase_distance = torch.abs(
        torch.remainder(
            leg_phase - stance_center + 0.5,
            1.0,
        ) - 0.5
    )

    # q 接近 1：期望接触；q 接近 0：期望摆动。
    desired_contact_prob = torch.sigmoid(
        (
            0.5 * stance_fraction.unsqueeze(1) - phase_distance
        ) / transition_fraction
    )

    # 仅在“明确应该摆动”的阶段惩罚接触。
    #
    # q = 0.0 -> 系数 1.0，若脚仍接触则最大惩罚；
    # q = 1.0 -> 系数 0.0，支撑脚接触不受惩罚；
    # q = 0.5 -> 系数 0.25，换脚边界仅轻微惩罚。
    swing_contact_error = (
        contacts.float() * torch.square(1.0 - desired_contact_prob)
    ).mean(dim=1)

    return swing_contact_error * active.float()

class SwingFootLiftReward(ManagerTermBase):
    """奖励摆动脚相对起跳位置的世界系抬升量。

    只在相位表明该脚处于摆动段时生效。
    使用足端世界系高度相对于摆动开始时高度的变化，
    避免机身下蹲被错误计算为足端抬升。
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 每个环境、每只脚分别记录本次摆动起点的高度。
        self._takeoff_z = torch.zeros(
            env.num_envs,
            2,
            device=env.device,
        )

        # 记录上一控制步是否处于摆动段，用于检测“刚进入摆动”。
        self._was_swing = torch.zeros(
            env.num_envs,
            2,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """环境重置时清除上一回合的起跳记录。"""
        if env_ids is None:
            self._takeoff_z.zero_()
            self._was_swing.zero_()
        else:
            self._takeoff_z[env_ids] = 0.0
            self._was_swing[env_ids] = False

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            command_name: str,
            period_s: float,
            slow_period_s: float,
            slow_speed: float,
            fast_speed: float,
            command_threshold: float,
            slow_stance_fraction: float,
            fast_stance_fraction: float,
            transition_fraction: float,
            target_lift: float,
            asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """返回每个环境的摆动脚抬升奖励。"""

        asset: Articulation = env.scene[asset_cfg.name]

        phase = adaptive_gait_phase(
            env,
            command_name=command_name,
            period_s=period_s,
            slow_period_s=slow_period_s,
            slow_speed=slow_speed,
            fast_speed=fast_speed,
            command_threshold=command_threshold,
        )

        command = env.command_manager.get_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)

        speed_ratio = ((command_speed - slow_speed) / (fast_speed - slow_speed)).clamp(0.0, 1.0)

        stance_fraction = (
            slow_stance_fraction + (fast_stance_fraction - slow_stance_fraction) * speed_ratio
        )

        # 左、右脚相差半周期。
        leg_phase = torch.stack((phase, torch.remainder(phase + 0.5, 1.0)), dim=1)

        # 环形计算
        stance_center = 0.5 * stance_fraction.unsqueeze(1)
        phase_distance = torch.abs(
            torch.remainder(leg_phase - stance_center + 0.5, 1.0) - 0.5
        )

        # q 接近 1 表示支撑期，接近 0 表示摆动期。
        desired_contact_prob = torch.sigmoid(
            (0.5 * stance_fraction.unsqueeze(1) - phase_distance) / transition_fraction
        )

        # 仅用明确摆动区检测，避免边界软过渡反复触发。
        is_swing = desired_contact_prob < 0.5
        swing_started = is_swing & (~self._was_swing)

        # 足端世界系高度，与评估器的 takeoff-to-current lift 定义保持一致。
        foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]

        # 在进入摆动相位时记录足端起始高度。
        self._takeoff_z = torch.where(
            swing_started,
            foot_z,
            self._takeoff_z,
        )

        # 足端相对于本次摆动起点的真实向上位移。
        # 机身下蹲不会再被错误计算成抬脚。
        lift = torch.clamp(
            foot_z - self._takeoff_z,
            min=0.0,
        )

        # 达到 target_lift 后饱和，避免策略无意义地高抬腿。
        lift_score = torch.clamp(
            lift / target_lift,
            min=0.0,
            max=1.0,
        )

        # 在摆动段中心权重最大；接近落脚/离脚边界时自然减弱。
        swing_confidence = 1.0 - desired_contact_prob
        # 只奖励当前真正处于摆动区间的脚。
        # 双足步态正常情况下每个时刻只有一只摆动脚，因此使用 sum，
        # 避免 mean 将奖励无意义地缩小一半。
        reward = (lift_score * swing_confidence * is_swing.float()).sum(dim=1)

        # 保存当前摆动状态，供下一控制步判断是否刚进入摆动。
        self._was_swing = is_swing

        return reward * (command_speed > command_threshold).float()


class ContactDutyBalance(ManagerTermBase):
    """惩罚完整步态周期内左右脚累计支撑时长不平衡。

    该项不要求每一个时刻两脚接触状态相同。
    它只要求经过一个完整周期后，左右脚实际支撑总时长接近。

    因此：
    - 正常左右交替步态：左右累计接触时长接近，惩罚小；
    - 一条腿长期赖在地上：左右累计接触时长不同，惩罚增大；
    - 双脚全程接触：该项本身不会惩罚，需由 swing-contact 项处理。
    """

    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRLEnv,
    ):
        super().__init__(cfg, env)

        self._left_contact_time = torch.zeros(
            env.num_envs,
            device=env.device,
        )
        self._right_contact_time = torch.zeros(
            env.num_envs,
            device=env.device,
        )
        self._elapsed_time = torch.zeros(
            env.num_envs,
            device=env.device,
        )
        self._previous_phase = torch.zeros(
            env.num_envs,
            device=env.device,
        )

        # 保存上一个完整周期的失衡程度，
        # 让惩罚在下一个周期中持续生效，而不是只在相位回绕那一帧出现。
        self._last_imbalance = torch.zeros(
            env.num_envs,
            device=env.device,
        )

    def reset(
            self,
            env_ids: Sequence[int] | None = None,
    ) -> None:
        if env_ids is None:
            self._left_contact_time.zero_()
            self._right_contact_time.zero_()
            self._elapsed_time.zero_()
            self._previous_phase.zero_()
            self._last_imbalance.zero_()
        else:
            self._left_contact_time[env_ids] = 0.0
            self._right_contact_time[env_ids] = 0.0
            self._elapsed_time[env_ids] = 0.0
            self._previous_phase[env_ids] = 0.0
            self._last_imbalance[env_ids] = 0.0

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            command_name: str,
            period_s: float,
            slow_period_s: float,
            slow_speed: float,
            fast_speed: float,
            command_threshold: float,
            yaw_threshold: float,
            min_cycle_time_s: float,
            force_threshold: float,
            sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """返回上一个完整周期的左右接触时长差，范围约为 [0, 1]。"""

        command = env.command_manager.get_command(command_name)
        command_speed = torch.linalg.norm(
            command[:, :2], dim=1
        )

        # 只在直行且有明确前进命令时累计。
        active = (
            (command_speed > command_threshold)
            & (torch.abs(command[:, 2]) <= yaw_threshold)
        )

        contact_sensor = env.scene.sensors[sensor_cfg.name]
        contacts = (
            contact_sensor.data.net_forces_w_history.torch[
                :, :, sensor_cfg.body_ids, :
            ]
            .norm(dim=-1)
            .max(dim=1)[0]
            > force_threshold
        ).float()

        phase = adaptive_gait_phase(
            env,
            command_name=command_name,
            period_s=period_s,
            slow_period_s=slow_period_s,
            slow_speed=slow_speed,
            fast_speed=fast_speed,
            command_threshold=command_threshold,
        )

        # phase 从接近 1 回到接近 0，表示刚完成一个完整周期。
        cycle_finished = (
            active
            & (phase < self._previous_phase)
            & (self._elapsed_time >= min_cycle_time_s)
        )

        # 用已完成周期内的累计接触时间计算左右支撑比例之差。
        completed_imbalance = torch.abs(
            self._left_contact_time - self._right_contact_time
        ) / self._elapsed_time.clamp_min(1e-6)

        # 周期结束时更新并保存本周期失衡度。
        self._last_imbalance = torch.where(
            active,
            torch.where(
                cycle_finished,
                completed_imbalance,
                self._last_imbalance,
            ),
            torch.zeros_like(self._last_imbalance),
        )

        # 周期结束后清零旧累计量；当前控制步会随后作为新周期的第一帧加入。
        left_time = torch.where(
            cycle_finished,
            torch.zeros_like(self._left_contact_time),
            self._left_contact_time
        )
        right_time = torch.where(
            cycle_finished,
            torch.zeros_like(self._right_contact_time),
            self._right_contact_time,
        )
        elapsed_time = torch.where(
            cycle_finished,
            torch.zeros_like(self._elapsed_time),
            self._elapsed_time,
        )

        # 仅 active 环境继续累计；不走路或转弯时清空状态，避免跨模式比较。
        self._left_contact_time = torch.where(
            active,
            left_time + contacts[:, 0] * env.step_dt,
            torch.zeros_like(left_time),
        )
        self._right_contact_time = torch.where(
            active,
            right_time + contacts[:, 1] * env.step_dt,
            torch.zeros_like(right_time),
        )
        self._elapsed_time = torch.where(
            active,
            elapsed_time + env.step_dt,
            torch.zeros_like(elapsed_time),
        )

        self._previous_phase = phase

        # 返回正的失衡误差，因此配置中使用负权重
        return self._last_imbalance * active.float()

        


class HalfCycleActiveJointSymmetry(ManagerTermBase):
    """奖励左右腿在相位相隔半周期时的镜像关节运动。

    对 Microduck 而言，左右关节默认姿态互为相反数，例如：

        left_hip_pitch  default = -0.4579 rad
        right_hip_pitch default = +0.4579 rad

    因此不能直接比较左右绝对关节角，而应比较“相对默认姿态的偏移”。

    对每个环境，从历史中寻找与当前相位相差约 0.5 周期的状态。
    周期随速度改变时，这个历史状态不一定恰好在固定的若干步之前。
    理想镜像关系为：

        left_offset(当前)  ~= -right_offset(半周期前)
        right_offset(当前) ~= -left_offset(半周期前)

    偏航命令较大时，奖励会平滑减弱，为差速转弯留出空间。

    和 v1 的区别：

    v1 只要求：
        左腿当前偏移 ~= 右腿半周期前偏移的相反数

    这样存在漏洞：
        两条腿都接近默认姿态、都少动，
        也可能得到较高镜像奖励。

    v2 额外要求：
        hip pitch 和 knee 必须具有足够的关节偏移，
        才能获得镜像奖励。

    当前实现还比较 hip roll，并用历史相位而非固定步数查找半周期前的状态。
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """创建奖励对象，并分配每个环境独立的历史缓存。"""

        # 调用父类构造函数。
        # 父类会保存 cfg 和 env，供 Manager 生命周期管理。
        super().__init__(cfg, env)
        
        # 计算最长步态周期的一半对应多少个控制步，并额外留出 2 步余量。
        # 取最长周期，是为了即使步态处于最慢状态，缓存也覆盖得了半个周期。
        # 为了取缓存长度
        max_period_s = max(
            float(cfg.params["period_s"]),
            float(cfg.params["slow_period_s"]),
        )
        self._history_len = math.ceil(0.5 * max_period_s / env.step_dt) + 2

        # 创建关节偏移和对应相位的环形历史缓存。
        #
        # shape:
        #   _history:       [环境数量, 历史槽位数, 8 个关节]
        #   _phase_history: [环境数量, 历史槽位数]
        #
        # 8 个关节的顺序与配置中的 joint_names 一致：
        # left_hip_roll, left_hip_pitch, left_knee, left_ankle,
        # right_hip_roll, right_hip_pitch, right_knee, right_ankle
        #
        # 每个环境独立保存自己的历史，因为并行环境可能在不同时间 reset。
        self._history = torch.zeros(
            env.num_envs,
            self._history_len,
            8,
            device=env.device,
        )
        self._phase_history = torch.zeros(
            env.num_envs,
            self._history_len,
            device=env.device,
        )
        # 记录每个环境已写入的有效槽位数；reset 后旧槽位不能参与匹配。
        # 是否真正覆盖半周期，要在 __call__ 中根据历史相位差判断。
        self._valid_steps = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        # 环形缓存当前写入位置。
        #
        # 所有并行环境每个控制步同步推进，
        # 所以写入指针可以由所有环境共享。
        self._write_index = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # 环境重置后清掉旧历史，避免新一轮动作和上一轮动作进行比较。
        if env_ids is None:
            self._history.zero_()
            self._phase_history.zero_()
            self._valid_steps.zero_()
        else:
            self._history[env_ids] = 0.0
            self._phase_history[env_ids] = 0.0
            self._valid_steps[env_ids] = 0

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            command_name: str,
            period_s: float,
            slow_period_s: float,
            slow_speed: float,
            fast_speed: float,
            command_threshold: float,
            yaw_scale: float,
            std: float,
            roll_std: float,    # hip_roll 镜像误差的容忍尺度，单位 rad；越小越严格
            roll_fraction: float,   # roll 对镜像匹配度的影响比例；0 表示不考虑
            min_motion: float, # hip/knee 的最小有效关节偏移，单位 rad。小于此幅度时，主动摆动门控将降低，防止策略靠“不动”获得高镜像分。
            joint_weights: tuple[float, float, float],  # 三个关节的镜像误差权重。顺序：[hip_pitch, knee, ankle]
            motion_joint_weights: tuple[float, float, float],  # 三个关节的“主动摆动”门控权重。
            asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """计算每个并行环境的半周期镜像奖励。

        返回：
            shape 为 [num_envs] 的奖励张量。
        """
        
        # 取出奖励配置指定的机器人资产。
        asset: Articulation = env.scene[asset_cfg.name]

        # 计算这 8 个关节当前角度与默认角度的差值。
        # shape:
        #   [num_envs, 8]
        joint_offset = (
            asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        )

        # 与策略观测、接触奖励共用同一自适应相位。
        # 相位由每个环境的速度命令决定；同一控制步重复调用不会重复推进。
        phase = adaptive_gait_phase(
            env,
            command_name=command_name,
            period_s=period_s,
            slow_period_s=slow_period_s,
            slow_speed=slow_speed,
            fast_speed=fast_speed,
            command_threshold=command_threshold,
        )

        # 枚举历史槽位，而不是环境编号；下方会为每个环境单独选槽位。
        # 这会创建全部槽位编号，例如历史长度为 25：
        # slot_ids = [0, 1, 2, ..., 24]
        slot_ids = torch.arange(self._history_len, device=env.device)

        # 按环形写入指针计算各槽位距今的步数；reset 后尚未写入的槽位无效。
        # `age` 表示：每个槽位里的数据距离当前有多少个控制步
        # 环境0 (valid_steps=3):   age=[1,2,3,4,5] <= 3  →  [T, T, T, F, F]
        age = (self._write_index - 1 - slot_ids) % self._history_len + 1
        valid = age.unsqueeze(0) <= self._valid_steps.unsqueeze(1)

        # 在 [0, 1) 的环形相位上计算历史状态到当前状态的相位进度。
        # 因此不会把更早的整周期误认成半周期；改动周期范围时需重新核对这一条件。
        phase_advance = torch.remainder(
            phase.unsqueeze(1) - self._phase_history,
            1.0,
        )
        # 与半周期 0.5 的距离；无效槽位不能参与最小误差选择。
        phase_error = torch.abs(phase_advance - 0.5)
        phase_error = phase_error.masked_fill(~valid,float("inf"))

        # 每个环境单独选择最接近“半周期前”的历史槽位，不做时间插值。
        # 没有有效历史时虽会得到默认索引，但下方 history_ready 会使奖励为零。
        best_slot = phase_error.argmin(dim=1)
        env_ids = torch.arange(env.num_envs, device=env.device)
        delayed_offset = self._history[env_ids, best_slot]

        # 只有有效历史的相位跨度达到半周期，才允许发放镜像奖励。
        covered_phase = phase_advance.masked_fill(~valid, -1.0).max(dim=1).values
        history_ready = covered_phase >= 0.5 - 1e-4

        # 将 Python tuple 转成 GPU tensor。
        symmetry_weights = torch.tensor(
            joint_weights,
            device=env.device,
            dtype=joint_offset.dtype,
        )

        # 将“主动摆动”权重转成 GPU tensor。
        motion_weights = torch.tensor(
            motion_joint_weights,
            device=env.device,
            dtype=joint_offset.dtype,
        )

        # 当前左腿与半周期前右腿的逐关节镜像误差。
        #
        # 理想关系：
        # current_left + delayed_right = 0。
        #
        # 输出 shape:
        # [num_envs, 3]
        # pitch、knee、ankle：仍按原来的三个权重计算
        left_pair_error = torch.abs(
            joint_offset[:, 1:4] + delayed_offset[:, 5:8]
        )
        # 当前右腿与半周期前左腿的逐关节镜像误差。
        right_pair_error = torch.abs(
            joint_offset[:, 5:8] + delayed_offset[:, 1:4]
        )
        # hip_roll 单独计算
        left_roll_error = torch.abs(
            joint_offset[:, 0] + delayed_offset[:, 4]
        )
        right_roll_error = torch.abs(
            joint_offset[:, 4] + delayed_offset[:, 0]
        )
        # 对左侧配对误差进行加权平均。
        #
        # weight 越大，该关节不对称时造成的损失越大。
        left_weighted_error = (
            left_pair_error * symmetry_weights
        ).sum(dim=1) / symmetry_weights.sum()
        right_weighted_error = (
            right_pair_error * symmetry_weights
        ).sum(dim=1) / symmetry_weights.sum()

        # 把误差转成 0 到 1 的镜像匹配度。
        #
        # error = 0 时，match = 1。
        # error 增大时，match 平滑衰减。
        # 原有矢状面关节（pitch、knee、ankle）的匹配度
        left_sagittal_match = torch.exp(
            -torch.square(left_weighted_error / std)
        )
        right_sagittal_match = torch.exp(
            -torch.square(right_weighted_error / std)
        )
        # 新增 hip_roll 的镜像匹配度。
        # 误差为 0 时等于 1；误差增大时平滑下降。   
        left_roll_match = torch.exp(
            -torch.square(left_roll_error / roll_std)
        )
        right_roll_match = torch.exp(
            -torch.square(right_roll_error / roll_std)
        )

        left_match = left_sagittal_match * (
            1.0 - roll_fraction + roll_fraction * left_roll_match
        )
        right_match = right_sagittal_match * (
            1.0 - roll_fraction + roll_fraction * right_roll_match
        )

        # 计算左侧镜像配对的平均运动幅度。
        #
        # 比较：
        # 当前左腿，以及半周期前右腿。
        #
        # abs(offset) 越大，代表关节越偏离默认姿态、
        # 越可能是在真实摆动，而不是静止站立。
        left_pair_motion = 0.5 * (
            torch.abs(joint_offset[:, 1:4])
            + torch.abs(delayed_offset[:, 5:8])
        )
        right_pair_motion = 0.5 * (
            torch.abs(joint_offset[:, 5:8])
            + torch.abs(delayed_offset[:, 1:4])
        )

        # 按配置中的 motion_joint_weights 计算主动摆动幅度。
        # 当前配置把 ankle 权重设为 0，只用 hip pitch 和 knee 判断是否在走路。
        left_motion = (
            left_pair_motion * motion_weights
        ).sum(dim=1) / motion_weights.sum()
        right_motion = (
            right_pair_motion * motion_weights
        ).sum(dim=1) / motion_weights.sum()

        # 将运动强度映射到 [0, 1]。
        #
        # motion = 0          -> gate = 0
        # motion = min_motion -> gate = 1
        # motion 更大         -> gate 仍为 1
        #
        # 这样机器人不能通过“双方都不动”获得镜像奖励。
        left_motion_gate = torch.clamp(
            left_motion / min_motion,
            min=0.0,
            max=1.0,
        )
        right_motion_gate = torch.clamp(
            right_motion / min_motion,
            min=0.0,
            max=1.0,
        )
        
        # 左、右两个方向的镜像奖励分别乘以主动摆动门控。
        #
        # 只有“镜像且有动作”才能得到高奖励。
        reward = 0.5 * (
            left_match * left_motion_gate
            + right_match * right_motion_gate
        )

        # 取出运动命令，并计算前后、左右方向合成的移动速度。
        command = env.command_manager.get_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)

        # 不转弯时 yaw_gate 为 1；转弯命令越大，这个系数越小。
        # yaw__scale 控制它变小的速度。
        yaw_gate = torch.exp(-torch.square(command[:, 2] / yaw_scale))

        # 把当前关节偏移及其相位写入同一个槽位，供未来按相位查找。
        # 写入指针循环移动；有效槽位计数在缓存长度处封顶。
        self._history[:, self._write_index] = joint_offset
        self._phase_history[:, self._write_index] = phase
        self._write_index = (
            self._write_index + 1
        ) % self._history_len
        # 槽位数达到上限不代表已覆盖半周期；仍由 history_ready 按相位判断。
        self._valid_steps = torch.clamp(
            self._valid_steps + 1,
            max= self._history_len,
        )

        # 四个条件组合：
        #
        # 1. reward：镜像程度；
        # 2. history_ready：有效历史的相位跨度必须达到半周期；
        # 3. command_speed > threshold：只有走路命令时才约束；
        # 4. yaw_gate：转弯时自动弱化。
        #
        # .float() 将 bool 张量转换为 0.0 或 1.0。
        return(
            reward
            * history_ready.float()
            * (command_speed > command_threshold).float()
            * yaw_gate
        )



def hip_yaw_neutral_l1(env: ManagerBasedRLEnv, command_name: str, command_threshold: float, yaw_scale:float, asset_cfg: SceneEntityCfg,) -> torch.Tensor:
    """惩罚直行时左右 hip yaw 偏离默认姿态。

    Microduck 没有独立脚尖 yaw 关节；脚的内八或外八主要由
    left_hip_yaw 与 right_hip_yaw 决定。

    当命令要求明显转弯时，此惩罚会平滑减弱，避免限制未来的差速转弯。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算左右 hip yaw 相对默认姿态的偏移。
    #
    # asset_cfg 只会选择 left_hip_yaw、right_hip_yaw，
    # 因此 shape 是 [num_envs, 2]。
    yaw_offset = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )

    # 对左右两髋取平均绝对偏移。
    #
    # 完全回到默认 yaw 时返回 0；
    # 外八或内八越严重，返回值越大。
    yaw_deviation = torch.mean(torch.abs(yaw_offset), dim=1)

    # 读取速度命令，格式为 [vx, vy, yaw_rate]。
    command = env.command_manager.get_command(command_name)

    # 平面速度命令。
    command_speed = torch.linalg.norm(command[:, :2], dim=1)

    # 仅在机器人被要求行走时启用惩罚。
    #
    # 静止命令下不强迫机器人为获得奖励而额外移动髋 yaw。
    walking_gate = (command_speed > command_threshold).float()

    # 根据偏航速度命令得到转弯门控：
    #
    # yaw_rate = 0             -> gate = 1，完整惩罚外八/内八；
    # abs(yaw_rate) = yaw_scale -> gate = 0，不限制正常转弯；
    # 更大转速                  -> gate 保持 0。
    yaw_gate = torch.clamp(
        1.0 - torch.abs(command[:, 2]) / yaw_scale,
        min=0.0,
        max=1.0,
    )

    # 此函数返回“成本”而不是负数。
    #
    # 配置中的 reward weight 会设为负数，因此偏移越大，
    # 最终总奖励越低。
    return yaw_deviation * yaw_gate * walking_gate

def hip_yaw_target_neutral_l1(env: ManagerBasedRLEnv, command_name: str, command_threshold: float, yaw_scale: float, action_name: str, action_indices: tuple[int,int], asset_cfg: SceneEntityCfg,) -> torch.Tensor:
    """惩罚直行时策略给左右 hip yaw 下达的过大目标角。

    与 hip_yaw_neutral_l1 的区别：

    - hip_yaw_neutral_l1：
      惩罚实际关节角偏离默认值。

    - 本函数：
      惩罚策略输出的 processed action target 偏离默认值。

    右髋被物理关节限位卡住后，实际角度最多只能到 +0.436 rad，
    即使策略继续输出 +2.7 rad，实际姿态惩罚也不会继续增加。
    本函数直接惩罚 +2.7 rad target，因此不会出现这种“惩罚饱和”。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 取得名为 joint_pos 的位置动作项。
    #
    # processed_actions 已包含动作缩放和默认关节偏置，
    # 因而这里读取的是策略真正交给 PD 控制器的目标关节角。
    action_term = env.action_manager.get_term(action_name)

    # 将 Python tuple 转为当前设备上的索引张量。
    #
    # action_indices=(0, 1) 对应：
    # 0 -> left_hip_yaw
    # 1 -> right_hip_yaw
    action_indices_tensor = torch.tensor(
        action_indices,
        device=env.device,
        dtype=torch.long,
    )

    # 读取两个 hip yaw 的 processed target。
    #
    # shape: [num_envs, 2]
    yaw_target = action_term.processed_actions[
        :,
        action_indices_tensor,
    ]

    # 读取相同两个关节的默认角度。
    #
    # asset_cfg 中必须保持 left_hip_yaw、right_hip_yaw 顺序。
    yaw_default = asset.data.default_joint_pos.torch[
        :,
        asset_cfg.joint_ids,
    ]

    # 计算策略 target 相对默认姿态的偏移
    yaw_target_offset = yaw_target - yaw_default

    # 对左右髋 yaw 的 target 偏移取平均绝对值。
    #
    # 右髋 target 为 +2.7 rad 时，这个成本会继续增大，
    # 不会像实际关节角那样被 +0.436 rad 的物理限位截断。
    yaw_target_deviation = torch.mean(
        torch.abs(yaw_target_offset),
        dim=1,
    )

    # 读取速度命令，格式为 [vx, vy, yaw_rate]。
    command = env.command_manager.get_command(command_name)

    # 平面速度命令。
    command_speed = torch.linalg.norm(command[:, :2], dim=1)

    # 仅在机器人被要求行走时启用惩罚。
    #
    # 静止命令下不强迫机器人为获得奖励而额外移动髋 yaw。
    walking_gate = (command_speed > command_threshold).float()

    # 根据偏航速度命令得到转弯门控：
    #
    # yaw_rate = 0             -> gate = 1，完整惩罚外八/内八；
    # abs(yaw_rate) = yaw_scale -> gate = 0，不限制正常转弯；
    # 更大转速                  -> gate 保持 0。
    yaw_gate = torch.clamp(
        1.0 - torch.abs(command[:, 2]) / yaw_scale,
        min=0.0,
        max=1.0,
    )

    return yaw_target_deviation * yaw_gate * walking_gate


def hip_yaw_target_soft_limit(
        env: ManagerBasedRLEnv,
        command_name: str,
        command_threshold: float,
        yaw_scale: float,
        action_name: str,
        action_indices: tuple[int, int],
        free_limit_fraction: float,
        asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚直行时髋 yaw 的目标角接近或超过物理关节限位。

    这是软屏障，不是 action clip：

    - 在关节可用行程的 free_limit_fraction 范围内：不惩罚；
    - 接近限位后：开始平方惩罚；
    - target 超过物理限位后：惩罚会快速增大；
    - 有明显转向命令时：通过 yaw_gate 平滑减弱该项。

    因此，正常行走需要的小幅髋 yaw 调整仍然允许；
    但像 right_hip_yaw target = +1.03 rad 这种远超
    +0.436 rad 机械上限的目标会受到强烈惩罚。
    """

    # 获取机器人资产。
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取位置动作项。
    #
    # processed_actions 已经过 scale 和 default offset 处理，
    # 是实际发送给 PD 执行器的关节位置目标。
    action_term = env.action_manager.get_term(action_name)

    # 将两个 hip yaw 在 action 向量中的位置转换为 GPU tensor。
    #
    # 当前 Action Manager 的解析顺序：
    # 0 -> left_hip_yaw
    # 1 -> right_hip_yaw
    action_indices_tensor = torch.tensor(
        action_indices,
        device=env.device,
        dtype=torch.long,
    )

    # 读取左右 hip yaw 的最终位置目标。
    #
    # shape: [num_envs, 2]
    yaw_target = action_term.processed_actions[
        :,
        action_indices,
    ]

    # 读取左右 hip yaw 的默认关节角。
    #
    # asset_cfg 的 joint_names 必须保持：
    # ["left_hip_yaw", "right_hip_yaw"]
    # 且 preserve_order=True。
    yaw_default = asset.data.default_joint_pos.torch[
        :,
        asset_cfg.joint_ids,
    ]

    # 计算 target 相对默认姿态的偏移。
    #
    # 正数代表朝关节正向限位移动；
    # 负数代表朝关节负向限位移动。
    yaw_target_offset = yaw_target - yaw_default

    # 读取真实关节位置限位。
    #
    # shape: [num_envs, 2, 2]
    # 最后一维：
    # 0 -> lower limit
    # 1 -> upper limit
    yaw_limits = asset.data.joint_pos_limits.torch[
        :,
        asset_cfg.joint_ids,
        :,
    ]

    lower_limit = yaw_limits[:, :, 0]
    upper_limit = yaw_limits[:, :, 1]

    # 计算从默认姿态到两个方向限位的可用行程。
    #
    # 左右髋 yaw 的正负限位并不完全对称，
    # 所以不能直接对 abs(offset) 使用同一尺度。
    positive_range = (upper_limit - yaw_default).clamp_min(1.0e-6)
    negative_range = (yaw_default - lower_limit).clamp_min(1.0e-6)

    # 将每个 target 偏移归一化为“占该方向可用行程的比例”。
    #
    # 0.00：默认姿态。
    # 0.35：使用 35% 的可用行程。
    # 1.00：刚好达到物理限位。
    # >1.00：target 已经超过物理限位。
    normalized_excursion = torch.where(
        yaw_target_offset >= 0.0,
        yaw_target_offset / positive_range,
        - yaw_target_offset / negative_range,
    )

    # 只有超出自由区间的部分才被惩罚。
    #
    # 例如 free_limit_fraction=0.35：
    # 正常的 ±15% 或 ±30% 髋 yaw 调整没有成本；
    # 接近机械限位才开始产生惩罚。
    limit_excess = torch.relu(
        normalized_excursion - free_limit_fraction
    )

    # 对超出的比例做平方惩罚。
    #
    # 目标越接近限位，成本增长越快；
    # target 超过限位时，成本会非常大。
    # 对左右髋取平均，输出 shape 为 [num_envs]。
    yaw_soft_limit_cost = torch.mean(
        limit_excess.square(),
        dim=1,
    )

    # 读取速度命令：[vx, vy, yaw_rate]。
    command = env.command_manager.get_command(command_name)

    # 仅在机器人被要求移动时启用该项。
    command_speed = torch.linalg.norm(command[:, :2], dim=1)
    walking_gate = (command_speed > command_threshold).float()

    # 转向时逐渐解除约束。
    #
    # yaw_rate = 0：完整惩罚；
    # abs(yaw_rate) = yaw_scale：惩罚为 0；
    # 更大的转向命令：仍保持 0。
    yaw_gate = torch.clamp(
        1.0 - torch.abs(command[:, 2]) / yaw_scale,
        min=0.0,
        max=1.0,
    )

    # 返回正的“成本”。
    # 配置中必须使用负 weight，使成本越高、最终奖励越低。
    return yaw_soft_limit_cost * walking_gate * yaw_gate

def hip_yaw_target_neutral_deadband(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    yaw_scale: float,
    action_name: str,
    action_indices: tuple[int, int],
    free_yaw_rad: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚直行时过大的 hip yaw 目标，允许小幅自由摆动。

    - |target - default| <= free_yaw_rad：不惩罚；
    - 超过 free_yaw_rad：平方惩罚；
    - 外八、内八都会被惩罚；
    - 有转向命令时通过 yaw_gate 逐渐关闭惩罚。

    这不是 action clip，策略仍可输出任意 yaw target。
    """

    # 读取机器人和 joint-position 动作项。
    asset: Articulation = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term(action_name)

    # 当前 Action Manager 的 yaw action 维度：
    # 0 -> left_hip_yaw
    # 1 -> right_hip_yaw
    action_indices_tensor = torch.tensor(
        action_indices,
        device=env.device,
        dtype=torch.long,
    )

    # processed_actions 是已经过 scale 与 default offset 处理后的实际 PD 目标。
    yaw_target = action_term.processed_actions[
        :,
        action_indices_tensor,
    ]

    # 读取同一对关节的默认角度。
    #
    # asset_cfg 中必须使用：
    # ["left_hip_yaw", "right_hip_yaw"]
    # 并开启 preserve_order=True。
    yaw_default = asset.data.default_joint_pos.torch[
        :,
        asset_cfg.joint_ids,
    ]

    # 计算目标相对默认姿态的偏移。
    yaw_target_offset = yaw_target - yaw_default

    # 允许正常摆腿、平衡和落脚所需的小幅 yaw 调整。
    #
    # abs() 后无论正负方向都会被同等对待：
    # 正方向外八、负方向内八，或反过来，都会在超出自由区后受罚。
    yaw_excess = torch.relu(
        torch.abs(yaw_target_offset) - free_yaw_rad
    )

    # 将超出量以自由区间归一化后平方。
    #
    # 举例，free_yaw_rad=0.10：
    # 偏移 0.10 rad -> 0 成本；
    # 偏移 0.20 rad -> (0.10 / 0.10)^2 = 1；
    # 偏移 0.40 rad -> (0.30 / 0.10)^2 = 9。
    #
    # 因而髋 yaw 靠近限位时会有很强梯度，
    # 但正常的小 yaw 摆动完全不受影响。
    normalized_excess = yaw_excess / free_yaw_rad
    yaw_cost = torch.mean(
        normalized_excess.square(),
        dim=1,
    )

    # 获取速度命令：[vx, vy, yaw_rate]。
    command = env.command_manager.get_command(command_name)

    # 静止命令下不施加该项。
    command_speed = torch.linalg.norm(command[:, :2], dim=1)
    walking_gate = (command_speed > command_threshold).float()

    # 转向越明显，惩罚越弱。
    #
    # yaw_rate=0 时完整启用；
    # abs(yaw_rate)>=yaw_scale 时关闭，
    # 因此未来可保留髋 yaw 用于差速转弯。
    yaw_gate = torch.clamp(
        1.0 - torch.abs(command[:, 2]) / yaw_scale,
        min=0.0,
        max=1.0,
    )

    # 返回正成本；配置中使用负权重。
    return yaw_cost * walking_gate * yaw_gate
