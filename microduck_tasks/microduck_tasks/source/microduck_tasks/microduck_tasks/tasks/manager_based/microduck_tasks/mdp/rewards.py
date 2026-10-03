# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


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

    # 只有恰好一只脚接触地面时，才认为是合理的单脚支撑状态
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    # 对非单脚支撑状态置零，只保留单脚支撑期间的有效时间
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), mode_time, 0.0), dim=1)[0]

    # 限制最大奖励值，避免极端情况下奖励过大
    reward = torch.clamp(reward, max=threshold)

    # 只有当前命令速度高于阈值时才给奖励，避免静止时也获得步态奖励
    command_speed = torch.linalg.norm(
        env.command_manager.get_command(command_name)[:, :2],
        dim=1,
    )
    return reward * (command_speed > command_threshold)


def feet_slide(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """惩罚足端在接触地面时的水平滑移。

    若脚在地面上仍有明显的水平速度，说明它正在滑动，
    这通常会降低稳定性。该 reward 会在接触状态下对滑移进行惩罚。
    """
    # 读取接触传感器
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # 仅在脚处于接触状态时计算滑移惩罚；
    # 这里用接触力大小判断脚是否落地，超过阈值即认为有接触
    contacts = (
        contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
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
        command_threshold: float = 0.02,
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

    # 对刚 reset 的环境强制从相位 0 开始。
    phase = torch.where(restarted, torch.zeros_like(phase), phase)

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
    # 将一整个周期 [0, 1) 映射为角度 [0, 2*pi)；phase=0.5 对应 pi。
    angle = 2.0 * torch.pi * phase
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
    # 再沿历史时间维度取最大值，得到该脚在当前时刻的代表性接触力
    contacts = (
        contact_sensor.data.net_forces_w_history.torch[
            :, :, sensor_cfg.body_ids, :
        ]
        .norm(dim=-1)
        .max(dim=1)[0]
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

    # 前半个周期：左脚应支撑、右脚应摆动
    left_should_stand = phase < 0.5
    # 期望接触状态：左脚/右脚
    desired_contacts = torch.stack((left_should_stand, ~left_should_stand),dim=1,)
    # 计算真实接触状态与期望状态的匹配程度
    # 若完全一致则为 1.0；若完全相反则为 0.0；若一对一错则为 0.5
    contact_match = (contacts == desired_contacts).float().mean(dim=1)
    # 将 [0,1] 映射到 [-0.5, 0.5]
    # 全匹配 -> +0.5
    # 全相反 -> -0.5
    # 一半匹配 -> 0
    reward =contact_match - 0.5

    # 只在命令速度大于阈值时给奖励，避免静止时通过奇怪接触模式作弊
    command_speed = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2],dim=1)

    return reward * (command_speed > command_threshold)

class HalfCycleActiveJointSymmetry(ManagerTermBase):
    """奖励左右腿相隔半个步态周期后的镜像关节运动。

    对 Microduck 而言，左右关节默认姿态互为相反数，例如：

        left_hip_pitch  default = -0.4579 rad
        right_hip_pitch default = +0.4579 rad

    因此不能直接比较左右绝对关节角，而应比较“相对默认姿态的偏移”。

    理想镜像关系为：

        left_offset(t)  ~= -right_offset(t - T/2)
        right_offset(t) ~= -left_offset(t - T/2)

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
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """创建奖励对象，并分配每个环境独立的历史缓存。"""

        # 调用父类构造函数。
        # 父类会保存 cfg 和 env，供 Manager 生命周期管理。
        super().__init__(cfg, env)
        
        # 计算半周期包含多少个控制步。
        #
        # env.step_dt 是一个控制步的时间，例如 0.02 秒。
        # 0.5 * 0.60 / 0.02 = 15 步。
        #
        # round() 避免浮点数误差把 15 算成 14 或 16。
        # max(1, ...) 防止错误配置导致历史长度为 0
        period_s = float(cfg.params["period_s"])
        self._half_cycle_steps = max(
            1,
            round(0.5 * period_s / env.step_dt),
        )

        # 创建环形历史缓存。
        #
        # shape:
        #   [env.num_envs, self._half_cycle_steps, 8]
        #
        # 第 1 维：并行环境编号。
        # 第 2 维：过去半周期内的历史控制步。
        # 第 3 维：八个待比较的关节，顺序与配置中的 joint_names 一致：
        # left_hip_roll, left_hip_pitch, left_knee, left_ankle,
        # right_hip_roll, right_hip_pitch, right_knee, right_ankle
        #
        # 每个环境都必须各自保存历史。
        # 不能把所有环境共用一个历史，因为它们会在不同时间 reset。
        self._history = torch.zeros(
            env.num_envs,
            self._half_cycle_steps,
            8,
            device=env.device,
        )
        # 记录每个环境已经积累了多少历史步。
        #
        # episode 刚 reset 后，尚未积满 15 步，
        # 此时不能拿全零历史去做镜像比较。
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
            self._valid_steps.zero_()
        else:
            self._history[env_ids] = 0.0
            self._valid_steps[env_ids] = 0

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            command_name: str,
            period_s: float,
            command_threshold: float,
            yaw_scale: float,
            std: float,
            roll_std: float,    # hip_roll 镜像误差的容忍尺度，单位 rad；越小越严格
            roll_fraction: float,   # roll 在原有镜像匹配度中的影响比例；0 表示不考虑
            min_motion: float, # hip/knee 的最小有效关节偏移，单位 rad。小于此幅度时，主动摆动门控将降低，防止策略靠“不动”获得高镜像分。
            joint_weights: tuple[float, float, float],  # 三个关节的镜像误差权重。顺序：[hip_pitch, knee, ankle]
            motion_joint_weights: tuple[float, float, float],  # # 三个关节的“主动摆动”门控权重。
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

        # 取出当前历史里保存的旧动作。因为每次调用都会循环移动编号，
        # 正常填满缓冲区后，这里取到的就是半个周期前的关节偏移。
        delayed_offset = self._history[:, self._write_index]

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

        # 只使用 hip pitch 和 knee 的摆动幅度。
        #
        # ankle 权重为 0，不让它影响“是否在主动走路”的判断。
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

        # 判断每个环境是否已积累完整半周期历史。
        #
        # reset 后前 15 步不计算镜像奖励，
        # 否则会把当前姿态和无意义的全零缓存相比较。
        history_ready = self._valid_steps >= self._half_cycle_steps

        # 取出运动命令，并计算前后、左右方向合成的移动速度。
        command = env.command_manager.get_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1)

        # 不转弯时 yaw_gate 为 1；转弯命令越大，这个系数越小。
        # yaw__scale 控制它变小的速度。
        yaw_gate = torch.exp(-torch.square(command[:, 2] / yaw_scale))

        # 将当前帧写入刚刚读取的槽位，供半周期后的调用读取。
        # 更新指针使其循环遍历缓冲区；计数封顶后表示历史已完整可用。
        # 达到末尾时 % 会让它回到 0。
        self._history[:, self._write_index] = joint_offset
        self._write_index = (
            self._write_index + 1
        ) % self._half_cycle_steps
        # 历史步数最多记到半周期长度；再增加也不影响“历史是否已准备好”。
        self._valid_steps = torch.clamp(
            self._valid_steps + 1,
            max= self._half_cycle_steps,
        )

        # 四个条件组合：
        #
        # 1. reward：镜像程度；
        # 2. history_ready：必须积满半周期历史；
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
