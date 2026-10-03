# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import warnings

warnings.warn(
    "scripts/reinforcement_learning/rsl_rl/play.py is deprecated. Use "
    "`./isaaclab.sh play --rl_library rsl_rl --task <TASK>` instead. "
    "Example: `./isaaclab.sh play --rl_library rsl_rl --task Isaac-Cartpole-v0`.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

import microduck_tasks.tasks  # noqa: F401
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

#调试
parser.add_argument(
      "--eval-episodes",
      type=int,
      default=0,
      help="Stop after this many completed episodes and print success statistics. 0 means run forever.",
  )


parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function. Both sides of this
# intersection are pre-fold (the callback reads the user's original sys.argv), so
# preset tokens like ``physics=NAME`` compare correctly here. Fold runs after.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        # configure_seed must be called after runner construction so that PyTorch deterministic settings
        # do not interfere with the runner's internal initialization.
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt

        # reset environment
        # obs = env.get_observations()
        # 显式 reset，确保首个可视化 episode 执行完整 reset event。
        obs, _ = env.reset()
        timestep = 0

        # 调试
        completed_episodes = 0
        timeout_episodes = 0
        failed_episodes = 0
        bad_orientation_only_episodes = 0
        base_too_low_only_episodes = 0
        both_failure_episodes = 0
        other_failure_episodes = 0

        horizontal_speed_sum = 0.0
        horizontal_speed_samples = 0
        max_horizontal_speed = 0.0

        # 足端接触与滑移统计。
        robot = env.unwrapped.scene["robot"]
        contact_sensor = env.unwrapped.scene.sensors["feet_contact"]

        sensor_foot_ids, sensor_foot_names = contact_sensor.find_sensors(
            ["ankle_left", "ankle_right"],
            preserve_order=True,
        )
        robot_foot_ids, robot_foot_names = robot.find_bodies(
            ["ankle_left", "ankle_right"],
            preserve_order=True,
        )

        print(f"[DEBUG] Contact sensor feet: {sensor_foot_names}")
        print(f"[DEBUG] Robot feet:          {robot_foot_names}")

        # ------------------------------------------------------------
        # 横向漂移与机身 heading 诊断。
        #
        # root_lin_vel_b 的第 1 维是机身坐标系的横向速度：
        # 正负值分别表示向机身左右侧横移。它用于区分“真正侧移”
        # 与“机身已偏航、但仍沿自身正前方行走”。
        # ------------------------------------------------------------
        lateral_velocity_sum = 0.0
        lateral_velocity_abs_sum = 0.0
        lateral_velocity_max_abs = 0.0
        lateral_velocity_sample_count = 0

        # 每个并行环境当前 episode 开始时的航向角。
        #
        # Isaac Lab 的 root_quat_w.torch 顺序为 [x, y, z, w]。只提取
        # 绕世界 z 轴的 yaw，忽略走路中正常的 roll / pitch 摆动。
        root_quat_w = robot.data.root_quat_w.torch
        qx, qy, qz, qw = root_quat_w.unbind(dim=-1)
        episode_start_heading = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy.square() + qz.square()),
        ).clone()

        # 累计当前 heading 相对各自 episode 初始 heading 的偏移。
        heading_error_sum = 0.0
        heading_error_abs_sum = 0.0
        heading_error_max_abs = 0.0
        heading_error_sample_count = 0

        # ------------------------------------------------------------
        # Yaw 角速度命令跟踪诊断。
        #
        # 速度命令的第 2 列是绕机身 z 轴的目标角速度（rad/s）。
        # 以 0.05 rad/s 为界，把样本分为直行、左转和右转三组。
        # 这里的“左 / 右”遵循右手系：正 yaw 为左转，负 yaw 为右转。
        # ------------------------------------------------------------
        yaw_command_threshold = 0.05
        yaw_group_names = (
            "straight (|cmd| <= 0.05)",
            "left turn (cmd > 0.05)",
            "right turn (cmd < -0.05)",
        )
        yaw_tracking_sample_count = torch.zeros(3, dtype=torch.long)
        yaw_command_sum = torch.zeros(3)
        yaw_actual_sum = torch.zeros(3)
        yaw_abs_error_sum = torch.zeros(3)
        # 直行组不统计方向正确率；后两个元素记录实际角速度与命令同号的次数。
        yaw_direction_correct_count = torch.zeros(3, dtype=torch.long)

        # 按该 episode 结束前的命令类别记录完成数和 timeout 数，
        # 用于单独观察直行、左转、右转是否存在稳定性差异。
        yaw_episode_count = torch.zeros(3, dtype=torch.long)
        yaw_timeout_episode_count = torch.zeros(3, dtype=torch.long)

        # ------------------------------------------------------------
        # 前进速度分桶诊断。
        #
        # 站立命令为零，不纳入行走分桶。将 0.16 m/s 以上的命令
        # 单独统计，检查旧策略在扩展速度范围时是否真的跟上命令。
        # ------------------------------------------------------------
        forward_speed_bin_names = (
            "low speed  (0.04 <= cmd < 0.10)",
            "mid speed  (0.10 <= cmd < 0.16)",
            "top speed  (cmd >= 0.16)",
        )
        forward_speed_sample_count = torch.zeros(3, dtype=torch.long)
        forward_speed_command_sum = torch.zeros(3)
        forward_speed_actual_sum = torch.zeros(3)
        forward_speed_abs_error_sum = torch.zeros(3)
        forward_speed_yaw_abs_error_sum = torch.zeros(3)
        # 两列依次为 left_hip_yaw、right_hip_yaw 的实际绝对偏移。
        forward_speed_hip_yaw_abs_offset_sum = torch.zeros(3, 2)
        forward_speed_contact_count = torch.zeros(3, dtype=torch.long)
        forward_speed_sliding_contact_count = torch.zeros(3, dtype=torch.long)
        forward_speed_episode_count = torch.zeros(3, dtype=torch.long)
        forward_speed_timeout_episode_count = torch.zeros(3, dtype=torch.long)

        # 取得负责十个腿部关节的显式延迟 PD 执行器。
        #
        # 这一步只需在评估开始时解析一次；不能放在控制循环内，
        # 否则每个控制步都会重复查找关节并打印调试信息。
        leg_actuator = robot.actuators["legs"]

        # 要诊断的两个髋 pitch 关节。
        hip_actuator_names = [
            "left_hip_pitch",
            "right_hip_pitch",
        ]

        # 在 legs 执行器自己的关节顺序中查找这两个关节的列号。
        #
        # 不直接写死列号，避免将来修改执行器配置的关节排序后，
        # 诊断结果悄悄对应到错误关节。
        hip_actuator_ids = [
            leg_actuator.joint_names.index(name)
            for name in hip_actuator_names
        ]

        # 只在评估启动时打印一次，作为关节映射的核对证据。
        print(f"[DEBUG] Legs actuator joint order: {leg_actuator.joint_names}")
        print(
            "[DEBUG] Hip torque actuator columns: "
            f"{list(zip(hip_actuator_names, hip_actuator_ids))}"
        )

        contacted_foot_speed_sum = 0.0
        contacted_foot_count = 0
        contact_sample_count = 0
        sliding_contact_count = 0

        single_support_step_count = 0
        double_support_step_count = 0
        flight_step_count = 0

        # 左、右脚分别的接触与滑移统计。
        left_contact_step_count = 0
        right_contact_step_count = 0

        left_contacted_speed_sum = 0.0
        right_contacted_speed_sum = 0.0

        left_contact_count = 0
        right_contact_count = 0

        left_sliding_contact_count = 0
        right_sliding_contact_count = 0


        # 记录左右髋 pitch、膝、踝的动作幅度与实际关节活动范围。
        gait_joint_names = [
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
        ]

        gait_joint_ids, gait_joint_names = robot.find_joints(
            gait_joint_names,
            preserve_order=True,
        )

        # 打印左右腿关节的默认角度和软限位，用于检查模型是否左右对称。
        default_joint_pos = robot.data.default_joint_pos.torch[
            0, gait_joint_ids
        ]
        joint_limits = robot.data.soft_joint_pos_limits.torch[
            0, gait_joint_ids
        ]

        print("\n[Joint configuration diagnostics]")
        print(
            "Joint                  "
            "default      lower       upper       allowed range"
        )

        for i, joint_name in enumerate(gait_joint_names):
            lower = joint_limits[i, 0].item()
            upper = joint_limits[i, 1].item()

            print(
                f"{joint_name:22s} "
                f"{default_joint_pos[i].item():+9.4f} "
                f"{lower:+10.4f} "
                f"{upper:+10.4f} "
                f"{(upper - lower):13.4f} rad"
            )

        # 这六个关节对应 10 维策略动作中的位置。
        gait_action_indices = torch.tensor(
            [4, 6, 8, 5, 7, 9],
            device=robot.data.joint_pos.torch.device,
        )

        joint_action_term = env.unwrapped.action_manager.get_term("joint_pos")
        action_scale = float(joint_action_term.cfg.scale)

        raw_action_abs_sum = torch.zeros(
            len(gait_joint_names),
            device=robot.data.joint_pos.torch.device,
        )
        raw_action_abs_max = torch.zeros_like(raw_action_abs_sum)
        raw_action_samples = 0

        gait_joint_pos_min = torch.full_like(
            raw_action_abs_sum,
            float("inf"),
        )
        gait_joint_pos_max = torch.full_like(
            raw_action_abs_sum,
            float("-inf"),
        )
         # 按相位前、后半周期分别记录关节轨迹。
        # phase 0: 左脚应支撑；phase 1: 右脚应支撑。
        phase_joint_pos_sum = torch.zeros(
            2,
            len(gait_joint_names),
            device=robot.data.joint_pos.torch.device,
        )
        phase_joint_pos_min = torch.full(
            (2, len(gait_joint_names)),
            float("inf"),
            device=robot.data.joint_pos.torch.device,
        )
        phase_joint_pos_max = torch.full(
            (2, len(gait_joint_names)),
            float("-inf"),
            device=robot.data.joint_pos.torch.device,
        )
        phase_joint_sample_count = torch.zeros(
            2,
            dtype=torch.long,
            device=robot.data.joint_pos.torch.device,
        )
        # ------------------------------------------------------------
        # 左右 hip pitch 的“PD 目标位置 vs 实际位置”诊断。
        #
        # phase 0：左脚应支撑 / 右脚应摆动。
        # phase 1：右脚应支撑 / 左脚应摆动。
        #
        # 第 0 列：left_hip_pitch。
        # 第 1 列：right_hip_pitch。
        # ------------------------------------------------------------

        # 在机器人全部关节中找到左右 hip pitch 的 joint id。
        hip_joint_ids, hip_joint_names = robot.find_joints(
            ["left_hip_pitch", "right_hip_pitch"],
            preserve_order=True,
        )

        # 在 Action Manager 的 10 维动作中：
        #
        # index 4 -> left_hip_pitch
        # index 5 -> right_hip_pitch
        hip_action_indices = torch.tensor(
            [4, 5],
            device=robot.data.joint_pos.torch.device,
        )

        # ------------------------------------------------------------
        # hip yaw / hip roll 姿态诊断。
        #
        # 脚尖没有独立的 yaw 自由度时，脚的内八/外八主要由 hip_yaw
        # 决定；hip_roll 也会改变脚在画面中的横向倾斜感。
        #
        # 这些诊断只统计 |yaw command| <= 0.05 rad/s 的近似直行样本，
        # 避免正常转弯时的 hip_yaw 动作混入“外八”判断。
        # ------------------------------------------------------------
        yaw_roll_joint_names = [
            "left_hip_yaw",
            "left_hip_roll",
            "right_hip_yaw",
            "right_hip_roll",
        ]
        yaw_roll_joint_ids, yaw_roll_joint_names = robot.find_joints(
            yaw_roll_joint_names,
            preserve_order=True,
        )

        # Action Manager 已确认的 10 维动作顺序：
        # 0 L yaw, 1 R yaw, 2 L roll, 3 R roll, ...
        # 本诊断的关节顺序是 L yaw, L roll, R yaw, R roll，
        # 因而需要按 [0, 2, 1, 3] 重排动作列。
        yaw_roll_action_indices = torch.tensor(
            [0, 2, 1, 3],
            device=robot.data.joint_pos.torch.device,
        )

        # 默认姿态由机器人配置/资产提供；它是判断“初始就外八”时的
        # 关节角参考，而不是策略目标。
        yaw_roll_default_pos = robot.data.default_joint_pos.torch[
            0,
            yaw_roll_joint_ids,
        ].clone()

        # 在 policy 第一次输出动作前记录初始物理关节角。
        # 若它相对 default 已经有明显偏移，优先排查 reset 或 USD。
        yaw_roll_initial_pos = robot.data.joint_pos.torch[
            0,
            yaw_roll_joint_ids,
        ].clone()

        # 仅累计近似直行时的 target / actual 相对默认姿态偏移。
        yaw_roll_target_offset_sum = torch.zeros(
            len(yaw_roll_joint_names),
            device=robot.data.joint_pos.torch.device,
        )
        yaw_roll_actual_offset_sum = torch.zeros_like(
            yaw_roll_target_offset_sum
        )
        yaw_roll_actual_pos_min = torch.full_like(
            yaw_roll_target_offset_sum,
            float("inf"),
        )
        yaw_roll_actual_pos_max = torch.full_like(
            yaw_roll_target_offset_sum,
            float("-inf"),
        )
        yaw_roll_straight_sample_count = 0

        # 每个相位、每个髋关节的目标偏移绝对值总和。
        #
        # shape: [2 个相位, 2 个髋关节]
        hip_target_abs_offset_sum = torch.zeros(
            2,
            2,
            device=robot.data.joint_pos.torch.device,
        )

        # 每个相位、每个髋关节的实际偏移绝对值总和。
        hip_actual_abs_offset_sum = torch.zeros(
            2,
            2,
            device=robot.data.joint_pos.torch.device,
        )

        # 每个相位、每个髋关节的绝对跟踪误差总和：
        #
        # abs(target_position - actual_position)
        hip_abs_tracking_error_sum = torch.zeros(
            2,
            2,
            device=robot.data.joint_pos.torch.device,
        )

        # 每个相位、每个髋关节的有符号跟踪误差总和：
        #
        # target_position - actual_position
        #
        # 正值：实际角度偏小，未跟上目标。
        # 负值：实际角度偏大，超过目标。
        hip_signed_tracking_error_sum = torch.zeros(
            2,
            2,
            device=robot.data.joint_pos.torch.device,
        )

        # 每个相位累计了多少个环境样本。
        hip_phase_sample_count = torch.zeros(
            2,
            dtype=torch.long,
            device=robot.data.joint_pos.torch.device,
        )

        # ------------------------------------------------------------
        # 左右 hip pitch 的执行器力矩诊断。
        #
        # 第 0 行对应 phase 0：左腿支撑、右腿摆动。
        # 第 1 行对应 phase 1：右腿支撑、左腿摆动。
        #
        # 两列依次是：
        # 第 0 列 left_hip_pitch；
        # 第 1 列 right_hip_pitch。
        # ------------------------------------------------------------

        # 统一使用机器人关节数据所在的设备，通常是 CUDA GPU。
        diagnostic_device = robot.data.joint_pos.torch.device

        # 逐次摆动诊断：只比较直行且全程处于同一前进速度档的完整摆动。
        # 脚的位置取 ankle body 原点；抬高量相对离地位置，并非脚底净空。
        num_eval_envs = robot.data.root_pos_w.torch.shape[0]
        swing_contact_prev = (
            contact_sensor.data.net_forces_w.torch[:, sensor_foot_ids, :]
            .norm(dim=-1) > 1.0
        ).clone()
        swing_foot_pos_prev = robot.data.body_pos_w.torch[
            :, robot_foot_ids, :
        ].clone()
        swing_root_pos_prev = robot.data.root_pos_w.torch.clone()
        swing_heading_prev = episode_start_heading.clone()
        swing_active = torch.zeros(
            (num_eval_envs, 2), dtype=torch.bool, device=diagnostic_device
        )
        swing_speed_bin = torch.full(
            (num_eval_envs, 2), -1, dtype=torch.long, device=diagnostic_device
        )
        swing_start_pos_w = torch.zeros(
            (num_eval_envs, 2, 3), device=diagnostic_device
        )
        swing_start_root_pos_w = torch.zeros(
            (num_eval_envs, 2, 3), device=diagnostic_device
        )
        swing_start_heading = torch.zeros(
            (num_eval_envs, 2), device=diagnostic_device
        )
        swing_liftoff_reach = torch.zeros(
            (num_eval_envs, 2), device=diagnostic_device
        )
        swing_max_ankle_lift = torch.zeros(
            (num_eval_envs, 2), device=diagnostic_device
        )
        swing_step_count = torch.zeros(
            (num_eval_envs, 2), dtype=torch.long, device=diagnostic_device
        )
        # 维度：[低/中/高速档，左/右脚]。
        swing_count = torch.zeros((3, 2), dtype=torch.long, device=diagnostic_device)
        swing_forward_sum = torch.zeros((3, 2), device=diagnostic_device)
        swing_root_forward_sum = torch.zeros((3, 2), device=diagnostic_device)
        swing_duration_sum = torch.zeros((3, 2), device=diagnostic_device)
        swing_lift_sum = torch.zeros((3, 2), device=diagnostic_device)
        swing_liftoff_reach_sum = torch.zeros((3, 2), device=diagnostic_device)
        swing_touchdown_reach_sum = torch.zeros((3, 2), device=diagnostic_device)

        # 与 half_cycle_active_joint_symmetry 奖励一致：比较当前关节偏移
        # 和半周期前对侧关节偏移。仅统计连续直行且速度档不变的片段。
        symmetry_cfg = env.unwrapped.cfg.rewards.half_cycle_active_joint_symmetry
        symmetry_half_steps = max(
            1, round(0.5 * symmetry_cfg.params["period_s"] / env.unwrapped.step_dt)
        )
        symmetry_joint_ids, _ = robot.find_joints(
            symmetry_cfg.params["asset_cfg"].joint_names,
            preserve_order=True,
        )
        symmetry_history = torch.zeros(
            (num_eval_envs, symmetry_half_steps, 8), device=diagnostic_device
        )
        symmetry_history_index = 0
        symmetry_previous_bin = torch.full(
            (num_eval_envs,), -1, dtype=torch.long, device=diagnostic_device
        )
        symmetry_straight_run = torch.zeros(
            (num_eval_envs,), dtype=torch.long, device=diagnostic_device
        )
        # [速度档, 当前左/右腿, 髋侧摆/髋俯仰/膝/踝]
        symmetry_error_sum = torch.zeros((3, 2, 4), device=diagnostic_device)
        symmetry_sample_count = torch.zeros(
            (3, 2), dtype=torch.long, device=diagnostic_device
        )

        # 每个相位累计了多少个环境样本。
        hip_torque_sample_count = torch.zeros(
            2,
            dtype=torch.long,
            device=diagnostic_device,
        )

        # PD 控制器计算出的、尚未被力矩上限裁剪的绝对力矩总和。
        #
        # shape: [2 个相位, 2 个髋关节]
        hip_computed_torque_abs_sum = torch.zeros(
            2,
            2,
            device=diagnostic_device,
        )

        # 最终真正施加到仿真关节上的绝对力矩总和。
        #
        # 若发生限幅，它会小于 computed_effort。
        hip_applied_torque_abs_sum = torch.zeros(
            2,
            2,
            device=diagnostic_device,
        )

        # 实际力矩占可用力矩上限的比例总和。
        #
        # 例如累计后的平均值为 0.80，表示平均使用了 80% 的力矩能力。
        hip_torque_utilization_sum = torch.zeros(
            2,
            2,
            device=diagnostic_device,
        )

        # 记录每个相位、每个髋发生力矩裁剪的次数。
        hip_torque_clipped_count = torch.zeros(
            2,
            2,
            dtype=torch.long,
            device=diagnostic_device,
        )

        # ------------------------------------------------------------
        # hip target 与实际关节角的时间对齐诊断。
        #
        # 延迟以 physics step 为单位；在当前配置下，2--8 个 physics
        # step 约等于 0.5--2 个控制周期。这里额外考虑动力学响应，
        # 比较当前及过去 8 个完整控制周期的 target。
        # ------------------------------------------------------------
        max_hip_target_lag_steps = 8

        # history[0] 是当前 target，history[1] 是上一控制周期的 target。
        # shape: [候选延迟数, 并行环境数, 左右两个 hip]
        hip_target_history = torch.zeros(
            max_hip_target_lag_steps + 1,
            robot.data.joint_pos.torch.shape[0],
            len(hip_actuator_names),
            device=diagnostic_device,
        )

        # 依次累计：候选延迟、步态相位、髋关节的绝对角度误差。
        hip_latency_error_abs_sum = torch.zeros(
            max_hip_target_lag_steps + 1,
            2,
            len(hip_actuator_names),
            device=diagnostic_device,
        )
        hip_latency_sample_count = torch.zeros(
            max_hip_target_lag_steps + 1,
            2,
            dtype=torch.long,
            device=diagnostic_device,
        )


        # simulate environment
        try:
            while True:
                start_time = time.time()
                # run everything in inference mode


                # with torch.inference_mode():
                #     # agent stepping
                #     actions = policy(obs)
                #     # env stepping
                #     obs, _, dones, _ = env.step(actions)
                #     # reset recurrent states for episodes that have terminated
                #     if version.parse(installed_version) >= version.parse("4.0.0"):
                #         policy.reset(dones)
                #     else:
                #         policy_nn.reset(dones)
                # if args_cli.video:
                #     timestep += 1
                #     if timestep == args_cli.video_length:
                #         break

            # 修改开始
                # run policy and simulation
                with torch.inference_mode():
                    actions = policy(obs)

                    # 在 env.step() 前保存本控制步实际使用的 yaw 命令。
                    # 若环境在本步结束，env.step() 内会 reset 并为下个
                    # episode 采样新命令；因此不能在后面再读取命令来给
                    # 当前物理状态或当前 episode 的终止结果分类。
                    velocity_command_before_step = (
                        env.unwrapped.command_manager.get_command(
                            "base_velocity"
                        ).clone()
                    )
                    forward_command_before_step = (
                        velocity_command_before_step[:, 0]
                    )
                    yaw_command_before_step = velocity_command_before_step[:, 2]

                    # ---------- 新增 ----------
                     # 记录策略输出的原始动作幅度。
                    raw_gait_actions = actions[:, gait_action_indices]

                    raw_action_abs_sum += raw_gait_actions.abs().sum(dim=0)
                    raw_action_abs_max = torch.maximum(
                        raw_action_abs_max,
                        raw_gait_actions.abs().max(dim=0).values,
                    )
                    raw_action_samples += raw_gait_actions.shape[0]
                    # ---------- 新增结束 ----------


                    obs, _, dones, extras = env.step(actions)

                    # ------------------------------------------------
                    # 横向速度与 heading 漂移诊断。
                    #
                    # 刚结束的环境在 env.step() 内已经 reset；不将这些
                    # reset 后的姿态纳入本控制步统计，避免污染 episode 数据。
                    # ------------------------------------------------
                    dones_bool_for_drift = dones.bool()
                    valid_drift_mask = ~dones_bool_for_drift

                    # 机身坐标系横向速度。绝对值衡量侧移大小，有符号均值
                    # 则可判断侧移是否长期偏向同一侧。
                    lateral_velocity = robot.data.root_lin_vel_b.torch[:, 1]

                    if torch.any(valid_drift_mask):
                        valid_lateral_velocity = lateral_velocity[valid_drift_mask]
                        lateral_velocity_sum += valid_lateral_velocity.sum().item()
                        lateral_velocity_abs_sum += (
                            valid_lateral_velocity.abs().sum().item()
                        )
                        lateral_velocity_max_abs = max(
                            lateral_velocity_max_abs,
                            valid_lateral_velocity.abs().max().item(),
                        )
                        lateral_velocity_sample_count += valid_lateral_velocity.numel()

                    # 将当前 root quaternion 转成世界坐标航向角。
                    root_quat_w = robot.data.root_quat_w.torch
                    qx, qy, qz, qw = root_quat_w.unbind(dim=-1)
                    current_heading = torch.atan2(
                        2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy.square() + qz.square()),
                    )

                    # atan2(sin, cos) 将角差规约到 [-pi, pi]，防止跨越
                    # +pi / -pi 时把很小的偏差误判成接近 360 度。
                    heading_error = torch.atan2(
                        torch.sin(current_heading - episode_start_heading),
                        torch.cos(current_heading - episode_start_heading),
                    )

                    if torch.any(valid_drift_mask):
                        valid_heading_error = heading_error[valid_drift_mask]
                        heading_error_sum += valid_heading_error.sum().item()
                        heading_error_abs_sum += (
                            valid_heading_error.abs().sum().item()
                        )
                        heading_error_max_abs = max(
                            heading_error_max_abs,
                            valid_heading_error.abs().max().item(),
                        )
                        heading_error_sample_count += valid_heading_error.numel()

                    # ------------------------------------------------
                    # Yaw 角速度跟踪统计。
                    #
                    # root_ang_vel_b[:, 2] 是当前物理状态的实际机身 yaw
                    # 角速度。结束环境已经 reset，因此只统计 valid mask。
                    # ------------------------------------------------
                    actual_yaw_rate = robot.data.root_ang_vel_b.torch[:, 2]
                    yaw_group_masks = (
                        torch.abs(yaw_command_before_step)
                        <= yaw_command_threshold,
                        yaw_command_before_step > yaw_command_threshold,
                        yaw_command_before_step < -yaw_command_threshold,
                    )

                    for yaw_group_id, yaw_group_mask in enumerate(
                        yaw_group_masks
                    ):
                        valid_yaw_mask = valid_drift_mask & yaw_group_mask
                        if not torch.any(valid_yaw_mask):
                            continue

                        group_command = yaw_command_before_step[valid_yaw_mask]
                        group_actual = actual_yaw_rate[valid_yaw_mask]
                        yaw_tracking_sample_count[yaw_group_id] += (
                            group_command.numel()
                        )
                        yaw_command_sum[yaw_group_id] += group_command.sum().cpu()
                        yaw_actual_sum[yaw_group_id] += group_actual.sum().cpu()
                        yaw_abs_error_sum[yaw_group_id] += (
                            torch.abs(group_actual - group_command).sum().cpu()
                        )

                        # 仅转弯组有“方向是否正确”的意义。零附近的实际
                        # 角速度不会被算作正确，避免虚假的高正确率。
                        if yaw_group_id != 0:
                            yaw_direction_correct_count[yaw_group_id] += (
                                (group_actual * group_command > 0).sum().cpu()
                            )

                    # 按前进速度分桶记录 yaw 跟踪质量。站立命令不属于任一
                    # 行走分桶，因此不会出现在这里。
                    forward_speed_bin_masks = (
                        (forward_command_before_step >= 0.04)
                        & (forward_command_before_step < 0.10),
                        (forward_command_before_step >= 0.10)
                        & (forward_command_before_step < 0.16),
                        forward_command_before_step >= 0.16,
                    )
                    for speed_bin_id, speed_bin_mask in enumerate(
                        forward_speed_bin_masks
                    ):
                        valid_speed_mask = valid_drift_mask & speed_bin_mask
                        if not torch.any(valid_speed_mask):
                            continue

                        speed_bin_command = forward_command_before_step[
                            valid_speed_mask
                        ]
                        speed_bin_yaw_command = yaw_command_before_step[
                            valid_speed_mask
                        ]
                        speed_bin_yaw_actual = actual_yaw_rate[valid_speed_mask]
                        # 与命令同在机体坐标系，且取前向 x 分量，避免
                        # 把横向漂移算进前进速度。
                        speed_bin_actual = robot.data.root_lin_vel_b.torch[
                            valid_speed_mask, 0
                        ]
                        forward_speed_sample_count[speed_bin_id] += (
                            speed_bin_command.numel()
                        )
                        forward_speed_command_sum[speed_bin_id] += (
                            speed_bin_command.sum().cpu()
                        )
                        forward_speed_actual_sum[speed_bin_id] += (
                            speed_bin_actual.sum().cpu()
                        )
                        forward_speed_abs_error_sum[speed_bin_id] += (
                            torch.abs(speed_bin_actual - speed_bin_command)
                            .sum().cpu()
                        )
                        forward_speed_yaw_abs_error_sum[speed_bin_id] += (
                            torch.abs(
                                speed_bin_yaw_actual - speed_bin_yaw_command
                            ).sum().cpu()
                        )

                    # 对已结束并完成 reset 的环境，更新新 episode 的起始
                    # heading；下一个控制步会以它作为相对航向参考。
                    if torch.any(dones_bool_for_drift):
                        episode_start_heading[dones_bool_for_drift] = (
                            current_heading[dones_bool_for_drift]
                        )

                    # 统计所有环境的机身水平速度，用于衡量滑移。
                    horizontal_speed = torch.linalg.vector_norm(
                        robot.data.root_lin_vel_b.torch[:, :2],
                        dim=1,
                    )

                    horizontal_speed_sum += horizontal_speed.sum().item()
                    horizontal_speed_samples += horizontal_speed.numel()
                    max_horizontal_speed = max(max_horizontal_speed,
                    horizontal_speed.max().item())


                     # ---------- 新增：足端接触与滑移统计 ----------
                    contacts = (
                        contact_sensor.data.net_forces_w_history.torch[
                            :, :, sensor_foot_ids, :
                        ]
                        .norm(dim=-1)
                        .max(dim=1)[0]
                        > 1.0
                    )

                    foot_vel_xy = robot.data.body_lin_vel_w.torch[
                        :, robot_foot_ids, :2
                    ]
                    foot_speed = torch.linalg.vector_norm(
                        foot_vel_xy,
                        dim=-1,
                    )

                     # contacts / foot_speed 的第 0 列是左脚，第 1 列是右脚。
                    left_contact = contacts[:, 0]
                    right_contact = contacts[:, 1]

                    left_foot_speed = foot_speed[:, 0]
                    right_foot_speed = foot_speed[:, 1]

                    # 每只脚接触地面的时间占比。
                    left_contact_step_count += left_contact.sum().item()
                    right_contact_step_count += right_contact.sum().item()

                    # 每只脚触地时的水平速度总和。
                    left_contacted_speed_sum += (
                        left_foot_speed * left_contact
                    ).sum().item()

                    right_contacted_speed_sum += (
                        right_foot_speed * right_contact
                    ).sum().item()

                    left_contact_count += left_contact.sum().item()
                    right_contact_count += right_contact.sum().item()

                    # 每只脚触地且速度超过 0.02 m/s 的次数。
                    left_sliding_contact_count += (
                        left_contact & (left_foot_speed > 0.02)
                    ).sum().item()

                    right_sliding_contact_count += (
                        right_contact & (right_foot_speed > 0.02)
                    ).sum().item()

                    # 所有“正在接触的脚”的水平速度总和。
                    contacted_foot_speed_sum += (
                        foot_speed * contacts
                    ).sum().item()
                    contacted_foot_count += contacts.sum().item()

                    # 接触时，脚速超过 0.02 m/s 的次数。
                    sliding_contact_count += (
                        contacts & (foot_speed > 0.02)
                    ).sum().item()

                    # 同一速度分桶下，仅对接触脚计算滑移比例。
                    for speed_bin_id, speed_bin_mask in enumerate(
                        forward_speed_bin_masks
                    ):
                        valid_speed_mask = valid_drift_mask & speed_bin_mask
                        if not torch.any(valid_speed_mask):
                            continue
                        forward_speed_contact_count[speed_bin_id] += (
                            contacts[valid_speed_mask].sum().cpu()
                        )
                        forward_speed_sliding_contact_count[speed_bin_id] += (
                            (
                                contacts[valid_speed_mask]
                                & (foot_speed[valid_speed_mask] > 0.02)
                            ).sum().cpu()
                        )

                    # 每个环境当前有几只脚接触地面：0、1 或 2。
                    num_contacts = contacts.sum(dim=1)

                    single_support_step_count += (
                        num_contacts == 1
                    ).sum().item()

                    double_support_step_count += (
                        num_contacts == 2
                    ).sum().item()

                    flight_step_count += (
                        num_contacts == 0
                    ).sum().item()

                    contact_sample_count += num_contacts.numel()
                    # ---------- 新增结束 ----------

                    # 逐次摆动：当前帧接触力识别离地/落地，不使用上面的历史窗口。
                    # 已 reset 的环境无效；命令中途转弯或换速度档则丢弃该摆动。
                    swing_contact_now = (
                        contact_sensor.data.net_forces_w.torch[
                            :, sensor_foot_ids, :
                        ].norm(dim=-1) > 1.0
                    )
                    swing_foot_pos_now = robot.data.body_pos_w.torch[
                        :, robot_foot_ids, :
                    ]
                    swing_root_pos_now = robot.data.root_pos_w.torch
                    swing_bin_now = torch.full(
                        (num_eval_envs,), -1,
                        dtype=torch.long, device=diagnostic_device,
                    )
                    for speed_bin_id, speed_bin_mask in enumerate(
                        forward_speed_bin_masks
                    ):
                        swing_bin_now[speed_bin_mask] = speed_bin_id
                    valid_swing_command = (
                        valid_drift_mask
                        & (torch.abs(yaw_command_before_step)
                           <= yaw_command_threshold)
                        & (swing_bin_now >= 0)
                    )
                    swing_active &= (
                        valid_swing_command[:, None]
                        & (swing_speed_bin == swing_bin_now[:, None])
                    )
                    swing_step_count[~swing_active] = 0

                    lift_off = (
                        swing_contact_prev & ~swing_contact_now
                        & valid_swing_command[:, None]
                    )
                    for foot_id in range(2):
                        started = lift_off[:, foot_id]
                        if not torch.any(started):
                            continue
                        swing_active[started, foot_id] = True
                        swing_speed_bin[started, foot_id] = swing_bin_now[started]
                        swing_start_pos_w[started, foot_id] = (
                            swing_foot_pos_prev[started, foot_id]
                        )
                        swing_start_root_pos_w[started, foot_id] = (
                            swing_root_pos_prev[started]
                        )
                        swing_start_heading[started, foot_id] = (
                            swing_heading_prev[started]
                        )
                        start_offset = (
                            swing_foot_pos_prev[started, foot_id, :2]
                            - swing_root_pos_prev[started, :2]
                        )
                        start_heading = swing_heading_prev[started]
                        swing_liftoff_reach[started, foot_id] = (
                            start_offset[:, 0] * torch.cos(start_heading)
                            + start_offset[:, 1] * torch.sin(start_heading)
                        )
                        swing_max_ankle_lift[started, foot_id] = 0.0
                        swing_step_count[started, foot_id] = 0

                    ankle_lift = (
                        swing_foot_pos_now[:, :, 2]
                        - swing_start_pos_w[:, :, 2]
                    ).clamp_min(0.0)
                    swing_max_ankle_lift = torch.where(
                        swing_active,
                        torch.maximum(swing_max_ankle_lift, ankle_lift),
                        swing_max_ankle_lift,
                    )
                    swing_step_count += swing_active.long()
                    touchdown = (
                        ~swing_contact_prev & swing_contact_now
                        & valid_drift_mask[:, None]
                    )
                    for speed_bin_id in range(3):
                        for foot_id in range(2):
                            completed_swing = (
                                touchdown[:, foot_id]
                                & swing_active[:, foot_id]
                                & (swing_speed_bin[:, foot_id] == speed_bin_id)
                                & (swing_step_count[:, foot_id] >= 2)
                            )
                            if not torch.any(completed_swing):
                                continue
                            start_heading = swing_start_heading[
                                completed_swing, foot_id
                            ]
                            displacement = (
                                swing_foot_pos_now[completed_swing, foot_id, :2]
                                - swing_start_pos_w[completed_swing, foot_id, :2]
                            )
                            forward_displacement = (
                                displacement[:, 0] * torch.cos(start_heading)
                                + displacement[:, 1] * torch.sin(start_heading)
                            )
                            root_displacement = (
                                swing_root_pos_now[completed_swing, :2]
                                - swing_start_root_pos_w[
                                    completed_swing, foot_id, :2
                                ]
                            )
                            root_forward_displacement = (
                                root_displacement[:, 0] * torch.cos(start_heading)
                                + root_displacement[:, 1] * torch.sin(start_heading)
                            )
                            touchdown_offset = (
                                swing_foot_pos_now[completed_swing, foot_id, :2]
                                - swing_root_pos_now[completed_swing, :2]
                            )
                            touchdown_heading = current_heading[completed_swing]
                            touchdown_reach = (
                                touchdown_offset[:, 0] * torch.cos(touchdown_heading)
                                + touchdown_offset[:, 1] * torch.sin(touchdown_heading)
                            )
                            swing_count[speed_bin_id, foot_id] += (
                                completed_swing.sum()
                            )
                            swing_forward_sum[speed_bin_id, foot_id] += (
                                forward_displacement.sum()
                            )
                            swing_root_forward_sum[speed_bin_id, foot_id] += (
                                root_forward_displacement.sum()
                            )
                            swing_duration_sum[speed_bin_id, foot_id] += (
                                swing_step_count[completed_swing, foot_id]
                                .float().sum() * dt
                            )
                            swing_lift_sum[speed_bin_id, foot_id] += (
                                swing_max_ankle_lift[completed_swing, foot_id].sum()
                            )
                            swing_liftoff_reach_sum[speed_bin_id, foot_id] += (
                                swing_liftoff_reach[completed_swing, foot_id].sum()
                            )
                            swing_touchdown_reach_sum[speed_bin_id, foot_id] += (
                                touchdown_reach.sum()
                            )
                    swing_active[touchdown] = False
                    swing_speed_bin[touchdown] = -1
                    swing_step_count[touchdown] = 0
                    swing_contact_prev.copy_(swing_contact_now)
                    swing_foot_pos_prev.copy_(swing_foot_pos_now)
                    swing_root_pos_prev.copy_(swing_root_pos_now)
                    swing_heading_prev.copy_(current_heading)

                    # 记录真实关节位置的全程最小值与最大值。
                    gait_joint_pos = robot.data.joint_pos.torch[
                        :, gait_joint_ids
                    ]

                    symmetry_bin_now = torch.full(
                        (num_eval_envs,), -1,
                        dtype=torch.long, device=diagnostic_device,
                    )
                    for speed_bin_id, speed_bin_mask in enumerate(
                        forward_speed_bin_masks
                    ):
                        symmetry_bin_now[speed_bin_mask] = speed_bin_id
                    symmetry_straight_now = (
                        valid_drift_mask
                        & (torch.abs(yaw_command_before_step)
                           <= yaw_command_threshold)
                        & (symmetry_bin_now >= 0)
                    )
                    symmetry_same_run = (
                        symmetry_straight_now
                        & (symmetry_bin_now == symmetry_previous_bin)
                    )
                    symmetry_straight_run = torch.where(
                        symmetry_straight_now,
                        torch.where(
                            symmetry_same_run,
                            symmetry_straight_run + 1,
                            torch.ones_like(symmetry_straight_run),
                        ),
                        torch.zeros_like(symmetry_straight_run),
                    )
                    symmetry_offset = (
                        robot.data.joint_pos.torch[:, symmetry_joint_ids]
                        - robot.data.default_joint_pos.torch[:, symmetry_joint_ids]
                    )
                    symmetry_delayed = symmetry_history[:, symmetry_history_index]
                    symmetry_pair_error = torch.stack(
                        (
                            torch.abs(
                                symmetry_offset[:, :4] + symmetry_delayed[:, 4:]
                            ),
                            torch.abs(
                                symmetry_offset[:, 4:] + symmetry_delayed[:, :4]
                            ),
                        ),
                        dim=1,
                    )
                    for speed_bin_id in range(3):
                        symmetry_mask = (
                            symmetry_straight_now
                            & (symmetry_bin_now == speed_bin_id)
                            & (symmetry_straight_run > symmetry_half_steps)
                        )
                        if torch.any(symmetry_mask):
                            symmetry_error_sum[speed_bin_id] += (
                                symmetry_pair_error[symmetry_mask].sum(dim=0)
                            )
                            symmetry_sample_count[speed_bin_id] += (
                                symmetry_mask.sum()
                            )
                    symmetry_history[:, symmetry_history_index] = symmetry_offset
                    symmetry_history[dones_bool_for_drift] = 0.0
                    symmetry_history_index = (
                        symmetry_history_index + 1
                    ) % symmetry_half_steps
                    symmetry_previous_bin = torch.where(
                        symmetry_straight_now,
                        symmetry_bin_now,
                        torch.full_like(symmetry_bin_now, -1),
                    )

                    gait_joint_pos_min = torch.minimum(
                        gait_joint_pos_min,
                        gait_joint_pos.min(dim=0).values,
                    )
                    gait_joint_pos_max = torch.maximum(
                        gait_joint_pos_max,
                        gait_joint_pos.max(dim=0).values,
                    )
                    # 用与 phase_foot_contact 完全相同的 0.60 s 周期分桶。
                    phase = torch.remainder(
                        env.unwrapped.episode_length_buf.float()
                        * env.unwrapped.step_dt
                        / 0.60,
                        1.0,
                    )
                    phase_bin = (phase >= 0.5).long()

                    # ------------------------------------------------
                    # 读取执行器在本控制周期最后一个 physics step 的力矩。
                    #
                    # computed_effort：延迟后的目标经过 PD 控制器计算、
                    #                  但尚未经过力矩上限裁剪的力矩。
                    # applied_effort ：真正写入仿真的力矩。
                    #
                    # 两者不同即意味着此 physics step 发生了限幅。
                    # ------------------------------------------------
                    hip_computed_torque = leg_actuator.computed_effort[
                        :, hip_actuator_ids
                    ]
                    hip_applied_torque = leg_actuator.applied_effort[
                        :, hip_actuator_ids
                    ]
                    hip_effort_limit = leg_actuator.effort_limit[
                        :, hip_actuator_ids
                    ]

                    # 使用很小的容差，忽略浮点数舍入造成的假阳性。
                    hip_torque_was_clipped = (
                        torch.abs(hip_computed_torque - hip_applied_torque)
                        > 1e-5
                    )

                    # 实际力矩占当前可用力矩上限的比例。
                    hip_torque_utilization = (
                        torch.abs(hip_applied_torque)
                        / torch.abs(hip_effort_limit).clamp_min(1e-6)
                    )

                    # ------------------------------------------------
                    # 读取 PD 控制器真正收到的目标关节位置。
                    #
                    # processed_actions 已经完成：
                    # raw action * scale + default joint offset
                    #
                    # 因此它是比 policy 原始 raw action 更可靠的
                    # “实际关节目标角”。
                    # ------------------------------------------------
                    hip_target_pos = (
                        joint_action_term.processed_actions[
                            :, hip_action_indices
                        ]
                    )

                    # 读取物理仿真后的实际髋关节角度。
                    hip_actual_pos = robot.data.joint_pos.torch[
                        :, hip_joint_ids
                    ]

                    # 将本周期 target 写入历史队列。
                    # clone() 防止移动切片时覆盖仍需保留的旧目标。
                    hip_target_history[1:] = hip_target_history[:-1].clone()
                    hip_target_history[0] = hip_target_pos

                    # 新回合的开头没有足够历史，不能参与时延比较。
                    hip_latency_history_is_valid = (
                        env.unwrapped.episode_length_buf
                        > max_hip_target_lag_steps
                    )

                    # 读取左右 hip pitch 的默认姿态。
                    hip_default_pos = robot.data.default_joint_pos.torch[
                        :, hip_joint_ids
                    ]

                    # 计算带符号跟踪误差。
                    #
                    # target - actual > 0：
                    # 实际位置落后于目标位置。
                    #
                    # target - actual < 0：
                    # 实际位置超过目标位置。
                    hip_tracking_error = (
                        hip_target_pos - hip_actual_pos
                    )

                    # 读取 yaw / roll 的当前位置 target 与物理实际角度。
                    yaw_roll_target_pos = (
                        joint_action_term.processed_actions[
                            :, yaw_roll_action_indices
                        ]
                    )
                    yaw_roll_actual_pos = robot.data.joint_pos.torch[
                        :, yaw_roll_joint_ids
                    ]

                    # 按速度档记录两个髋 yaw 的实际偏移。这里包含转弯样本；
                    # 它用于判断高速时关节是否更常被推向极限，而不是用于
                    # 判断外八本身。
                    hip_yaw_actual_abs_offset = torch.abs(
                        yaw_roll_actual_pos[:, [0, 2]]
                        - yaw_roll_default_pos[[0, 2]]
                    )
                    for speed_bin_id, speed_bin_mask in enumerate(
                        forward_speed_bin_masks
                    ):
                        valid_speed_mask = valid_drift_mask & speed_bin_mask
                        if torch.any(valid_speed_mask):
                            forward_speed_hip_yaw_abs_offset_sum[
                                speed_bin_id
                            ] += hip_yaw_actual_abs_offset[
                                valid_speed_mask
                            ].sum(dim=0).cpu()

                    # 只取近似直行命令的样本。
                    #
                    # 在真实转弯中，hip_yaw 偏离默认姿态是合理且必要的；
                    # 若把它混进来，会错误地把转弯当成“外八”。
                    current_velocity_command = (
                        env.unwrapped.command_manager.get_command(
                            "base_velocity"
                        )
                    )
                    straight_command_mask = (
                        torch.abs(current_velocity_command[:, 2]) <= 0.05
                    )

                    if torch.any(straight_command_mask):
                        yaw_roll_target_offset_sum += (
                            yaw_roll_target_pos[straight_command_mask]
                            - yaw_roll_default_pos
                        ).sum(dim=0)
                        yaw_roll_actual_offset_sum += (
                            yaw_roll_actual_pos[straight_command_mask]
                            - yaw_roll_default_pos
                        ).sum(dim=0)
                        yaw_roll_actual_pos_min = torch.minimum(
                            yaw_roll_actual_pos_min,
                            yaw_roll_actual_pos[
                                straight_command_mask
                            ].min(dim=0).values,
                        )
                        yaw_roll_actual_pos_max = torch.maximum(
                            yaw_roll_actual_pos_max,
                            yaw_roll_actual_pos[
                                straight_command_mask
                            ].max(dim=0).values,
                        )
                        yaw_roll_straight_sample_count += int(
                            straight_command_mask.sum().item()
                        )

                    for phase_id in range(2):
                        mask = phase_bin == phase_id
                        if torch.any(mask):
                            # 对 0--4 个候选控制周期延迟逐一计算误差。
                            # 最小误差所对应的 lag 是最接近的时序对齐。
                            latency_mask = (
                                mask & hip_latency_history_is_valid
                            )
                            if torch.any(latency_mask):
                                for lag_step in range(
                                    max_hip_target_lag_steps + 1
                                ):
                                    hip_latency_error_abs_sum[
                                        lag_step,
                                        phase_id,
                                    ] += torch.abs(
                                        hip_target_history[
                                            lag_step,
                                            latency_mask,
                                        ]
                                        - hip_actual_pos[latency_mask]
                                    ).sum(dim=0)
                                    hip_latency_sample_count[
                                        lag_step,
                                        phase_id,
                                    ] += latency_mask.sum()

                            # 累积这一相位、两个髋的执行器力矩统计。
                            # 所有统计张量都在 GPU 上，避免 CPU/GPU 混用错误。
                            hip_torque_sample_count[phase_id] += mask.sum()
                            hip_computed_torque_abs_sum[phase_id] += (
                                torch.abs(hip_computed_torque[mask]).sum(dim=0)
                            )
                            hip_applied_torque_abs_sum[phase_id] += (
                                torch.abs(hip_applied_torque[mask]).sum(dim=0)
                            )
                            hip_torque_utilization_sum[phase_id] += (
                                hip_torque_utilization[mask].sum(dim=0)
                            )
                            hip_torque_clipped_count[phase_id] += (
                                hip_torque_was_clipped[mask].sum(dim=0)
                            )

                            phase_joint_pos = gait_joint_pos[mask]

                            # 取出当前相位中的髋关节目标、实际位置和默认位置。
                            phase_hip_target_pos = hip_target_pos[mask]
                            phase_hip_actual_pos = hip_actual_pos[mask]
                            phase_hip_default_pos = hip_default_pos[mask]
                            phase_hip_tracking_error = (
                                hip_tracking_error[mask]
                            )

                            # 记录“目标偏离默认姿态”的平均大小。
                            #
                            # 若右髋目标偏移本身很小，说明问题属于：
                            # 策略没有要求右髋迈更大步。
                            hip_target_abs_offset_sum[phase_id] += (
                                torch.abs(
                                    phase_hip_target_pos
                                    - phase_hip_default_pos
                                ).sum(dim=0)
                            )

                            # 记录“实际偏离默认姿态”的平均大小。
                            #
                            # 若目标很大、实际很小，则可能是跟踪能力问题。
                            hip_actual_abs_offset_sum[phase_id] += (
                                torch.abs(
                                    phase_hip_actual_pos
                                    - phase_hip_default_pos
                                ).sum(dim=0)
                            )

                            # 记录绝对目标跟踪误差。
                            hip_abs_tracking_error_sum[phase_id] += (
                                torch.abs(
                                    phase_hip_tracking_error
                                ).sum(dim=0)
                            )

                            # 记录有符号目标跟踪误差。
                            hip_signed_tracking_error_sum[phase_id] += (
                                phase_hip_tracking_error.sum(dim=0)
                            )

                            # 累计该相位中参与统计的环境数量。
                            hip_phase_sample_count[phase_id] += (
                                mask.sum()
                            )

                            phase_joint_pos_sum[phase_id] += (
                                phase_joint_pos.sum(dim=0)
                            )
                            phase_joint_pos_min[phase_id] = torch.minimum(
                                phase_joint_pos_min[phase_id],
                                phase_joint_pos.min(dim=0).values,
                            )
                            phase_joint_pos_max[phase_id] = torch.maximum(
                                phase_joint_pos_max[phase_id],
                                phase_joint_pos.max(dim=0).values,
                            )
                            phase_joint_sample_count[phase_id] += mask.sum()

                    # ---------- 新增结束 ----------

                    # Reset recurrent policy state for completed environments.
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)

                # Count completed episodes.
                dones_bool = dones.bool()
                if torch.any(dones_bool):
                    # 不让上一个回合的 target 混入新回合的历史。
                    hip_target_history[:, dones_bool] = 0.0

                    time_outs = extras["time_outs"].bool()
                    failed_mask = dones_bool & ~time_outs

                    # 读取这一步中每个环境触发的终止条件。
                    termination_manager = env.unwrapped.termination_manager
                    bad_orientation = termination_manager.get_term("bad_orientation").bool()
                    base_too_low = termination_manager.get_term("base_too_low").bool()

                    completed_episodes += dones_bool.sum().item()
                    timeout_episodes += (dones_bool & time_outs).sum().item()
                    failed_episodes += failed_mask.sum().item()

                    # 失败原因：分别统计“仅一个条件触发”和“两者同时触发”。
                    bad_orientation_only_episodes += (
                        failed_mask & bad_orientation & ~base_too_low
                    ).sum().item()

                    base_too_low_only_episodes += (
                        failed_mask & base_too_low & ~bad_orientation
                    ).sum().item()

                    both_failure_episodes += (
                        failed_mask & bad_orientation & base_too_low
                    ).sum().item()

                    other_failure_episodes += (
                        failed_mask & ~bad_orientation & ~base_too_low
                    ).sum().item()

                    # 终止归属到 env.step() 前的命令类别，而不是 reset 后
                    # 的新 episode 命令。这样三组成功率才可比较。
                    yaw_group_masks_for_done = (
                        torch.abs(yaw_command_before_step)
                        <= yaw_command_threshold,
                        yaw_command_before_step > yaw_command_threshold,
                        yaw_command_before_step < -yaw_command_threshold,
                    )
                    for yaw_group_id, yaw_group_mask in enumerate(
                        yaw_group_masks_for_done
                    ):
                        group_done_mask = dones_bool & yaw_group_mask
                        yaw_episode_count[yaw_group_id] += (
                            group_done_mask.sum().cpu()
                        )
                        yaw_timeout_episode_count[yaw_group_id] += (
                            (group_done_mask & time_outs).sum().cpu()
                        )

                    # 终止按 episode 最后一个控制步的前进速度命令归类。
                    forward_speed_bin_masks_for_done = (
                        (forward_command_before_step >= 0.04)
                        & (forward_command_before_step < 0.10),
                        (forward_command_before_step >= 0.10)
                        & (forward_command_before_step < 0.16),
                        forward_command_before_step >= 0.16,
                    )
                    for speed_bin_id, speed_bin_mask in enumerate(
                        forward_speed_bin_masks_for_done
                    ):
                        speed_bin_done_mask = dones_bool & speed_bin_mask
                        forward_speed_episode_count[speed_bin_id] += (
                            speed_bin_done_mask.sum().cpu()
                        )
                        forward_speed_timeout_episode_count[speed_bin_id] += (
                            (speed_bin_done_mask & time_outs).sum().cpu()
                        )


                    if args_cli.eval_episodes > 0 and completed_episodes >= args_cli.eval_episodes:
                        success_rate = timeout_episodes / completed_episodes

                        print("\n[Evaluation complete]")
                        print(f"Completed episodes: {completed_episodes}")
                        print(f"Timeout / success:  {timeout_episodes}")
                        print(f"  bad_orientation only: {bad_orientation_only_episodes}")
                        print(f"  base_too_low only:    {base_too_low_only_episodes}")
                        print(f"  both conditions:      {both_failure_episodes}")
                        print(f"  other failure:        {other_failure_episodes}")

                        mean_horizontal_speed = horizontal_speed_sum / horizontal_speed_samples

                        print(f"Mean horizontal speed: {mean_horizontal_speed:.4f} m/s")
                        print(f"Max horizontal speed:  {max_horizontal_speed:.4f} m/s")

                        # ------------------------------------------------
                        # 输出直行 / 左转 / 右转分组的 yaw 跟踪质量。
                        #
                        # MAE 越小越好；转弯组 direction correct 越接近
                        # 100% 越好。直行组不输出方向正确率，因为其命令
                        # 本来接近零。
                        # ------------------------------------------------
                        print("\n[Yaw-rate tracking diagnostics]")
                        print(
                            "Group                         samples  mean cmd  "
                            "mean actual      MAE  direction  success"
                        )
                        for yaw_group_id, yaw_group_name in enumerate(
                            yaw_group_names
                        ):
                            sample_count = yaw_tracking_sample_count[
                                yaw_group_id
                            ].item()
                            episode_count = yaw_episode_count[
                                yaw_group_id
                            ].item()

                            if sample_count > 0:
                                mean_yaw_command = (
                                    yaw_command_sum[yaw_group_id].item()
                                    / sample_count
                                )
                                mean_yaw_actual = (
                                    yaw_actual_sum[yaw_group_id].item()
                                    / sample_count
                                )
                                yaw_mae = (
                                    yaw_abs_error_sum[yaw_group_id].item()
                                    / sample_count
                                )
                                if yaw_group_id == 0:
                                    direction_text = "    n/a"
                                else:
                                    direction_correct = (
                                        yaw_direction_correct_count[
                                            yaw_group_id
                                        ].item()
                                        / sample_count
                                    )
                                    direction_text = (
                                        f"{direction_correct:8.2%}"
                                    )
                            else:
                                mean_yaw_command = float("nan")
                                mean_yaw_actual = float("nan")
                                yaw_mae = float("nan")
                                direction_text = "    n/a"

                            if episode_count > 0:
                                group_success = (
                                    yaw_timeout_episode_count[
                                        yaw_group_id
                                    ].item()
                                    / episode_count
                                )
                                success_text = f"{group_success:7.2%}"
                            else:
                                success_text = "    n/a"

                            print(
                                f"{yaw_group_name:28s} {sample_count:7d}  "
                                f"{mean_yaw_command:+8.4f}  "
                                f"{mean_yaw_actual:+11.4f}  "
                                f"{yaw_mae:7.4f}  {direction_text}  "
                                f"{success_text}"
                            )

                        # ------------------------------------------------
                        # 输出三个前进速度档的质量对比。
                        # ------------------------------------------------
                        print("\n[Forward-speed-binned diagnostics]")
                        print(
                            "Group                         samples  mean cmd  mean vx  "
                            "vx MAE  yaw MAE  L/R hip-yaw |offset|  slide  success"
                        )
                        for speed_bin_id, speed_bin_name in enumerate(
                            forward_speed_bin_names
                        ):
                            sample_count = forward_speed_sample_count[
                                speed_bin_id
                            ].item()
                            episode_count = forward_speed_episode_count[
                                speed_bin_id
                            ].item()

                            if sample_count > 0:
                                mean_forward_command = (
                                    forward_speed_command_sum[
                                        speed_bin_id
                                    ].item()
                                    / sample_count
                                )
                                mean_forward_actual = (
                                    forward_speed_actual_sum[speed_bin_id].item()
                                    / sample_count
                                )
                                forward_speed_mae = (
                                    forward_speed_abs_error_sum[speed_bin_id].item()
                                    / sample_count
                                )
                                speed_bin_yaw_mae = (
                                    forward_speed_yaw_abs_error_sum[
                                        speed_bin_id
                                    ].item()
                                    / sample_count
                                )
                                mean_hip_yaw_offset = (
                                    forward_speed_hip_yaw_abs_offset_sum[
                                        speed_bin_id
                                    ]
                                    / sample_count
                                )
                                contact_count = forward_speed_contact_count[
                                    speed_bin_id
                                ].item()
                                sliding_ratio = (
                                    forward_speed_sliding_contact_count[
                                        speed_bin_id
                                    ].item()
                                    / max(contact_count, 1)
                                )
                            else:
                                mean_forward_command = float("nan")
                                mean_forward_actual = float("nan")
                                forward_speed_mae = float("nan")
                                speed_bin_yaw_mae = float("nan")
                                mean_hip_yaw_offset = torch.full((2,), float("nan"))
                                sliding_ratio = float("nan")

                            if episode_count > 0:
                                speed_bin_success = (
                                    forward_speed_timeout_episode_count[
                                        speed_bin_id
                                    ].item()
                                    / episode_count
                                )
                                speed_success_text = f"{speed_bin_success:7.2%}"
                            else:
                                speed_success_text = "    n/a"

                            print(
                                f"{speed_bin_name:28s} {sample_count:7d}  "
                                f"{mean_forward_command:8.4f}  "
                                f"{mean_forward_actual:7.4f}  "
                                f"{forward_speed_mae:6.4f}  "
                                f"{speed_bin_yaw_mae:7.4f}  "
                                f"{mean_hip_yaw_offset[0].item():7.4f}/"
                                f"{mean_hip_yaw_offset[1].item():7.4f}  "
                                f"{sliding_ratio:6.2%}  {speed_success_text}"
                            )

                        # ------------------------------------------------
                        # 输出横向漂移与 heading 诊断。
                        #
                        # 横向速度使用机身坐标系；heading 漂移使用相对本
                        # episode 初始朝向的世界 yaw，并在这里转换为角度。
                        # ------------------------------------------------
                        mean_lateral_velocity = (
                            lateral_velocity_sum
                            / max(lateral_velocity_sample_count, 1)
                        )
                        mean_abs_lateral_velocity = (
                            lateral_velocity_abs_sum
                            / max(lateral_velocity_sample_count, 1)
                        )
                        mean_heading_error_deg = (
                            heading_error_sum
                            / max(heading_error_sample_count, 1)
                            * 180.0
                            / torch.pi
                        )
                        mean_abs_heading_error_deg = (
                            heading_error_abs_sum
                            / max(heading_error_sample_count, 1)
                            * 180.0
                            / torch.pi
                        )
                        max_heading_error_deg = (
                            heading_error_max_abs * 180.0 / torch.pi
                        )

                        print("\n[Lateral drift and heading diagnostics]")
                        print(
                            f"Mean body lateral velocity: "
                            f"{mean_lateral_velocity:+.4f} m/s"
                        )
                        print(
                            f"Mean |body lateral velocity|: "
                            f"{mean_abs_lateral_velocity:.4f} m/s"
                        )
                        print(
                            f"Max |body lateral velocity|:  "
                            f"{lateral_velocity_max_abs:.4f} m/s"
                        )
                        print(
                            f"Mean heading drift: "
                            f"{mean_heading_error_deg:+.2f} deg"
                        )
                        print(
                            f"Mean |heading drift|: "
                            f"{mean_abs_heading_error_deg:.2f} deg"
                        )
                        print(
                            f"Max |heading drift|:  "
                            f"{max_heading_error_deg:.2f} deg"
                        )

                        # ---------- 新增：足端接触诊断结果 ----------
                        mean_contacted_foot_speed = (
                            contacted_foot_speed_sum
                            / max(contacted_foot_count, 1)
                        )

                        sliding_contact_ratio = (
                            sliding_contact_count
                            / max(contacted_foot_count, 1)
                        )

                        single_support_ratio = (
                            single_support_step_count
                            / contact_sample_count
                        )

                        double_support_ratio = (
                            double_support_step_count
                            / contact_sample_count
                        )

                        flight_ratio = (
                            flight_step_count
                            / contact_sample_count
                        )

                        print("\n[Foot contact diagnostics]")
                        print(
                            f"Mean contacted-foot speed: "
                            f"{mean_contacted_foot_speed:.4f} m/s"
                        )
                        print(
                            f"Sliding-contact ratio (>0.02 m/s): "
                            f"{sliding_contact_ratio:.2%}"
                        )
                        print(
                            f"Single-support ratio: {single_support_ratio:.2%}"
                        )
                        print(
                            f"Double-support ratio: {double_support_ratio:.2%}"
                        )
                        print(f"Flight ratio:         {flight_ratio:.2%}")

                        left_contact_ratio = (
                            left_contact_step_count / contact_sample_count
                        )
                        right_contact_ratio = (
                            right_contact_step_count / contact_sample_count
                        )

                        left_mean_contacted_speed = (
                            left_contacted_speed_sum
                            / max(left_contact_count, 1)
                        )
                        right_mean_contacted_speed = (
                            right_contacted_speed_sum
                            / max(right_contact_count, 1)
                        )

                        left_sliding_ratio = (
                            left_sliding_contact_count
                            / max(left_contact_count, 1)
                        )
                        right_sliding_ratio = (
                            right_sliding_contact_count
                            / max(right_contact_count, 1)
                        )

                        print("\n[Left-right foot diagnostics]")
                        print(
                            f"Left contact ratio:   {left_contact_ratio:.2%}"
                        )
                        print(
                            f"Right contact ratio:  {right_contact_ratio:.2%}"
                        )
                        print(
                            f"Left contact speed:   "
                            f"{left_mean_contacted_speed:.4f} m/s"
                        )
                        print(
                            f"Right contact speed:  "
                            f"{right_mean_contacted_speed:.4f} m/s"
                        )
                        print(
                            f"Left sliding ratio:   {left_sliding_ratio:.2%}"
                        )
                        print(
                            f"Right sliding ratio:  {right_sliding_ratio:.2%}"
                        )

                        print("\n[Per-swing straight-walk diagnostics]")
                        print(
                            "Only complete swings with |yaw cmd| <= 0.05 rad/s "
                            "and unchanged forward-speed bin."
                        )
                        print(
                            "Positions use ankle body origins; lift is above "
                            "the ankle's takeoff height, not sole clearance."
                        )
                        print(
                            "forward = world takeoff-to-landing displacement "
                            "projected onto takeoff heading; x = ankle relative to base."
                        )
                        print(
                            "Speed bin                       Foot  swings  "
                            "forward cm  base cm  lift cm  "
                            "takeoff x cm  landing x cm  swing ms"
                        )
                        for speed_bin_id, speed_bin_name in enumerate(
                            forward_speed_bin_names
                        ):
                            for foot_id, foot_name in enumerate(("left", "right")):
                                count = swing_count[speed_bin_id, foot_id].item()
                                if count:
                                    forward_cm = (
                                        swing_forward_sum[speed_bin_id, foot_id].item()
                                        / count * 100.0
                                    )
                                    lift_cm = (
                                        swing_lift_sum[speed_bin_id, foot_id].item()
                                        / count * 100.0
                                    )
                                    base_cm = (
                                        swing_root_forward_sum[
                                            speed_bin_id, foot_id
                                        ].item() / count * 100.0
                                    )
                                    duration_ms = (
                                        swing_duration_sum[
                                            speed_bin_id, foot_id
                                        ].item() / count * 1000.0
                                    )
                                    takeoff_cm = (
                                        swing_liftoff_reach_sum[
                                            speed_bin_id, foot_id
                                        ].item() / count * 100.0
                                    )
                                    landing_cm = (
                                        swing_touchdown_reach_sum[
                                            speed_bin_id, foot_id
                                        ].item() / count * 100.0
                                    )
                                    values_text = (
                                        f"{forward_cm:+10.2f}  {base_cm:+7.2f}  "
                                        f"{lift_cm:7.2f}  {takeoff_cm:+12.2f}  "
                                        f"{landing_cm:+12.2f}  {duration_ms:8.1f}"
                                    )
                                else:
                                    values_text = (
                                        "        n/a      n/a      n/a           n/a"
                                        "           n/a       n/a"
                                    )
                                print(
                                    f"{speed_bin_name:31s} {foot_name:5s} "
                                    f"{count:7d}  {values_text}"
                                )

                        print("\n[Half-cycle joint mirror diagnostics]")
                        print(
                            f"Lag: {symmetry_half_steps} control steps "
                            f"({symmetry_half_steps * env.unwrapped.step_dt * 1000:.1f} ms). "
                            "Only continuous straight walking in one speed bin."
                        )
                        print(
                            "Current left compares with past right; current right "
                            "compares with past left. Values are mean |offset sum| in rad."
                        )
                        print(
                            "Speed bin                       Side    samples  "
                            "hip roll  hip pitch     knee    ankle"
                        )
                        for speed_bin_id, speed_bin_name in enumerate(
                            forward_speed_bin_names
                        ):
                            for side_id, side_name in enumerate(("left", "right")):
                                count = symmetry_sample_count[
                                    speed_bin_id, side_id
                                ].item()
                                if count:
                                    mean_error = (
                                        symmetry_error_sum[speed_bin_id, side_id]
                                        / count
                                    )
                                    values_text = (
                                        f"{mean_error[0].item():9.4f} "
                                        f"{mean_error[1].item():8.4f} "
                                        f"{mean_error[2].item():8.4f} "
                                        f"{mean_error[3].item():8.4f}"
                                    )
                                else:
                                    values_text = "      n/a      n/a      n/a      n/a"
                                print(
                                    f"{speed_bin_name:31s} {side_name:5s} "
                                    f"{count:9d}  {values_text}"
                                )


                        print("\n[Leg motion diagnostics]")
                        print(
                            "Joint                  "
                            "mean |raw|   max |raw|   "
                            "mean target offset   actual range"
                        )

                        mean_raw_actions = (
                            raw_action_abs_sum / raw_action_samples
                        )
                        actual_joint_ranges = (
                            gait_joint_pos_max - gait_joint_pos_min
                        )

                        for i, joint_name in enumerate(gait_joint_names):
                            mean_raw = mean_raw_actions[i].item()
                            max_raw = raw_action_abs_max[i].item()
                            mean_target_offset = mean_raw * action_scale
                            actual_range = actual_joint_ranges[i].item()

                            print(
                                f"{joint_name:22s} "
                                f"{mean_raw:10.3f} "
                                f"{max_raw:10.3f} "
                                f"{mean_target_offset:18.4f} rad "
                                f"{actual_range:12.4f} rad"
                            )


                        phase_joint_pos_mean = (
                            phase_joint_pos_sum
                            / phase_joint_sample_count.clamp_min(1).unsqueeze(-1)
                        )
                        phase_joint_ranges = (
                            phase_joint_pos_max - phase_joint_pos_min
                        )

                        print("\n[Phase-binned leg diagnostics]")
                        print(
                            "phase 0 = left stance / right swing; "
                            "phase 1 = right stance / left swing"
                        )
                        print(
                            "Joint             "
                            "L stance mean/range   R stance mean/range   "
                            "L swing mean/range    R swing mean/range"
                        )

                        # 左关节 i 与右侧同名关节 i + 3 配对。
                        for i, joint_label in enumerate(
                            ["hip_pitch", "knee", "ankle"]
                        ):
                            left_stance_mean = phase_joint_pos_mean[0, i].item()
                            left_stance_range = phase_joint_ranges[0, i].item()

                            right_stance_mean = phase_joint_pos_mean[1, i + 3].item()
                            right_stance_range = phase_joint_ranges[1, i + 3].item()

                            left_swing_mean = phase_joint_pos_mean[1, i].item()
                            left_swing_range = phase_joint_ranges[1, i].item()

                            right_swing_mean = phase_joint_pos_mean[0, i + 3].item()
                            right_swing_range = phase_joint_ranges[0, i + 3].item()

                            print(
                                f"{joint_label:16s} "
                                f"{left_stance_mean:+.3f}/{left_stance_range:.3f}   "
                                f"{right_stance_mean:+.3f}/{right_stance_range:.3f}   "
                                f"{left_swing_mean:+.3f}/{left_swing_range:.3f}   "
                                f"{right_swing_mean:+.3f}/{right_swing_range:.3f}"
                            )


                        # ------------------------------------------------
                        # 输出左右 hip pitch 的目标跟踪诊断。
                        # ------------------------------------------------

                        # 避免理论上某个相位没有样本时发生除以零。
                        hip_sample_count = (
                            hip_phase_sample_count
                            .clamp_min(1)
                            .unsqueeze(-1)
                        )

                        # 计算各项平均值。
                        mean_hip_target_offset = (
                            hip_target_abs_offset_sum / hip_sample_count
                        )
                        mean_hip_actual_offset = (
                            hip_actual_abs_offset_sum / hip_sample_count
                        )
                        mean_hip_abs_tracking_error = (
                            hip_abs_tracking_error_sum / hip_sample_count
                        )
                        mean_hip_signed_tracking_error = (
                            hip_signed_tracking_error_sum / hip_sample_count
                        )

                        print("\n[Hip target tracking diagnostics]")
                        print(
                            "Phase   Hip                 "
                            "|target-default|  |actual-default|  "
                            "|target-actual|  target-actual"
                        )

                        # phase 0 与 phase 1 分别输出左右髋。
                        for phase_id, phase_label in enumerate(
                            [
                                "0 L-stance",
                                "1 R-stance",
                            ]
                        ):
                            for hip_id, hip_name in enumerate(
                                [
                                    "left_hip_pitch",
                                    "right_hip_pitch",
                                ]
                            ):
                                print(
                                    f"{phase_label:10s} "
                                    f"{hip_name:18s} "
                                    f"{mean_hip_target_offset[phase_id, hip_id].item():17.4f} "
                                    f"{mean_hip_actual_offset[phase_id, hip_id].item():18.4f} "
                                    f"{mean_hip_abs_tracking_error[phase_id, hip_id].item():15.4f} "
                                    f"{mean_hip_signed_tracking_error[phase_id, hip_id].item():14.4f}"
                                )

                        # ------------------------------------------------
                        # 输出实际执行器力矩诊断。
                        # ------------------------------------------------
                        torque_sample_count = (
                            hip_torque_sample_count
                            .clamp_min(1)
                            .unsqueeze(-1)
                        )

                        mean_hip_computed_torque = (
                            hip_computed_torque_abs_sum / torque_sample_count
                        )
                        mean_hip_applied_torque = (
                            hip_applied_torque_abs_sum / torque_sample_count
                        )
                        mean_hip_torque_utilization = (
                            hip_torque_utilization_sum / torque_sample_count
                        )
                        hip_torque_clipped_ratio = (
                            hip_torque_clipped_count / torque_sample_count
                        )

                        print("\n[Hip actuator torque diagnostics]")
                        print(
                            "Measured after env.step(): final physics step "
                            "of each control interval."
                        )
                        print(
                            "Phase              Hip                 "
                            "|computed|  |applied|  utilization  clipped"
                        )

                        for phase_id, phase_label in enumerate(
                            [
                                "0 L-stance/R-swing",
                                "1 R-stance/L-swing",
                            ]
                        ):
                            for hip_id, hip_name in enumerate(hip_actuator_names):
                                print(
                                    f"{phase_label:18s} "
                                    f"{hip_name:22s} "
                                    f"{mean_hip_computed_torque[phase_id, hip_id].item():10.4f} "
                                    f"{mean_hip_applied_torque[phase_id, hip_id].item():10.4f} "
                                    f"{mean_hip_torque_utilization[phase_id, hip_id].item():10.2%} "
                                    f"{hip_torque_clipped_ratio[phase_id, hip_id].item():8.2%}"
                                )

                        # ------------------------------------------------
                        # 输出 target--actual 的最佳时间对齐。
                        # ------------------------------------------------
                        latency_sample_count = (
                            hip_latency_sample_count
                            .clamp_min(1)
                            .unsqueeze(-1)
                        )
                        mean_hip_latency_error = (
                            hip_latency_error_abs_sum / latency_sample_count
                        )

                        print("\n[Hip target latency-alignment diagnostics]")
                        print(
                            "Best lag minimizes |historical target - current actual|."
                        )
                        print(
                            "Phase              Hip                 "
                            "current MAE  best lag  best lag time  best MAE  improvement"
                        )

                        for phase_id, phase_label in enumerate(
                            [
                                "0 L-stance/R-swing",
                                "1 R-stance/L-swing",
                            ]
                        ):
                            for hip_id, hip_name in enumerate(hip_actuator_names):
                                current_mae = mean_hip_latency_error[
                                    0,
                                    phase_id,
                                    hip_id,
                                ]
                                best_lag_step = torch.argmin(
                                    mean_hip_latency_error[
                                        :,
                                        phase_id,
                                        hip_id,
                                    ]
                                )
                                best_mae = mean_hip_latency_error[
                                    best_lag_step,
                                    phase_id,
                                    hip_id,
                                ]
                                best_lag_ms = (
                                    best_lag_step.item()
                                    * env.unwrapped.step_dt
                                    * 1000.0
                                )
                                improvement = current_mae - best_mae

                                print(
                                    f"{phase_label:18s} "
                                    f"{hip_name:22s} "
                                    f"{current_mae.item():11.4f} "
                                    f"{best_lag_step.item():8d} "
                                    f"{best_lag_ms:12.1f} ms "
                                    f"{best_mae.item():9.4f} "
                                    f"{improvement.item():11.4f}"
                                )

                        # ------------------------------------------------
                        # 输出近似直行时 hip yaw / roll 的姿态信息。
                        # ------------------------------------------------
                        yaw_roll_sample_count = max(
                            yaw_roll_straight_sample_count,
                            1,
                        )
                        mean_yaw_roll_target_offset = (
                            yaw_roll_target_offset_sum
                            / yaw_roll_sample_count
                        )
                        mean_yaw_roll_actual_offset = (
                            yaw_roll_actual_offset_sum
                            / yaw_roll_sample_count
                        )
                        yaw_roll_actual_range = (
                            yaw_roll_actual_pos_max - yaw_roll_actual_pos_min
                        )
                        yaw_roll_initial_offset = (
                            yaw_roll_initial_pos - yaw_roll_default_pos
                        )

                        print("\n[Hip yaw/roll posture diagnostics]")
                        print(
                            "Straight-command samples: "
                            f"{yaw_roll_straight_sample_count} "
                            "(|yaw command| <= 0.05 rad/s)"
                        )
                        print(
                            "Joint             default  initial-default  "
                            "mean target-default  mean actual-default  actual range"
                        )
                        for joint_id, joint_name in enumerate(
                            yaw_roll_joint_names
                        ):
                            print(
                                f"{joint_name:18s} "
                                f"{yaw_roll_default_pos[joint_id].item():+8.4f} "
                                f"{yaw_roll_initial_offset[joint_id].item():+15.4f} "
                                f"{mean_yaw_roll_target_offset[joint_id].item():+19.4f} "
                                f"{mean_yaw_roll_actual_offset[joint_id].item():+19.4f} "
                                f"{yaw_roll_actual_range[joint_id].item():12.4f}"
                            )

                        # 打印完整误差曲线，而不只打印最优点。
                        #
                        # 若误差持续下降到 8 step，说明搜索范围仍不够；
                        # 若误差在中间某一步开始回升，则该谷底才是可用的
                        # 闭环时延估计。
                        print("\n[Hip latency MAE curve]")
                        print(
                            "lag step / ms       "
                            "P0 left     P0 right    P1 left     P1 right"
                        )
                        for lag_step in range(max_hip_target_lag_steps + 1):
                            lag_ms = (
                                lag_step * env.unwrapped.step_dt * 1000.0
                            )
                            print(
                                f"{lag_step:3d} / {lag_ms:5.1f} ms   "
                                f"{mean_hip_latency_error[lag_step, 0, 0].item():9.4f} "
                                f"{mean_hip_latency_error[lag_step, 0, 1].item():11.4f} "
                                f"{mean_hip_latency_error[lag_step, 1, 0].item():10.4f} "
                                f"{mean_hip_latency_error[lag_step, 1, 1].item():11.4f}"
                            )


                        print(f"Failed episodes:    {failed_episodes}")
                        print(f"Success rate:       {success_rate:.2%}")
                        break

                # Keep existing video behavior.
                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break

            # 修改结束


                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            # close the simulator
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
