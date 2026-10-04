from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg 
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.sensors import ContactSensorCfg

from microduck_tasks.assets.microduck_cfg import MICRODUCK_CFG

from . import mdp

@configclass
class MicroduckStandSceneCfg(InteractiveSceneCfg):
  """Microduck 在平地上的最小测试场景。"""

  # 继承自 InteractiveSceneCfg，默认包含 ground_plane、sky_light、camera 等。
  # 可在此添加其他场景元素，如障碍物、地形等。

  # 场景中包含的机器人资产
  ground = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
  )

  robot: ArticulationCfg = MICRODUCK_CFG.replace(
    prim_path="{ENV_REGEX_NS}/robot",
  )

  # 记录机器人刚体的接触、离地和接触持续时间。
  feet_contact = ContactSensorCfg(
      prim_path="{ENV_REGEX_NS}/robot/.*",
      history_length=3,
      track_air_time=True,
      debug_vis=False,
  )

  dome_lite = AssetBaseCfg(
    prim_path="/World/DomeLight",
    spawn=sim_utils.DomeLightCfg(
      intensity=1000.0,
      color=(0.9, 0.9, 0.9),    
    ),
  )


@configclass
class ActionsCfg:
  """策略动作空间配置"""

  joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    ],
    scale=0.15,
    use_default_offset=True,
  )


@configclass
class CommandsCfg:
    """指令配置"""
    base_velocity=mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), #每 10 秒重新抽一次目标速度
        rel_standing_envs=0.1, #10% 的环境会被指定为站立命令,策略不能只会走，还要在“没有移动命令”时稳定站住。
        rel_heading_envs=0.0,
        heading_command=False, #关闭“朝向目标”模式
        debug_vis=True, #在 Kit 可视化中显示速度指令的调试标记
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.04, 0.18),
            lin_vel_y=(0.0, 0.0),
            # ang_vel_z=(-0.30, 0.30),
            ang_vel_z=(-0.00, 0.00),
        ),
    )

# 参数字典
GAIT_PHASE_PARAMS = {
    "command_name": "base_velocity",
    # 中高速时的完整步态周期：左、右各完成一次摆动为一个周期。
    "period_s": 0.60,

    # 慢速时采用的完整步态周期。
    "slow_period_s": 0.60,

    # 线速度低于 slow_speed 时使用慢速周期；
    # 高于 fast_speed 时使用 period_s；中间连续插值。
    "slow_speed": 0.04,
    "fast_speed": 0.12,

    # 速度低于该值时视为站立，不推进自适应步态相位。
    "command_threshold": 0.02,
}

GAIT_CONTACT_SCHEDULE_PARAMS = {
    "slow_stance_fraction": 0.62,
    "fast_stance_fraction": 0.52,
    "transition_fraction": 0.04,
}

@configclass
class ObservationsCfg:
    """策略观测空间配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 实机不一定能获取到噪声低且准确的速度反馈
        velocity_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        gait_phase = ObsTerm(
            func=mdp.gait_phase_sin_cos,
            params={
                **GAIT_PHASE_PARAMS,
            },
        )        
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class PrivilegedCfg(ObsGroup):
        # 仿真真值只供训练时的 critic 使用；部署的 actor 看不到它。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    """第一版站立任务的奖励。"""

    # 每个未失败的时间步给正奖励。
    alive = RewTerm(func=mdp.is_alive,weight=1.0)

    # 倾倒等失败终止时给一次大惩罚；正常超时不罚。
    termination_penalty = RewTerm(
      func=mdp.is_terminated,
      weight=-10.0,
    )

    # 惩罚躯干偏离竖直方向。
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
    )

    # 惩罚前后/左右翻滚角速度
    base_ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    # # 训练站立，训练走路时注释
    # # 鼓励所有关节接近默认姿态。
    # joint_deviation = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.02,
    # )

    # 惩罚动作在连续时间步的变化过大。
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.005,
    )

    # 奖励速度跟踪
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": 0.10,
        },
    )

    # 奖励跟踪yaw角
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.25,
        },
    )

    # 鼓励行走时形成“单脚支撑、另一脚摆动”的节奏。
    air_time = RewTerm(
        func=mdp.biped_air_time,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.15,
            "command_threshold": 0.02,
            "sensor_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=[
                    "ankle_left",
                    "ankle_right"
                ],
            ),
        },
    )

    # 接触地面时，惩罚脚端水平拖滑。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.6,
        params={
            "sensor_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=[
                    "ankle_left",
                    "ankle_right"
                ],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "ankle_left",
                    "ankle_right"
                ],
            ),
        },
    )

    # 控制步态相位
    phase_foot_contact = RewTerm(
        func=mdp.phase_foot_contact,
        weight=1.0,
        params={
            **GAIT_PHASE_PARAMS,
            **GAIT_CONTACT_SCHEDULE_PARAMS,
            "force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=["ankle_left", "ankle_right"],
                preserve_order=True,
            ),
        },
    )

    phase_swing_contact_penalty = RewTerm(
        func=mdp.phase_swing_contact_penalty,
        weight=-0.0,
        params={
            **GAIT_PHASE_PARAMS,
            **GAIT_CONTACT_SCHEDULE_PARAMS,
            "yaw_scale": 0.30,
            "force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=["ankle_left", "ankle_right"],
                preserve_order=True,
            ),
        },
    )

    # 奖励摆动脚相对起跳姿态的抬升，减少低空拖脚。
    swing_foot_lift = RewTerm(
        func=mdp.SwingFootLiftReward,
        weight=0.20,
        params={
            **GAIT_PHASE_PARAMS,
            **GAIT_CONTACT_SCHEDULE_PARAMS,

            # 目标抬升 0.8 cm；达到后奖励饱和。
            # 先从较保守的数值开始，避免学成高抬腿或跳跃。
            "target_lift": 0.008,

            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["ankle_left", "ankle_right"],
                preserve_order=True,
            ),
        },
    )

    contact_duty_balance = RewTerm(
        func=mdp.ContactDutyBalance,
        weight=-0.0,
        params={
            **GAIT_PHASE_PARAMS,
            "yaw_threshold": 0.05,
            "min_cycle_time_s": 0.45,
            "force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=["ankle_left", "ankle_right"],
                preserve_order=True,
            ),
        },
    )

    # 直行时鼓励左右腿在半周期后呈镜像轨迹；
    # 偏航命令增大时自动减弱，避免限制未来的差速转弯。
    # v2：只有髋、膝确实在主动摆动时才奖励左右腿半周期后的镜像关系
    half_cycle_active_joint_symmetry = RewTerm(
        func=mdp.HalfCycleActiveJointSymmetry,
        weight=0.00,
        params={
            **GAIT_PHASE_PARAMS,
            "yaw_scale": 0.30,
            "std": 0.25,
            "roll_std": 0.25,
            "roll_fraction": 0.20,
            "min_motion": 0.10,
            "joint_weights": (1.0, 1.5, 0.2),
            "motion_joint_weights": (1.0, 1.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_roll",
                    "left_hip_pitch",
                    "left_knee",
                    "left_ankle",
                    "right_hip_roll",
                    "right_hip_pitch",
                    "right_knee",
                    "right_ankle",
                ],
                # 强制 Isaac Lab 保持上述关节顺序。
                #
                # 若不加，框架可能按内部关节编号重排，
                # 左右配对就会出错。
                preserve_order=True,
            ),
        },
    )


    hip_yaw_target_neutral_deadband = RewTerm(
        func=mdp.hip_yaw_target_neutral_deadband,
        weight=-0.03,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.02,
            # 当目标 yaw 速度达到 0.30 rad/s 时，
            # 该项惩罚自动衰减为 0，允许转弯。
            "yaw_scale": 0.30,
            "action_name": "joint_pos",
            # action 顺序：
            "action_indices": (0, 1),

            # 约 ±5.7 度内允许自由调整。
            "free_yaw_rad": 0.10,

            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_hip_yaw", "right_hip_yaw"],
                preserve_order=True,
            ),
        },
    )


@configclass
class TerminationsCfg:
    """结束并重置 episode 的条件。"""

    # 正常达到 episode 时长后结束；这不代表失败。
    time_out = DoneTerm(
       func=mdp.time_out,
       time_out=True,
    )

    # 躯干根节点低于 6 cm，通常意味着已跌倒或趴在地上。
    base_too_low = DoneTerm(
       func=mdp.root_height_below_minimum,
       params={"minimum_height": 0.06},
    )

    # 躯干相对竖直方向倾斜超过约 46°，视为跌倒。
    bad_orientation = DoneTerm(
       func=mdp.bad_orientation,
       params={"limit_angle": 0.8},
    )


@configclass
class EventCfg:
    """鲁棒站立,reset时施加初始扰动"""

    # # 每个环境启动时随机分配机器人碰撞材质。
    # # 材质在该环境整个运行期间保持不变。
    # randomize_robot_material = EventTerm(
    #     func = mdp.randomize_rigid_body_material,
    #     mode = "startup",
    #     params = {
    #         "asset_cfg" : SceneEntityCfg(
    #             "robot",
    #             body_names = ".*",
    #         ),
    #         "static_friction_range" : (0.15, 1.2),
    #         "dynamic_friction_range" : (0.1, 1.0),
    #         "restitution_range" : (0.0, 0.0),
    #         "num_buckets" : 64,
    #         "make_consistent" : True,
    #     },
    # )

    # # 随机化躯干负载质量
    # randomize_robot_mass = EventTerm(
    #     func = mdp.randomize_rigid_body_mass,
    #     mode = "startup",
    #     params = {
    #         "asset_cfg" : SceneEntityCfg(
    #             "robot",
    #             body_names = "trunk_base",
    #         ),
    #         "mass_distribution_params" : (0.8, 1.2),
    #         "operation" : "scale",
    #         "distribution" : "uniform",
    #         "recompute_inertia" : True,
    #     },
    # )

    # # 随机质心漂移
    # randomize_trunk_com = EventTerm(
    #     func = mdp.randomize_rigid_body_com,
    #     mode = "startup",
    #     params = {
    #         "asset_cfg" : SceneEntityCfg(
    #             "robot",
    #             body_names = "trunk_base",
    #         ),
    #         "com_range" : {
    #             "x" : (-0.005, 0.005),
    #             "y" : (-0.005, 0.005),
    #             "z" : (-0.003, 0.003),
    #         },
    #     },
    # )

    # # 随机执行器参数
    # randomize_leg_actuator_gains = EventTerm(
    #     func = mdp.randomize_actuator_gains,
    #     mode = "startup",
    #     params = {
    #         "asset_cfg" : SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "left_hip_yaw",
    #                 "right_hip_yaw",
    #                 "left_hip_roll",
    #                 "right_hip_roll",
    #                 "left_hip_pitch",
    #                 "right_hip_pitch",
    #                 "left_knee",
    #                 "right_knee",
    #                 "left_ankle",
    #                 "right_ankle",
    #             ],
    #         ),
    #         "stiffness_distribution_params" : (0.85, 1.15),
    #         "damping_distribution_params" : (0.80, 1.20),
    #         "operation" : "scale",
    #         "distribution" : "uniform",
    #     },
    # )

    # reset时改变初始状态
    reset_root_state = EventTerm(
      func = mdp.reset_root_state_uniform,
      mode = "reset",
      params = {
          "pose_range" : {
              "roll": (-0.05,0.05),
              "pitch": (-0.05,0.05),
          },
          "velocity_range" : {
              "x" : (-0.1, 0.1),
              "y" : (-0.1, 0.1),

              "roll" : (-0.25, 0.25),
              "pitch" : (-0.25, 0.25),
          },
          "asset_cfg" : SceneEntityCfg("robot"),
      },
    )

    # reset时扰动腿关节
    reset_leg_joints = EventTerm(
        func = mdp.reset_joints_by_offset,
        mode = "reset",
        params = {
            "position_range" : (-0.03, 0.03),
            "velocity_range" : (0.0, 0.0),
            "asset_cfg" : SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_yaw",
                    "left_hip_roll",
                    "left_hip_pitch",
                    "left_knee",
                    "left_ankle",
                    "right_hip_yaw",
                    "right_hip_roll",
                    "right_hip_pitch",
                    "right_knee",
                    "right_ankle",
                ],
            ),
        },
    )

    # # 初速扰动
    # push_robot = EventTerm(
    #     func = mdp.push_by_setting_velocity,
    #     mode = "interval",
    #     interval_range_s = (3.0, 6.0),
    #     params = {
    #         "velocity_range" : {
    #             "x" : (-0.2, 0.2),
    #             "y" : (-0.2, 0.2),
    #         },
    #         "asset_cfg" : SceneEntityCfg("robot"),
    #     },
    # )

@configclass
class MicroduckWalkPhaseSymmetryEnvCfg(ManagerBasedRLEnvCfg):
   """用于验证 Microduck 站立控制的最小强化学习环境。"""

   scene: MicroduckStandSceneCfg = MicroduckStandSceneCfg(
      num_envs=64,
      env_spacing=1.0,
   )

   observations:ObservationsCfg = ObservationsCfg()
   actions:ActionsCfg = ActionsCfg()
   commands:CommandsCfg = CommandsCfg()
   rewards:RewardsCfg = RewardsCfg()
   events:EventCfg = EventCfg()
   terminations:TerminationsCfg = TerminationsCfg()

   def __post_init__(self) -> None:
        # 每 4 个物理步执行一次策略动作.
        self.decimation = 4

        # 单个 episode 最长 10 秒。
        self.episode_length_s = 10.0

        # 物理仿真 200 Hz；策略控制 200 / 4 = 50 Hz
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation

        # 方便观察小尺寸 Microduck 的初始视角。
        self.viewer.eye = (0.6, 0.6, 0.35)
        self.viewer.lookat = (0.0, 0.0, 0.1)


