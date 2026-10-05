from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg



# 默认目录结构：
#
# microduck_isaaclab/
# ├── microduck_usd/
# │   └── robot_walk/robot_walk.usda
# └── microduck_tasks/
#     └── microduck_tasks/source/microduck_tasks/microduck_tasks/assets/
#         └── microduck_cfg.py
#
# 若资产位于其他位置，可在运行前设置：
# export MICRODUCK_ASSET_ROOT=/path/to/microduck_isaaclab

_DEFAULT_ASSET_ROOT = Path(__file__).resolve().parents[6]
ASSET_ROOT = Path(os.environ.get("MICRODUCK_ASSET_ROOT", _DEFAULT_ASSET_ROOT))

MICRODUCK_USD_PATH = ASSET_ROOT / "microduck_usd" / "robot_walk" / "robot_walk.usda"


MICRODUCK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(MICRODUCK_USD_PATH),
        # 是否在“机器人根 prim 下的所有刚体”上自动添加
        # PhysX Contact Report API
        activate_contact_sensors=False,
         # 对 USD 中所有刚体施加的 PhysX 刚体属性覆盖
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # 是否让 PhysX 保留刚体加速度相关状态。
            # 这是 PhysX 求解器内部状态选项，不是“给机器人增加加速度”。
            # 通常保持 False，沿用常规刚体仿真行为即可。
            retain_accelerations=False,
            linear_damping=0.0,   # 刚体线性阻尼，单位可理解为对平移速度施加的全局阻力。
            angular_damping=0.0,  # 刚体角阻尼，作用于旋转速度的全局阻尼。
            max_linear_velocity=1000.0,    # 刚体允许的最大线速度，单位 m/s。
            max_angular_velocity=1000.0,   # 刚体允许的最大角速度，单位 rad/s。
            max_depenetration_velocity=1.0,  # 接触体发生轻微穿透时，PhysX 允许用多大的最大速度 将两个物体分离，单位 m/s。
        ),
         # 对整套关节刚体系统（articulation）设置的 PhysX 求解器属性。
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 是否启用机器人自身不同 link 之间的碰撞。
            enabled_self_collisions=True,
            # 每个 physics step 中，位置约束的求解迭代次数。
            solver_position_iteration_count=8,
            # 每个 physics step 中，速度约束的求解迭代次数。
            solver_velocity_iteration_count=4,
        ),
    ),
    # 采用 upstream 的 STAND/HOME 关键帧作为初始姿态和零动作目标。
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.20),
        joint_pos={
            "left_hip_yaw": 0.0,
            "left_hip_roll": -0.0873,
            "left_hip_pitch": -0.457924,
            "left_knee": -0.004940,
            "left_ankle": 0.452984,

            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0873,
            "right_hip_pitch": 0.457924,
            "right_knee": 0.004940,
            "right_ankle": -0.452984,

            "neck_pitch": 0.349066,
            "head_pitch": 0.349066,
            "head_yaw": 0.0,
            "head_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 对应 MJCF chosen_actuator：
        # kp=0.55, forcerange=[-0.96, 0.96],
        # joint damping=0.053, armature=0.0018。
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
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
            effort_limit_sim=0.96,
            velocity_limit_sim=10.0,
            # stiffness=0.55,
            # damping=0.053,
            stiffness = {
                "left_hip_pitch|right_hip_pitch": 1.10,
                (
                    "left_hip_yaw|left_hip_roll|left_knee|left_ankle|"
                    "right_hip_yaw|right_hip_roll|right_knee|right_ankle"
                ):0.55,
            },
            damping={
                "left_hip_pitch|right_hip_pitch": 0.075,
                (
                    "left_hip_yaw|left_hip_roll|left_knee|left_ankle|"
                    "right_hip_yaw|right_hip_roll|right_knee|right_ankle"
                ):0.053,
            },
            armature=0.0018,
            min_delay=2,
            max_delay=8,
        ),
        # 第一版站立任务不让策略控制头部；PD 负责将其保持在默认姿态。
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "neck_pitch",
                "head_pitch",
                "head_yaw",
                "head_roll",
            ],
            effort_limit_sim=0.96,
            velocity_limit_sim=10.0,
            stiffness=0.55,
            damping=0.053,
            armature=0.0018,
        ),
    },
)