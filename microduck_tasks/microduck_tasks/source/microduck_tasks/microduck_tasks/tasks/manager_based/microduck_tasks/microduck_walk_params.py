"""Microduck 步态、命令门控与站立模式的共享参数。"""

from typing import Final

# 所有相关项使用同一个速度命令。
COMMAND_NAME: Final = "base_velocity"

# 判定“站立”的统一门限。
STANDING_LINEAR_SPEED_THRESHOLD: Final = 0.02  # m/s
STANDING_YAW_RATE_THRESHOLD: Final = 0.05  #rad/s

# 步态时钟参数。
# 中高速时的完整步态周期：左、右各完成一次摆动为一个周期。
GAIT_PERIOD_S: Final = 0.60
# 慢速时采用的完整步态周期。
GAIT_SLOW_PERIOD_S: Final = 0.80
# 线速度低于 slow_speed 时使用慢速周期；
# 高于 fast_speed 时使用 period_s；中间连续插值。
GAIT_SLOW_SPEED: Final = 0.04
GAIT_FAST_SPEED: Final = 0.12

# 参数字典
# 仅判断“是否有明确行走命令”时使用。
WALKING_GATE_PARAMS: Final = {
    "command_name": COMMAND_NAME,
    "command_threshold": STANDING_LINEAR_SPEED_THRESHOLD,
}

GAIT_PHASE_PARAMS: Final = {
    **WALKING_GATE_PARAMS,
    "period_s": GAIT_PERIOD_S,
    "slow_period_s": GAIT_SLOW_PERIOD_S,
    "slow_speed": GAIT_SLOW_SPEED,
    "fast_speed": GAIT_FAST_SPEED,
}

STANDING_GATE_PARAMS: Final = {
    "command_name": COMMAND_NAME,
    "command_threshold": STANDING_LINEAR_SPEED_THRESHOLD,
    "yaw_threshold": STANDING_YAW_RATE_THRESHOLD,
}

# 步态接触日程：
# slow_stance_fraction / fast_stance_fraction 分别决定低速、高速时
# 单脚处于支撑期的周期占比；transition_fraction 是落脚和离地边界的平滑区宽度。
GAIT_CONTACT_SCHEDULE_PARAMS: Final = {
    "slow_stance_fraction": 0.62,
    "fast_stance_fraction": 0.52,
    "transition_fraction": 0.04,
}

# 转弯软门控尺度。
#
# 当 |yaw_rate| = TURNING_YAW_SCALE 时，常见的
# exp(-(yaw_rate / yaw_scale)^2) 门控会降至 exp(-1) ≈ 0.368。
# 数值越小，转弯时越快解除步态对称、摆动接触等直行约束。
TURNING_YAW_SCALE: Final = 0.30  # rad/s

TURNING_SOFT_GATE_PARAMS: Final = {
    "yaw_scale": TURNING_YAW_SCALE,
}

# 足端接触判定阈值，单位 N。
#
# 接触传感器报告的合力范数超过该值时，才将该脚视为接触地面。
# 过低会把轻微碰撞或数值噪声视为落脚；过高可能漏掉真实接触。
FOOT_CONTACT_FORCE_THRESHOLD: Final = 1.0

FOOT_CONTACT_FORCE_PARAMS: Final = {
    "force_threshold": FOOT_CONTACT_FORCE_THRESHOLD,
}
