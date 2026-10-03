# Microduck Isaac Lab

基于 NVIDIA Isaac Lab 的 Microduck 双足机器人行走强化学习实验。本仓库包含机器人仿真资产、Isaac Lab 任务、RSL-RL 训练与评估脚本；目前主要用于仿真训练和步态诊断，**尚不代表策略已完成真机部署验证**。

本项目参考了 Pollen Robotics 的 [Microduck](https://github.com/pollen-robotics/microduck) 和 [microduck_rl](https://github.com/pollen-robotics/microduck_rl)。上游项目使用 mjlab/MuJoCo Warp；这里使用 Isaac Lab/Isaac Sim，训练配置及奖励函数并非上游的原样复制。

## 目录

```text
microduck_isaaclab/
├── microduck_mjcf/                   # MJCF 与网格资产
├── microduck_usd/                    # Isaac Sim 使用的 USD 机器人资产
│   └── robot_walk/robot_walk.usda
├── microduck_tasks/microduck_tasks/  # Isaac Lab 扩展、训练与评估脚本
└── upstream_microduck_rl/           # 本地上游源码参考，不提交到本仓库
```

运行时默认从仓库根目录读取 `microduck_usd/robot_walk/robot_walk.usda`。若资产放在别处，可设置 `MICRODUCK_ASSET_ROOT` 指向包含 `microduck_usd/` 的目录。USD 文件是运行所需资产，不作为普通生成缓存忽略。

## 环境与安装

需要能够运行 Isaac Sim 6.0、Isaac Lab 和 RSL-RL 的 Python 环境；本项目的扩展包要求 Python 3.12 或更新版本。以下命令从仓库根目录执行；请先激活自己的 Isaac Lab 环境：

```bash
cd microduck_tasks/microduck_tasks
python -m pip install -e source/microduck_tasks
```

## 训练与评估

当前主要实验任务为 `Template-Microduck-Walk-Phase-Symmetry-v0`，配置位于 `source/microduck_tasks/microduck_tasks/tasks/manager_based/microduck_tasks/microduck_walk_phase_symmetry_env_cfg.py`。当前实验使用速度自适应步态相位；固定半周期镜像奖励暂设为零，以免固定历史延迟与可变周期冲突。训练前请核对速度和转向指令范围是否符合本次实验。

在 `microduck_tasks/microduck_tasks` 目录下从头训练：

```bash
python scripts/rsl_rl/train.py \
  --task Template-Microduck-Walk-Phase-Symmetry-v0 \
  --num_envs 1024 \
  --max_iterations 5000 \
  --seed 42 \
  --run_name microduck_adaptive_phase_scratch5000_v1 \
  --headless
```

检查点默认保存到 `logs/rsl_rl/microduck_stand/` 下带时间戳的运行目录。评估时把 `YOUR_RUN_DIRECTORY_NAME` 和 `model_XXXX.pt` 替换为实际名称：

```bash
python scripts/rsl_rl/play.py \
  --task Template-Microduck-Walk-Phase-Symmetry-v0 \
  --num_envs 64 \
  --eval-episodes 800 \
  --seed 42 \
  --load_run YOUR_RUN_DIRECTORY_NAME \
  --checkpoint model_XXXX.pt \
  --headless
```

评估脚本中的部分相位分桶和半周期镜像诊断仍按固定 0.60 秒/15 控制步计算；同步更新诊断前，不应用这些列判断自适应周期的效果。可先比较成功率、各速度档速度误差、每次摆动的持续时间与抬脚高度、脚部滑移及可视化步态。

策略的 `policy` 观测不包含仿真真值 `base_lin_vel`；训练时 `critic` 可额外使用该特权观测。真机部署还需核对可用传感器、观测顺序、执行器模型、控制频率及延迟，不能仅凭仿真成功率判断部署就绪。

## Git 仓库边界

`upstream_microduck_rl/` 是本地参考仓库，已在根目录 `.gitignore` 中排除，不随本项目提交。`microduck_tasks/microduck_tasks/` 的源码由根仓库直接管理；原内层 Git 元数据保存在被忽略的 `/.git-backups/microduck_tasks.git/`，需要时可以移回原位置恢复。训练日志、检查点和缓存也不会提交。

## 许可与来源

根目录 [LICENSE](LICENSE) 中的 MIT 条款仅适用于项目作者有权以 MIT 发布的原创内容。来源于 Isaac Lab 模板的文件保留其 BSD-3-Clause 标头；本地上游 `microduck_rl` 的代码保留 Apache-2.0 许可。上游 README 另声明其 3D 模型文件采用 Creative Commons BY-SA-NC；本仓库的 MJCF、网格及由其转换的 USD 资产不能仅凭根目录 MIT 文件认定为 MIT 授权。分发这些资产前请核对原始来源及适用许可。
