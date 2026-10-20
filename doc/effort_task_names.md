# Effort 任务命名

任务 ID 逐项写明训练默认配置，格式为：

`Unitree-G1-Effort-0PD-{地形}-{IMU}-{关节位置}-{关节速度}-{相位}[-HeightScan]-{历史}-{编码器}-{奖励}-{命令课程}[-TerrainCurr]`

每个字段都显式写出取值，没有“省略即默认”的字段。所有 effort 任务都输出绝对力矩动作，
Actor 均包含速度命令和上一策略动作；观测字段只描述 Actor，Critic 仍使用完整的特权观测。
所有任务的 Actor 和 Critic 都启用 EmpiricalNormalization，因此不再写进 ID。
`tests/test_effort_task.py` 会逐个解析任务 ID，并核对它和实际配置一致。

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `0PD` | — | 主动控制增益 `Kp = Kd = 0`；被动关节阻尼仍按机器人配置保留 |
| 地形 | `Flat` / `Rough` | 平地 / 随机地形 |
| IMU | `IMU` / `NoIMU` | Actor 保留 / 移除机身角速度与投影重力 |
| 关节位置 | `JPos` / `JPosNoAnkle` | Actor 观测全部 / 除四个踝关节外的受控关节相对位置；踝关节仍受控 |
| 关节速度 | `JVel` / `NoJVel` | Actor 保留 / 移除关节速度 |
| 相位 | `Phase` / `NoPhase` | Actor 保留 / 移除步态相位 |
| `HeightScan` | 仅 `Rough` | Actor 包含地形高度扫描 |
| 历史 | `H1` / `H5` | Actor 和 Critic 使用一帧 / 五帧观测（含当前帧） |
| 编码器 | `MLP` / `TimeMHA` / `GroupTimeFusion` | 展平历史 MLP / 过去帧时间自注意力加当前帧编码 / 逐关节条件检索加全身融合 |
| 奖励 | `L1Pose` / `SpeedPose` | 姿态项用 L1 偏离加 `stand_still` / 按速度平滑放宽的分组高斯姿态奖励 |
| 命令课程 | `CmdCurr` / `NoCurr` | 启用命令速度课程 / 速度命令从第 0 步起使用满量程 |
| `TerrainCurr` | 仅 `Rough` | 启用地形难度课程 |

Play 模式按原有配置关闭课程；CLI 覆盖观测、归一化或课程后，应以该次运行保存的
`params/env.yaml` 和 `params/agent.yaml` 为准。

实验目录由完整任务 ID 去掉 `Unitree-` 前缀、转小写、将连字符换成下划线得到，
每个任务独立一个目录，自动选择 checkpoint 时不会混入其他任务的运行。

## 当前任务

| 任务 ID | Actor 观测项 |
| --- | --- |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-L1Pose-CmdCurr` | `base_ang_vel`, `projected_gravity`, `command`, `phase`, `joint_pos`, `joint_vel`, `actions` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-L1Pose-CmdCurr` | `base_ang_vel`, `projected_gravity`, `command`, `phase`, `joint_pos`, `joint_vel`, `actions` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H1-MLP-L1Pose-NoCurr` | `base_ang_vel`, `projected_gravity`, `command`, `joint_pos`, `actions` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-CmdCurr` | `base_ang_vel`, `projected_gravity`, `command`, `joint_pos`, `actions` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-NoCurr` | `base_ang_vel`, `projected_gravity`, `command`, `joint_pos`, `actions` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-MLP-L1Pose-CmdCurr` | `base_ang_vel`, `projected_gravity`, `command`, `joint_pos`, `actions` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-MLP-L1Pose-NoCurr` | `base_ang_vel`, `projected_gravity`, `command`, `joint_pos`, `actions` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-TimeMHA-L1Pose-CmdCurr` | `base_ang_vel`, `projected_gravity`, `command`, `joint_pos`, `actions` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-TimeMHA-L1Pose-NoCurr` | `base_ang_vel`, `projected_gravity`, `command`, `joint_pos`, `actions` |
| `Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-NoPhase-H5-MLP-L1Pose-CmdCurr` | `command`, `joint_pos`, `actions` |
| `Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-NoPhase-H5-TimeMHA-L1Pose-CmdCurr` | `command`, `joint_pos`, `actions` |
| `Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-CmdCurr` | `command`, `joint_pos`, `actions` |
| `Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-L1Pose-CmdCurr-TerrainCurr` | `base_ang_vel`, `projected_gravity`, `command`, `phase`, `joint_pos`, `joint_vel`, `actions`, `height_scan` |
| `Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-L1Pose-CmdCurr-TerrainCurr` | `base_ang_vel`, `projected_gravity`, `command`, `phase`, `joint_pos`, `joint_vel`, `actions`, `height_scan` |

## 当前 GroupTimeFusion 任务

| Actor 配置 | 任务 ID |
| --- | --- |
| 保留 IMU、隐藏踝位置、无关节速度 | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-CmdCurr` |
| 同上，速度命令从第 0 步就是满量程 | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-NoCurr` |
| 无 IMU、隐藏踝位置、无关节速度 | `Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-CmdCurr` |

这些任务使用同一套 actor：可学习关节身份、当前全局状态和当前命令进入 Query，每个关节检索自己的五帧历史，摘要层固定做全身融合。保留 IMU 的任务没有步态相位。`NoCurr` 去掉命令速度课程，采样范围保持 `lin_vel_x=(-2, 2)`、`lin_vel_y=(-1, 1)`、`ang_vel_z=(-1, 1)`。同样的观测和命令范围也有 MLP 版本 `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-MLP-L1Pose-NoCurr`，以及时间自注意力版本 `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-TimeMHA-L1Pose-NoCurr`。`Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H1-MLP-L1Pose-NoCurr` 与无课程五帧 MLP 使用同一套观测项和命令范围，但 Actor 和 Critic 都只有当前帧。训练入口 `train_effort_joint_xattn.sh` 指向无课程的五帧 MLP 任务。

`JointXAttn`、`JointSharedXAttn` 和没有融合开关的 `GroupTime` 已从本分支移除。
旧 checkpoint 不能加载到 `GroupTimeFusion`，需要按新结构重新训练。

## 运行

```bash
uv run python scripts/list_envs.py --keyword Effort
```

## 旧名称迁移表

旧任务 ID 不再注册；已有日志、checkpoint 和导出文件不移动。旧日志仍在原来的实验目录下，
可用 `--checkpoint-file` 直接指定。Go2 和 G1-23Dof 的 effort 任务已移除。

| 旧任务 ID | 当前任务 ID |
| --- | --- |
| `Unitree-G1-23Dof-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm` | 已移除，不再注册 |
| `Unitree-G1-23Dof-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm` | 已移除，不再注册 |
| `Unitree-G1-23Dof-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm` | 已移除，不再注册 |
| `Unitree-G1-23Dof-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm` | 已移除，不再注册 |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H1-MLP-NoCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H1-MLP-L1Pose-NoCurr` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-CmdCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-NoCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-NoCurr` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-CmdCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-MLP-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-NoCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-MLP-L1Pose-NoCurr` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-TimeMHA-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-NoCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-TimeMHA-L1Pose-NoCurr` |
| `Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-MLP-CmdCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-NoPhase-H5-MLP-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-NoPhase-H5-TimeMHA-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-H5-GroupTimeFusion-CmdCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-L1Pose-CmdCurr-TerrainCurr` |
| `Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm` | `Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-L1Pose-CmdCurr-TerrainCurr` |
| `Unitree-Go2-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm` | 已移除，不再注册 |
| `Unitree-Go2-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm` | 已移除，不再注册 |
| `Unitree-Go2-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm` | 已移除，不再注册 |
| `Unitree-Go2-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm` | 已移除，不再注册 |
| `Unitree-G1-Effort-Flat` | `Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-Flat-MHA` | `Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-Rough` | `Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-L1Pose-CmdCurr-TerrainCurr` |
| `Unitree-G1-Effort-Rough-MHA` | `Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-L1Pose-CmdCurr-TerrainCurr` |
| `Unitree-G1-23Dof-Effort-Flat` | 已移除，不再注册 |
| `Unitree-G1-23Dof-Effort-Flat-MHA` | 已移除，不再注册 |
| `Unitree-G1-23Dof-Effort-Rough` | 已移除，不再注册 |
| `Unitree-G1-23Dof-Effort-Rough-MHA` | 已移除，不再注册 |
| `Unitree-Go2-Effort-Flat` | 已移除，不再注册 |
| `Unitree-Go2-Effort-Flat-MHA` | 已移除，不再注册 |
| `Unitree-Go2-Effort-Rough` | 已移除，不再注册 |
| `Unitree-Go2-Effort-Rough-MHA` | 已移除，不再注册 |
| `Unitree-G1-Effort-Flat-Partial` | `Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-NoPhase-H5-MLP-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-Flat-Partial-MHA` | `Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-NoPhase-H5-TimeMHA-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-Flat-Partial-XAttn` | 已移除，不再注册 |
| `Unitree-G1-Effort-Flat-Partial-Phase1` | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-MLP-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-Flat-Partial-Phase1-MHA` | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-TimeMHA-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-Flat-Partial-Phase1-XAttn` | 已移除，不再注册 |
| `Unitree-G1-Effort-Flat-Partial-NoAnkle-XAttn` | 已移除，不再注册 |
| `Unitree-G1-Effort-Flat-Partial-Phase1-XAttn-NoCurriculum-FP32` | 已移除，不再注册 |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H8-GroupTime-CmdCurr-ObsNorm` | 已移除。重新训练 `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H8-GroupTimeFusion-CmdCurr-ObsNorm` | 已移除。重新训练 `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-CmdCurr` |
| `Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-H8-GroupTime-CmdCurr-ObsNorm` | 已移除。重新训练 `Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-NoPhase-H5-GroupTimeFusion-L1Pose-CmdCurr` |
