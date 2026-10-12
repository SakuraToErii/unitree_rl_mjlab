# Effort 任务命名

任务 ID 描述训练默认配置，格式为：

`Unitree-{机器人}-Effort-0PD-{地形}-{Actor 观测}-{历史}-{编码器}-{课程}-ObsNorm`

所有 effort 任务都输出绝对力矩动作，Actor 均包含速度命令和上一策略动作。
观测字段只描述 Actor；Critic 仍使用其原有的完整、特权观测。

| 字段 | 含义 |
| --- | --- |
| `0PD` | 主动控制增益 `Kp = Kd = 0`；被动关节阻尼仍按机器人配置保留 |
| `Flat / Rough` | 平地 / 随机地形 |
| `IMU / NoIMU` | Actor 保留 / 移除机身角速度与投影重力 |
| `JPos` | Actor 观测全部受控关节的相对位置 |
| `JPosNoAnkle` | Actor 移除四个踝关节的位置；四个踝关节仍受控，动作历史仍保留 |
| `JVel / NoJVel` | Actor 保留 / 移除直接关节速度观测 |
| `Phase` | Actor 包含步态相位；没有此字段的任务不包含相位 |
| `HeightScan` | Actor 包含地形高度扫描 |
| `H5` | Actor 和 Critic 都使用五帧观测，含当前帧 |
| `H8` | Actor 和 Critic 都使用八帧观测，含当前帧 |
| `MLP` | Actor 将历史按观测项展平后输入 MLP |
| `TimeMHA` | 过去帧做时间自注意力，再拼接当前帧编码 |
| `GroupTime` | 每个关节用当前拍检索自己的八拍 `(q, τ, m)` 历史；IMU 单独检索 |
| `GroupTimeFusion` | 在 `GroupTime` 的时间摘要之后再加一层关节间残差自注意力 |
| `CmdCurr` | 训练启用命令速度课程 |
| `CmdFixed` | 训练固定命令采样范围，不启用命令速度课程 |
| `TerrainCurr` | 训练启用地形难度课程 |
| `ObsNorm` | Actor 和 Critic 均启用 EmpiricalNormalization |

名称描述注册时的训练配置。Play 模式按原有配置关闭课程；CLI 覆盖观测、
归一化或课程后，应以该次运行保存的 `params/env.yaml` 和 `params/agent.yaml` 为准。

实验目录由完整任务 ID 去掉 `Unitree-` 前缀、转小写、将连字符换成下划线得到。
不同地形、观测、编码器和课程的任务使用独立目录。
旧任务 ID 不再注册；已有日志、checkpoint 和导出文件不移动。
`JointXAttn` 与 `JointSharedXAttn` 已从本分支移除，旧 checkpoint 不能加载到 `GroupTime`。

## 当前 GroupTime 任务

| Actor 配置 | 任务 ID |
| --- | --- |
| 保留 IMU、隐藏踝位置、无关节速度 | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H8-GroupTime-CmdCurr-ObsNorm` |
| 无 IMU、隐藏踝位置、无关节速度 | `Unitree-G1-Effort-0PD-Flat-NoIMU-JPosNoAnkle-NoJVel-H8-GroupTime-CmdCurr-ObsNorm` |
| 保留 IMU，时间摘要后再做关节融合 | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H8-GroupTimeFusion-CmdCurr-ObsNorm` |

保留 IMU 的任务没有步态相位。

## 运行

```bash
uv run python scripts/list_envs.py --keyword Effort
```

## 旧名称迁移表

| 旧任务 ID | 当前任务 ID |
| --- | --- |
| `Unitree-G1-Effort-Flat` | `Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm` |
| `Unitree-G1-Effort-Flat-MHA` | `Unitree-G1-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm` |
| `Unitree-G1-Effort-Rough` | `Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm` |
| `Unitree-G1-Effort-Rough-MHA` | `Unitree-G1-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm` |
| `Unitree-G1-23Dof-Effort-Flat` | `Unitree-G1-23Dof-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm` |
| `Unitree-G1-23Dof-Effort-Flat-MHA` | `Unitree-G1-23Dof-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm` |
| `Unitree-G1-23Dof-Effort-Rough` | `Unitree-G1-23Dof-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm` |
| `Unitree-G1-23Dof-Effort-Rough-MHA` | `Unitree-G1-23Dof-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm` |
| `Unitree-Go2-Effort-Flat` | `Unitree-Go2-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-MLP-CmdCurr-ObsNorm` |
| `Unitree-Go2-Effort-Flat-MHA` | `Unitree-Go2-Effort-0PD-Flat-IMU-JPos-JVel-Phase-H5-TimeMHA-CmdCurr-ObsNorm` |
| `Unitree-Go2-Effort-Rough` | `Unitree-Go2-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-MLP-CmdCurr-TerrainCurr-ObsNorm` |
| `Unitree-Go2-Effort-Rough-MHA` | `Unitree-Go2-Effort-0PD-Rough-IMU-JPos-JVel-Phase-HeightScan-H5-TimeMHA-CmdCurr-TerrainCurr-ObsNorm` |
| `Unitree-G1-Effort-Flat-Partial` | `Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-MLP-CmdCurr-ObsNorm` |
| `Unitree-G1-Effort-Flat-Partial-MHA` | `Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm` |
| `Unitree-G1-Effort-Flat-Partial-XAttn` | 已移除，不再注册 |
| `Unitree-G1-Effort-Flat-Partial-Phase1` | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-MLP-CmdCurr-ObsNorm` |
| `Unitree-G1-Effort-Flat-Partial-Phase1-MHA` | `Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-TimeMHA-CmdCurr-ObsNorm` |
| `Unitree-G1-Effort-Flat-Partial-Phase1-XAttn` | 已移除，不再注册 |
| `Unitree-G1-Effort-Flat-Partial-NoAnkle-XAttn` | 已移除，不再注册 |
| `Unitree-G1-Effort-Flat-Partial-Phase1-XAttn-NoCurriculum-FP32` | 已移除，不再注册 |
