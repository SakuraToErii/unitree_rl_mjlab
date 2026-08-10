# MJLab 1.5.3 电机命令延迟调研（TK3）

## 结论

MJLab 1.5.3 已原生支持 actuator command delay，不需要自定义
`ActionTerm`、修改 MJLab 源码或在 MJCF 中增加延迟元件。应直接在
`BuiltinPositionActuatorCfg` 上配置继承自 `ActuatorCfg` 的以下字段：

- `delay_min_lag`
- `delay_max_lag`
- `delay_hold_prob`
- `delay_update_period`
- `delay_per_env_phase`

本仓库的 TK3 训练 actuator 全部由
[`tk3_constants.py`](../../src/assets/robots/tiangong3/tk3_constants.py) 中的
`_position_actuator()` 创建，因此最小改动点就是该 helper，而不是
`scene_tiangong3.xml` 的 `<motor>` 块。

对当前 TK3 tracking 配置：

- physics timestep 是 `0.005 s`（5 ms）；
- `decimation=2`，policy timestep 是 `0.010 s`（10 ms）；
- actuator delay 的单位是 **physics timestep**，不是 policy timestep；
- 因此 lag 1/2/3 分别代表 5/10/15 ms。

本仓库现已按训练需求配置为每次 reference motion 重采样后从 `0..4`（闭区间）采样，
即当前 timestep 下每个环境独立采样 0/5/10/15/20 ms，并保持到下次 motion 重采样。
episode reset 和 reference motion 到末尾都会触发该路径。

## TK3 最小配置示例

当前 actuator 配置的核心实现如下：

```python
# Unit: physics timesteps. Current TK3 tasks use 5 ms/step.
TK3_COMMAND_DELAY_MIN_LAG = 0
TK3_COMMAND_DELAY_MAX_LAG = 4
TK3_USE_EXPLICIT_PD_GAINS = False


def _position_actuator(
  target_names_expr: tuple[str, ...],
  *,
  armature: float,
  effort_limit: float,
  stiffness_override: float | None = None,
  damping_override: float | None = None,
) -> BuiltinPositionActuatorCfg:
  calculated_stiffness = armature * NATURAL_FREQ**2
  calculated_damping = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
  resolved_stiffness = calculated_stiffness
  resolved_damping = calculated_damping
  if TK3_USE_EXPLICIT_PD_GAINS and stiffness_override is not None:
    resolved_stiffness = stiffness_override
  if TK3_USE_EXPLICIT_PD_GAINS and damping_override is not None:
    resolved_damping = damping_override
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=resolved_stiffness,
    damping=resolved_damping,
    effort_limit=effort_limit,
    armature=armature,
    frictionloss=FRICTIONLOSS,
    delay_min_lag=TK3_COMMAND_DELAY_MIN_LAG,
    delay_max_lag=TK3_COMMAND_DELAY_MAX_LAG,
    # MotionCommand samples after reset; keep it until the next motion resample.
    delay_hold_prob=1.0,
  )
```

TK3 tracking 在 `MotionCommandCfg` 中配置 lag range。每次
`MotionCommand._resample_command()` 先调用 `robot.reset()` 清除旧 command history，
再由 `randomize_actuator_command_lag()` 调用 actuator 的公开 `set_lags()` API。
reset 后重新写入不可省略：只有 `delay_hold_prob=1.0` 而没有手工采样时，buffer 会保持
默认 lag 0。

这会同时作用于 `TK3_ACTUATORS` 中的全部 29 个 position actuator。MJLab
1.5.3 会把 actuator type、transmission type 和全部 delay 参数相同的
builtin actuator 融合到一个 delayed group：每个并行环境有自己的 lag，
而同一环境内 29 个关节共享该 lag。这适合表达整机控制总线/控制器链路延迟，
也避免不真实的“每个关节独立随机抖动”。

如果以后不同关节挂在不同总线，给相应 actuator group 使用不同 delay 配置即可；
MJLab 会为不同配置建立不同的 buffer，并独立采样 lag。

## 参数语义

| 字段 | 1.5.3 语义 | 使用建议 |
| --- | --- | --- |
| `delay_min_lag` | 均匀采样区间的下界（包含），单位为 physics step | 必须 `>= 0` |
| `delay_max_lag` | 上界（包含）；`> 0` 才启用 delay | 固定延迟时令它等于 min |
| `delay_hold_prob` | 到达采样时刻时，继续保持当前 lag 的概率 | 增加随机延迟的时间相关性 |
| `delay_update_period` | 每多少个 physics step 才考虑重采样；`0` 表示每 step | 当前未显式设置，使用默认值 `0` |
| `delay_per_env_phase` | periodic 更新时，是否给不同 env 随机相位 | 只影响更新时刻，不控制 env 间是否共享 lag |

`ActuatorCfg` 没有暴露 `DelayBuffer.per_env`；actuator 路径使用其默认值
`True`，所以并行环境之间默认独立采样。

三种常见用法：

1. **固定延迟**：`min=max=N`。最容易解释和验证，也是本项目建议的起点。
2. **持续随机延迟**：`min<max`。默认 `update_period=0` 会在每个 physics step
   重采样；通常应结合 `update_period` 或 `hold_prob`，否则延迟抖动可能过快。
3. **每 episode 随机、episode 内固定**：配置允许范围，设置
   `delay_hold_prob=1.0`，并在 reset event 中对 actuator 调用 `set_lags()`。
   不能只设置 `hold_prob=1.0`：buffer 初始化/reset 会把 current lag 设为 0，
   若没有 reset event 写入采样值，它会一直保持 0。

本仓库当前选择的是第四种语义：**每个 reference motion 片段随机、片段内固定**。
它同样使用 `hold_prob=1.0`，但在每次 MotionCommand motion 重采样导致的 entity reset
之后立即调用 `set_lags()`。

还需注意：单独调用 `set_lags()` 后，如果自动重采样仍开启，下一次满足更新条件的
`compute()` 可能覆盖该值。要保持手工设置的 lag，应使用固定 `min=max`，或在 reset
event 写入后用 `hold_prob=1.0` 阻止自动改写。

## 实际执行链

当前训练链如下：

```text
policy raw action
  -> JointPositionAction.process_actions()  # scale + default offset
  -> JointPositionAction.apply_actions()    # encoder bias，写 joint_pos_target
  -> BuiltinActuatorGroup.apply_controls()
       -> DelayBuffer.append(target)
       -> DelayBuffer.compute()
       -> 写入 mjData.ctrl
  -> MuJoCo native <position> actuator
  -> physics step
```

lag 的重采样链为：

```text
MotionCommand._resample_command()
  -> 写入新 reference root/joint state
  -> robot.reset()                    # 清 history，同时把 current lag 清为 0
  -> randomize_actuator_command_lag() # 为新 reference 片段采样并 set_lags()
```

`ManagerBasedRlEnv.step()` 只在 policy step 开头处理一次新 action，但在每个
decimation substep 都依次调用 `apply_action()`、`scene.write_data_to_sim()` 和
`sim.step()`。所以 delay buffer 每 5 ms 推进一次；policy target 在两个 substep
之间不变，但仍会被各写入一次历史。

对当前 `BuiltinPositionActuatorCfg`，延迟的是 **position setpoint**。MuJoCo 的
native position actuator 收到延迟后的 setpoint，再使用当前的新鲜 joint state
计算 PD 力矩。它不会把已经计算完成的最终 PD torque 整体延后。这正是 MJLab
官方文档描述的典型实机模型：策略/通信下发的目标较旧，而电机侧控制环仍能直接读取
最新编码器状态。

因此要区分：

- 若“电机命令”指发给板端 position controller 的目标位置，内置 delay 正合适；
- 若要延迟 position controller 最终输出的 torque，则当前 builtin position 路径
  不是这个语义，需要另做显式控制器/输出力矩 buffer；
- 若使用 `BuiltinMotorActuatorCfg` 做纯 torque control，同一 delay 字段延迟的就是
  effort target。

## 与现有配置、观测的关系

- 延迟发生在 action scale、default offset 和 encoder bias 处理之后。
- `mdp.last_action` / ActionManager 的 `action` 仍是当前 policy 发出的 raw action，
  不是已经到达 actuator 的 delayed target。若策略必须知道实际到达电机的命令，
  需另加 observation；只增加 actuator delay 不会自动改变该 observation。
- episode reset 或 reference motion 重采样会清除相应环境的 delay history，并在 reset
  后为新片段采样 lag。第一次 append 会用首个新命令回填
  整个 history，因此不会把上个 episode 的旧命令带入新 episode，也不会人为注入零值。
  代价是 reset 后历史尚未积满时，lag 会被可用历史长度截断，第一帧看不到完整延迟。
- 随机 lag 的实现是“从历史中选择 `t-lag` 帧”，不是严格的带时间戳 FIFO 网络。
  当 lag 频繁变化时，输出可能重复、跳过，甚至选回更旧的命令。若实机通信保证顺序，
  应用 fixed lag，或通过 `update_period`/`hold_prob` 降低变化频率。
- 这些参数属于 MJLab Python actuator runtime，不会被序列化为 MJCF 的延迟属性；
  直接用普通 MuJoCo viewer 加载编译后的 XML 也不会自动运行 MJLab 的 delay buffer。

## 实机关节是否会表现为固定、共享 lag

不会精确地表现成单一纯延迟。实机从策略到运动响应通常至少包含：

```text
策略推理/进程调度
  -> ROS/IPC 与驱动层
  -> 总线周期与控制板锁存
  -> 电机侧位置/速度/电流环
  -> 机械惯量、摩擦和弹性
  -> 编码器/IMU 采样与状态回传
```

其中一部分接近固定周期，一部分会随调度、总线相位和负载产生 jitter；电机和机构本体
还表现为带宽、滤波、限流和惯性，而不是纯粹“等 N 步后原样执行”。实时总线和同步时钟
可以使同一轴组的执行时刻非常接近，但非实时主机上的策略推理和 ROS 2 链路仍可能变化。

一个更合适的抽象是：

```text
L_joint_i(t) = L_common(t) + L_group[group_i](t) + L_joint_i + jitter_i(t)
```

- `L_common`：策略计算、整帧打包、主机调度等所有关节共有的部分；
- `L_group`：某块控制板或某条 CAN/EtherCAT 总线共有的部分；
- `L_joint_i`：驱动器和关节的稳定偏差，通常比公共部分小；
- `jitter_i`：小幅时变残差，以及极少数 missed deadline / 丢包。

因此，“所有关节共享相同 lag”在**一次发送完整关节向量、同一同步轴组锁存**时是合理的
一阶模型；若腿、腰、手臂位于不同总线或控制板，更合理的是组内共享、组间不同。对同步
多轴系统，给 29 个关节各自独立抽取一个 5 ms 整数 lag 往往反而不真实，会制造过大的
关节间时序撕裂。真正的关节独立残差可能小于当前 5 ms physics timestep，MJLab 这个
离散 delay 无法分辨。

具身天工 3.0 的官方 SDK 表明上层使用 ROS 2 Jazzy，并用带时间戳的 `ArmCtrl` 消息携带
`MotorCtrl[]`；这只说明命令可以成批发送，不能证明驱动器在同一硬件时刻执行。当前
公开资料没有披露 TK3 内部电机总线拓扑、伺服周期、同步锁存机制、端到端延迟或 jitter，
所以不能把“全关节、reference 片段内固定”称为已经辨识出的 TK3 实机模型。官方 C++ 示例的
20 ms timer 也只是示例应用的发布周期，不是电机伺服频率或实机延迟。

已有 sim-to-real 实验同样说明应测量而不是猜测。例如 Google/UC Berkeley 的 Minitaur
工作用一次 PWM spike 测量命令到状态报告的往返延迟，测到板端 PD 与上位机 locomotion
controller 具有不同延迟，并依据实测值随机化控制步长和 latency。这些具体毫秒数属于
Minitaur，不能直接移植到 TK3；可复用的是测量方法和分层思路。

## 当前 `0..4` 配置的真实性边界

当前模型是：每个环境在 MotionCommand 重采样时均匀采样 0/5/10/15/20 ms，29 个关节
共享，并保持到下次 reference motion 重采样。它是一个合理的
**domain-randomization / robustness baseline**，但不是实机拟合模型：

- reference 片段固定值提供低频变化，但重采样时机与 motion 片段长度相关，并非实机
  latency 的时间模型；
- 全关节共享可表达公共控制链路，但漏掉多总线/多控制板的分组差异；
- 均匀分布和 0--20 ms 范围尚无 TK3 实测依据；
- lag 0 表示“不额外注入命令传输延迟”，并不表示电机和机构能瞬时响应；
- 当前只延迟 position setpoint，没有延迟策略看到的 joint state、IMU 等 observation；
- `DelayBuffer` 在 lag 改变时只是选择 `t-lag` 历史项，不是带序号和到达时间的网络
  FIFO。快速变 lag 可能重复、跳过或重新选回更旧的命令；
- `delay_hold_prob` 保持的是 **lag 值**，不是模拟丢包时“保持上一条命令”。后者需要
  单独的 hold-last/dropout 模型。

因此，如果当前目标是先训练一个对 0--20 ms 额外命令延迟不敏感的策略，可以保留现状；
如果目标是复现实机时序，则应先测量后改分布，不建议凭经验把区间改成另一个数字。

## 最真实的实施顺序

1. **分开测量各层。** 在安全的悬空单关节或测试台上给命令加 sequence number 和
   monotonic timestamp，记录 policy start/end、publish/write、板端 receive/apply（若
   可读）、目标回显、电流/力矩开始变化和 encoder/IMU 的采样时间。用小幅阶跃或 PRBS，
   不要只从机械位置开始移动的时刻反推通信延迟，因为该时刻还混入了死区和机械动态。
2. **统计分布与相关性。** 至少记录 min、P50、P95、P99、missed cycle 比例、连续丢帧
   长度和自相关；再看各关节延迟的互相关/聚类。高度相关的关节放进同一 delay group，
   不同控制板或总线分别建组。
3. **episode 基值加运行时残差。** 每个 episode 从实测 session/hardware 分布采样一个
   公共基值，再按实测总线组加偏差；只在实测确有 jitter 时做慢速、小范围更新。不要
   默认每个 physics step 独立均匀重采样。
4. **分别建模异常与反馈。** 用 hold-last/timeout 表达 missed deadline 或丢包；对 joint
   encoder、IMU、接触估计等 observation 按各自采样/传输链路另设 delay。命令和观测
   延迟不能用一个数代替。
5. **保留 actuator dynamics。** 继续校准 `kp/kd`、力矩/电流限制、速度或力矩变化率、
   friction、armature、死区和滤波。命令延迟只能补时序差异，不能代替电机模型。
6. **回放验证。** 用未参与拟合的实机阶跃/PRBS 日志，对比 delayed target、current/
   torque 和 joint response；最后在固定 lag、经验分布、实测分布三组策略上做消融。

MJLab 1.5.3 的现成 delay 适合实现“每个同步组一个整数步总延迟”。若需要公共延迟与
组延迟相关采样，可以由 reset/event 同时采样 common 和 group residual，再给各组写入
总 lag。若要复现带时间戳的实测 trace、严格 FIFO、低概率丢包/保持或小于 5 ms 的
jitter，则应实现自定义 command queue；必要时降低 physics timestep，但这会增加训练
成本并影响其余动力学参数。

## 为什么不改 standalone `<motor>`

交接文档已经确认，当前训练加载 robot-only `tiangong3.xml`，然后由
`tk3_constants.py` 注入 29 个 `BuiltinPositionActuatorCfg`。standalone
`scene_tiangong3.xml` 的 `<motor>` 只服务于直接通过 `mjData.ctrl` 做 torque
control 的 standalone scene。

因此：

- 修改 standalone `<motor>` 不会给当前 MJLab tracking 训练增加延迟；
- MuJoCo XML 本身也没有等价于 MJLab `DelayBuffer` 的“整数步命令延迟”配置；
- 对训练应修改 `tk3_constants.py` 的 actuator cfg；
- 对 standalone torque controller，应在写 `mjData.ctrl` 的应用代码中自行维护 FIFO。

## 本地验证

在本仓库锁定的 `mjlab==1.5.3` 环境中做了三项运行时验证：

1. 最小单关节 `BuiltinPositionActuatorCfg`，固定 `lag=2`，依次下发
   `[0.1, 0.2, 0.3, 0.4]`，实际 `ctrl` 为
   `[0.1, 0.1, 0.1, 0.2]`；reset 后首个 `0.9` 命令立即回填历史，实际
   `ctrl=0.9`。这与官方 fixed-delay 和 reset 测试一致。
2. 用 `dataclasses.replace()` 在内存中为真实 TK3 的全部 actuator 注入
   `min=1, max=2, update_period=2` 后编译/初始化：模型为 29 个 actuator，
   MJLab 建立了 1 个 delayed builtin group，包含全部 29 个 target，buffer 的
   batch size 与并行环境数一致。
3. 对当前 `robot.reset() → randomize_actuator_command_lag()` 核心路径做了隔离验证：
   fake actuator 在 reset 中把 lag 清零，随后采样能正确写入；第二次 motion resample
   会按新范围采新 lag。包含该路径的 compatibility 回归 13 项均通过；尚未运行完整
   GPU `ManagerBasedRlEnv` 观测 fused delay buffer。

当前实现修改了 `tk3_constants.py`，并在 tracking `MotionCommand` 的重采样路径中接入
delay sampler；没有修改任何 robot XML，也没有注册 command-delay EventManager term。

## 一手来源（固定到 v1.5.3）

- [MJLab v1.5.3 actuator 文档：直接在 actuator cfg 上增加 delay 字段](https://github.com/mujocolab/mjlab/blob/v1.5.3/docs/source/actuators.rst#L39-L52)
- [官方 command-delay 语义：target 延迟、feedback state 保持新鲜、单位为 physics step](https://github.com/mujocolab/mjlab/blob/v1.5.3/docs/source/actuators.rst#L245-L276)
- [`ActuatorCfg` 的五个 delay 字段、默认值与校验](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/actuator/actuator.py#L67-L112)
- [`Actuator.apply_delay()` 与 `set_lags()`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/actuator/actuator.py#L310-L362)
- [builtin actuator 按 delay 配置融合，并在写 `ctrl` 前运行 buffer](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/actuator/builtin_group.py#L98-L192)
- [`DelayBuffer` 的 lag、更新、hold、reset 及 history 选择语义](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/utils/buffers/delay_buffer.py#L12-L118)
- [环境在每个 decimation substep 应用 action、写控制并推进 physics](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/manager_based_rl_env.py#L418-L427)
- [官方 delayed actuator 回归测试](https://github.com/mujocolab/mjlab/blob/v1.5.3/tests/test_delayed_actuator.py)
- [v1.5.3 changelog；1.5.1 已修复/融合 custom actuator command delay 路径](https://github.com/mujocolab/mjlab/blob/v1.5.3/docs/source/changelog.rst#L17-L132)
- [具身天工 3.0 官方 SDK：ROS 2 Jazzy 与关节控制示例](https://github.com/Open-X-Humanoid/xhumanoid_sdk)
- [具身天工 3.0 官方 `ArmCtrl` / `MotorCtrl[]` 接口说明](https://github.com/Open-X-Humanoid/xhumanoid_sdk/blob/main/single_joint_control/README.md#armctrl-%E6%8E%A7%E5%88%B6%E6%B6%88%E6%81%AF)
- [具身天工 3.0 官方 C++ 示例：20 ms 应用 timer 与带时间戳消息](https://github.com/Open-X-Humanoid/xhumanoid_sdk/blob/main/single_joint_control/cpp/single_joint_control/src/cmd_publisher.cpp)
- [Hwangbo et al.：实测并随机化控制延迟的 sim-to-real 方法](https://arxiv.org/abs/1804.10332)
- [EtherCAT Technology Group：Distributed Clocks 与同步输出机制](https://www.ethercat.org/en/technology.html)
- [ros2_control Controller Manager：read-update-write 实时控制链](https://control.ros.org/rolling/doc/ros2_control/controller_manager/doc/userdoc.html)
