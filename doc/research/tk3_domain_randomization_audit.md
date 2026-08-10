# TK3 tracking 域随机化审计

审计日期：2026-08-10  
审计对象：仓库默认任务 `TK3-Tracking`  
审计基线：当前工作树（包括用户尚未提交的 TK3 配置变更）  
结论口径：本文区分 **[事实]**、**[实现复核]**、**[回归测试]**、**[运行时复核]** 与 **[判断]**；“范围”均指默认配置，CLI 手工覆盖不在内。

## 1. 结论摘要

**总体判断：此前确认的 command delay 与 pelvis COM 覆盖缺陷已在当前工作树修复；现有域随机化实现链已自洽，但各物理范围是否匹配实机仍需测量与完整环境验证。**

1. **[事实] 训练默认启用域随机化；官方 play 路径全部关闭。**  
   任务注册把普通配置交给训练，把 `play=True` 配置交给播放；训练脚本调用普通配置，播放脚本明确调用 `load_env_cfg(..., play=True)`。play 不但执行 `cfg.events.clear()`，还关闭 actor observation corruption，并将 motion 初始化改为首帧、零位姿/速度/关节扰动。仓库没有独立的 evaluation 配置：使用普通配置做评估会保留训练随机化，使用 `scripts/play.py` 则是 nominal 评估。证据：
   `src/tasks/tracking/config/tk3/__init__.py:11-18`、
   `src/tasks/tracking/config/tk3/env_cfgs.py:215-239`、
   `scripts/train.py:38-43`、
   `scripts/play.py:49-53`。

2. **[事实 + 实现复核 + 回归测试] actuator command delay 现已按每个 reference motion 片段、每 env 生效。**  
   `MotionCommand` 配置闭区间 `{0,1,2,3,4}` physics steps（0/5/10/15/20 ms）。每次 `_resample_command()` 写入 motion state 后先执行 `robot.reset()` 清除旧 command history，再立即为相关环境独立采样并写入新 lag。该路径同时覆盖 episode reset 和 reference motion 到末尾后的重采样，所以后者会为新 reference 片段采新 lag，而不是保持旧 episode lag。证据：
   `src/tasks/tracking/config/tk3/env_cfgs.py`、
   `src/tasks/tracking/mdp/events.py`、
   `src/tasks/tracking/mdp/commands.py`、
   `src/assets/robots/tiangong3/tk3_constants.py:36-40,64-72`；
   MJLab reset 顺序见
   [`manager_based_rl_env.py#L552-L585`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/manager_based_rl_env.py#L552-L585)，
   buffer reset 语义见
   [`delay_buffer.py#L193-L219`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/utils/buffers/delay_buffer.py#L193-L219)。
   回归测试用 fake entity 精确模拟 `Entity.reset()` 把 lag 清零，并验证 reset 后采样以及
   第二次 reference resample 换用新 lag；这是核心路径单测，不是完整 GPU
   `ManagerBasedRlEnv` 集成测试。来源：`tests/test_mjlab_compat.py`。

3. **[事实 + 实现复核 + 回归测试] `base_com` 与 mass/inertia 随机化现已组合生效。**  
   TK3 配置先 `pop("base_com")`，插入作用于 `(".*",)` 的 `pseudo_inertia`，再把 `base_com` 紧随其后 reinsert；因此全部 39 个 robot bodies（**仍包括 pelvis**）先获得各 body 独立 `U(0.7,1.3)` mass/inertia scale，随后 pelvis `body_ipos` 三轴各加 `U(-0.05,0.05) m`。MJLab EventManager 按配置 insertion order 执行 startup terms，并在全部 term 完成后按 strongest `RecomputeLevel` 仅 recompute 一次；两项均要求 `set_const`，所以最终 recompute 覆盖组合后的质量、惯量和 COM。证据：
   `src/tasks/tracking/tracking_env_cfg.py:173-184`、
   `src/tasks/tracking/config/tk3/env_cfgs.py:126-143`；
   上游默认值读取与写回见
   [`body.py#L416-L540`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/dr/body.py#L416-L540)，
   EventManager 顺序与 strongest recompute 见
   [`event_manager.py#L260-L333`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/managers/event_manager.py#L260-L333)。
   回归测试断言 TK3 startup 配置顺序为 pseudo-inertia 在前、base COM 在后，并确认
   `base_com` 只选择 pelvis：`tests/test_mjlab_compat.py`。

4. **[事实 + 判断] `joint_friction` 现为明确的绝对范围，但物理合理性仍待实测。**  
   用户已将配置改为 `operation="abs"`、`ranges=(0.01,0.6)`，实际就是每 DoF 独立
   `U(0.01,0.6) N·m`，不再乘 nominal `frictionloss=0.1 N·m`。API 语义与字面物理单位一致；
   但 0.01–0.6 N·m 是否覆盖 TK3 各关节真实静/库仑摩擦，仓库内仍无辨识数据。

5. **[事实 + 判断] 其余核心 API 用法基本正确，但控制链仍需校准。**  
   `abs` 直接设 geom friction，armature/PD 的 `scale` 按编译默认值缩放，伪惯量同时缩放 mass 与 inertia，PD 随机化同步写正确的 gain/bias slots，encoder bias 也形成闭环一致的标定误差。当前 `TK3_USE_EXPLICIT_PD_GAINS=False`，nominal Kp/Kd 是 armature 公式值；仍须确认实机部署最终使用同一组增益。

## 2. 依赖与真实后端

### 2.1 已锁定、已安装版本

| 组件 | 结论 | 一手证据 |
|---|---|---|
| Python | `3.11.*` | `pyproject.toml:10-14`，`uv.lock:1-3` |
| MJLab | `1.5.3` | `pyproject.toml:10-14`，`uv.lock:721-740` |
| MuJoCo | `3.10.0` | `uv.lock:809-827` |
| MuJoCo Warp | `3.10.0.3` | `pyproject.toml:12-14`，`uv.lock:829-843` |
| mujoco_playground | **未声明、未安装、仓库无调用** | `pyproject.toml:10-14`；本地 package metadata 复核 |
| mujoco-mjx / JAX | **未安装，非本任务后端** | 本地 package metadata 复核 |

**[事实]** 本任务是 `mjlab → mujoco-warp → MuJoCo model`，不是
`mujoco_playground`/MJX。MJLab 1.5.3 的 `Simulation` 也明确说明其 GPU 后端是
MuJoCo Warp：
[`sim.py#L200-L218`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/sim/sim.py#L200-L218)。

### 2.2 nominal 编译模型

**[运行时复核]** 使用当前 `get_tk3_robot_cfg()` 编译：

- 39 个 robot bodies、29 个 policy joints、29 个 native position actuators；
- nominal robot 总质量约 `65.8515 kg`；
- passive joint damping 全部为 `1.0 N·m·s/rad`；
- armature 范围 `0.0236–0.37 kg·m²`，frictionloss 全部为 `0.1 N·m`；
- 当前公式 Kp 范围约 `23.29–365.18`，Kd 范围约 `2.97–46.50`；
- 8 个 foot geoms nominal sliding friction 为 1.0、`condim=3`、`priority=2`。

来源：
`src/assets/robots/tiangong3/tk3_constants.py:28-72,74-165,190-205,210-227`，
robot MJCF 惯性/关节默认值见
`src/assets/robots/tiangong3/xmls/tiangong3.xml:11-12,58-156,183-284`。

## 3. 完整调用链与模式覆盖

### 3.1 配置组装

```text
src/tasks/tracking/config/tk3/__init__.py
  ├─ train: tk3_flat_tracking_env_cfg(play=False)
  └─ play:  tk3_flat_tracking_env_cfg(play=True)
         ↓
src/tasks/tracking/config/tk3/env_cfgs.py
  └─ make_tracking_env_cfg()
         ↓
src/tasks/tracking/tracking_env_cfg.py
  ├─ base events / observations / MotionCommand initial-state ranges
  └─ ManagerBasedRlEnvCfg
```

**[事实]** `load_env_cfg()` 返回注册配置的深拷贝；`play=True` 才选
`play_env_cfg`：
[`registry.py#L48-L55`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/tasks/registry.py#L48-L55)。

### 3.2 环境执行时序

```text
ManagerBasedRlEnv.__init__
  → EventManager 建立并解析 SceneEntityCfg
  → expand_model_fields(...)
  → 创建 command/action/observation managers
  → apply("startup")                  # 每个 env 构造时一次
      → pseudo_inertia                # 全部 39 bodies，含 pelvis
      → base_com                      # 最后写 pelvis body_ipos offset
      → strongest set_const recompute # 全部 startup terms 后统一一次

env.reset / episode auto-reset
  → sim.reset
  → scene.reset                       # actuator delay buffer 先清零
  → apply("reset")                    # 其它 reset events
  → command_manager.reset
      → MotionCommand._resample_command
          → 写 motion root/joint state
          → robot.reset               # 清旧 command history，并把 lag 清零
          → randomize command lag     # 为新 reference 片段立即采样 0–4

reference motion 到末尾
  → MotionCommand._resample_command
      → 同一 robot.reset → randomize 路径
      → 为新 reference 片段重新采样 lag

每个 policy step（10 ms）
  → 2 × physics substep（各 5 ms）
  → reward / termination / auto-reset
  → command update
  → interval event                    # push 在这里执行
  → actor/critic observation
```

MJLab 一手来源：

- startup 与 manager 初始化：
  [`manager_based_rl_env.py#L299-L354`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/manager_based_rl_env.py#L299-L354)
- 每个 decimation substep 写 control：
  [`manager_based_rl_env.py#L418-L427`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/manager_based_rl_env.py#L418-L427)
- reset/command/event 顺序：
  [`manager_based_rl_env.py#L552-L585`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/manager_based_rl_env.py#L552-L585)
- event mode 定义：
  [`event_manager.py#L100-L141`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/managers/event_manager.py#L100-L141)
- per-env interval timer：
  [`event_manager.py#L264-L288`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/managers/event_manager.py#L264-L288)

### 3.3 训练 / 评估 / 播放矩阵

| 模式 | physical events | actor white noise | motion 初始扰动 | 说明 |
|---|---:|---:|---:|---|
| `scripts/train.py` 默认 | 开 | 开 | 开 | 普通注册配置 |
| training video | 开 | 开 | 开 | 只是给同一训练 env 套 recorder |
| `scripts/play.py` | **全关** | **全关** | **全关** | nominal 参数，motion 从首帧开始 |
| 自建 evaluation 且 `load_env_cfg(..., play=False)` | 开 | 开 | 开 | 仓库无独立 eval preset |
| 自建 evaluation 且 `play=True` | 全关 | 全关 | 全关 | 与官方 play 一致 |

play 虽然保留 actuator cfg 的 `delay_max_lag=4`，但把
`MotionCommandCfg.actuator_command_lag_range` 设为 `None`；每次 entity reset 后跳过采样，
buffer lag 保持 0，因此实际是 nominal zero added command lag。

## 4. 已有随机化逐项清单

### 4.1 EventManager 项

| 项 | mode / 频率 | 对象与相关性 | 分布、操作与实际结果 | 状态 |
|---|---|---|---|---|
| `base_com` | startup；环境创建一次，episode 间不变；在 pseudo-inertia 后执行 | robot `pelvis`；xyz 各自采样 | 最终 pelvis `body_ipos = default + U(-0.05,0.05)m` | 生效；与 mass/inertia 组合 |
| `encoder_bias` | startup | 29 joints；每 env、每 joint 独立 | `U(-0.01,0.01) rad` | 生效 |
| `foot_friction` | startup | 8 个 `foot_*`；同一 env 内 8 个 geom 共用一个样本，env 间独立 | sliding µ `abs U(0.3,1.8)`；torsional/rolling 保留默认 | 生效 |
| `ground_friction` | startup | 单个 terrain plane；env 间独立 | sliding µ `abs U(0.1,1.0)` | 生效；不控制脚–地，控制普通身体–地 |
| `randomize_rigid_body_mass_others` | startup | **全部 39 bodies，包括 pelvis**；每 body 独立 | mass scale `U(0.7,1.3)`；mass 与 central inertia 同比例，COM 不变 | 生效；名称 “others” 不准确 |
| `joint_armature` | startup | 29 joint DoFs；独立 | compile default × `U(0.7,1.3)` | 生效 |
| `pd_gains` | startup | 29 native position actuators；每 target 的 Kp/Kd 独立 | Kp × `U(0.8,1.2)`；Kd ×独立 `U(0.8,1.2)` | 生效 |
| `joint_friction` | startup | 29 DoFs；独立 | `abs U(0.01,0.6) N·m` | 生效；范围合理性待实测 |
| `push_robot` | interval；每 env 独立等待 `U(1,3)s`，以 10 ms policy step 量化 | floating base root velocity 六维 | 向当前世界系 root velocity **加** `x/y U(-0.5,0.5)m/s`、`z U(-0.2,0.2)m/s`、roll/pitch `U(-0.52,0.52)rad/s`、yaw `U(-0.78,0.78)rad/s` | 生效 |

仓库来源：

- base events 与范围：
  `src/tasks/tracking/tracking_env_cfg.py:30-38,163-202`
- TK3 覆盖、新增 events：
  `src/tasks/tracking/config/tk3/env_cfgs.py`
- delay 参数：
  `src/assets/robots/tiangong3/tk3_constants.py:36-40,64-72`
- delay sampler 与 MotionCommand 集成：
  `src/tasks/tracking/mdp/events.py`、`src/tasks/tracking/mdp/commands.py`
- delay/COM 回归：
  `tests/test_mjlab_compat.py`

### 4.2 不在 `cfg.events` 中、但同样影响训练分布的随机化

#### Motion 初始状态 / 重采样

触发时机：每个 episode reset；以及 reference motion 到末尾时。来源：
`src/tasks/tracking/tracking_env_cfg.py:140-158`、
`src/tasks/tracking/mdp/commands.py:297-355,348-457,481-490`。

| 项 | 默认训练分布 | play |
|---|---|---|
| motion 时间起点 | adaptive failure-weighted sampling；含 uniform floor | 固定第 0 帧 |
| root position | reference + x/y `U(±0.05)m`、z `U(±0.01)m` | 0 offset |
| root orientation | reference 左乘 roll/pitch `U(±0.1)rad`、yaw `U(±0.2)rad` | 0 offset |
| root linear velocity | reference + x/y `U(±0.5)m/s`、z `U(±0.2)m/s` | 0 offset |
| root angular velocity | reference + roll/pitch `U(±0.52)rad/s`、yaw `U(±0.78)rad/s` | 0 offset |
| joint position | reference + 每 joint `U(-0.1,0.1)rad`，再裁到 soft limits | 0 offset |
| joint velocity | reference 原值，不加噪声 | reference 原值 |
| actuator command delay | 每次 motion 重采样后，每 env 采 `U{0,1,2,3,4}` physics steps（0–20 ms），保持到下次重采样 | 0 ms |

**[事实]** `HOME_KEYFRAME` 是 action offset、relative joint observation 和 nominal
asset initial state；tracking episode 的实际 root/joint state会由 MotionCommand 直接写成
reference motion + 上述扰动。来源：
`src/assets/robots/tiangong3/tk3_constants.py:168-187`、
`src/tasks/tracking/mdp/commands.py:357-457`。

#### Actor observation corruption

**[事实]** 每次 actor observation 计算（默认 100 Hz）独立添加 element-wise uniform
white noise；critic 无 corruption。TK3 deployable actor 还会删掉
`motion_anchor_pos_b` 和 `base_lin_vel`，所以这两项在 TK3 默认任务中既不存在，也不会加噪。

| actor term | white noise |
|---|---|
| `motion_anchor_ori_b`（rotation matrix 前两列，共 6 维） | `U(-0.05,0.05)` 逐元素相加 |
| `base_ang_vel` | `U(-0.2,0.2) rad/s` |
| `joint_pos` | `U(-0.01,0.01) rad`，另叠加 startup 固定 encoder bias `U(-0.01,0.01) rad` |
| `joint_vel` | `U(-0.5,0.5) rad/s` |
| command、last action | 无 |

来源：
`src/tasks/tracking/tracking_env_cfg.py:47-80,82-120`、
`src/tasks/tracking/config/tk3/env_cfgs.py:215-225`；
MJLab 的 corruption 开关与 compute→noise 顺序见
[`observation_manager.py#L17-L24`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/managers/observation_manager.py#L17-L24)、
[`observation_manager.py#L430-L439`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/managers/observation_manager.py#L430-L439)，
uniform noise 是 `[n_min,n_max)` 的逐元素独立加法：
[`noise_cfg.py#L61-L86`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/utils/noise/noise_cfg.py#L61-L86)。

## 5. API 真实语义核验

### 5.1 `abs`、`scale`、`add`

**[事实]**

- `abs`：忽略 base，直接写 sampled value；其 base 是当前值，但 combine 只返回 random。
- `scale`：以 **compile-time default** 为 base，再乘 sampled factor；重复调用不会累乘。
- `add`：以 **compile-time default** 为 base，再加 sampled offset；重复调用不会累加。

官方实现：
[`_types.py#L82-L100`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/dr/_types.py#L82-L100)、
[`_core.py#L109-L148`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/dr/_core.py#L109-L148)。

**[判断]** foot/ground/joint friction 用 `abs`、armature/PD 用 `scale` 均与当前配置意图
一致。`joint_friction` 因而直接得到 `U(0.01,0.6) N·m`，不再有 scale/绝对单位歧义。
COM 与 pseudo-inertia 都以 defaults 为基准写字段，因此顺序仍是语义的一部分；当前配置
通过 pop/reinsert 明确保证 pseudo-inertia 先写、base COM 最后写，避免覆盖。

### 5.2 mass / inertia

**[事实]** `pseudo_inertia(alpha_range=...)` 中 mass 和 inertia 的 scale 都是
`exp(2*alpha)`，COM 不变。仓库自定义 sampler 先把 alpha 上下界映射成 scale 上下界，
对 scale 做 uniform sampling，再转回 alpha，因此实现的是 **每 body 的真实
`U(0.7,1.3)` mass/inertia scale**，不是 log-uniform，也不是仅改 mass。

来源：
`src/tasks/tracking/config/tk3/env_cfgs.py:27-48,126-143`；
官方 pseudo-inertia 语义：
[`body.py#L416-L481`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/dr/body.py#L416-L481)。

**[推断]** 因 39 个 body 独立缩放，总质量并不是 `U(0.7,1.3)`。基于 nominal body
mass 的加权方差，total-mass scale 标准差约 5.0%，正态近似 95% 约为 nominal ±9.9%；
真正变化很宽的是各 link 的质量比。对于包含附件/装配误差的 humanoid，这是可用的鲁棒性
分布，但不应把它描述成“整机质量 ±30%”。

### 5.3 foot–ground friction pair

**[事实]** MuJoCo dynamic contact 的规则是：priority 不同则使用高 priority geom 的
`condim` 和 friction；相同才取 `condim=max(...)` 和 friction element-wise max。
显式 `<pair>` 则完全使用 pair 参数、忽略单个 geom。官方说明：
[MuJoCo contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)。

本仓库：

- foot：`priority=2`、`condim=3`、nominal µ=1.0；
- terrain：`priority=1`、`condim=3`、nominal µ=1.0；
- 普通 robot collision geom：`priority=0`、`condim=1`。

因此：

- 脚–地有效 sliding µ = foot sample `U(0.3,1.8)`，ground sample 不与它相乘、相加或取 max；
- 普通身体–地由 terrain priority 1 胜出，有效 µ = `U(0.1,1.0)`、`condim=3`；
- foot–普通身体由 foot 胜出。

来源：
`src/assets/robots/tiangong3/tk3_constants.py:190-205`、
`src/tasks/tracking/config/tk3/env_cfgs.py:101-122`。

**[判断]** 这不是重复摩擦随机化，而是有意把脚接触与摔倒/手撑地接触分开，语义明确且
正确。脚范围很宽，属于鲁棒性覆盖而非真实地面概率模型；是否合理取决于实测脚垫/目标地面。

### 5.4 actuator gain/bias 与 effort limit

**[事实]** 当前使用 `BuiltinPositionActuatorCfg`，MJLab 生成 native MuJoCo position
actuator：

```text
force = kp * ctrl - kp * q - kd * qdot
gainprm[0] = kp
biasprm[1] = -kp
biasprm[2] = -kd
```

`dr.pd_gains(operation="scale")` 对 Kp 同时缩放 `gainprm[0]` 和 `biasprm[1]`，对 Kd
缩放 `biasprm[2]`，因此没有破坏 position servo 公式。官方实现：
[`spec.py#L279-L350`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/utils/spec.py#L279-L350)、
[`dr/actuator.py#L29-L112`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/dr/actuator.py#L29-L112)。

**[事实]** force saturation 来自每 actuator 的 `effort_limit`，当前不随机化：
`src/assets/robots/tiangong3/tk3_constants.py:83-151`。armature 是 MuJoCo joint
reflected rotor inertia，joint friction event 改的是 load-independent
`dof_frictionloss`，并非 viscous damping。XML 的 passive joint damping=1 被保留且没有 DR：
`src/assets/robots/tiangong3/xmls/tiangong3.xml:11-12`。

**[事实]** 当前 joint friction 使用 `operation="abs"` 和 `ranges=(0.01,0.6)`，所以忽略
编译默认 `frictionloss=0.1 N·m`，直接为每 DoF 写入 `U(0.01,0.6) N·m`。
**[判断]** 该范围跨度仍大，且没有按关节负载/减速器分组；是否物理合理必须用实机
breakaway、低速跟踪或 decay 数据确认。

**[判断]** armature/PD API 使用正确；但独立采样 armature、Kp、Kd 会改变 closed-loop
natural frequency 与 damping ratio。这可表达机械与控制参数独立不确定性，却不是“始终
保持 5 Hz、ζ=2”的随机化。若目标是辨识后的物理分布，应按实机参数相关性联合采样。

### 5.5 encoder bias

**[事实]**

- startup 生成每 joint 固定 bias；
- actor `joint_pos_rel(biased=True)` 读取 `q + bias`；
- critic 读取真实 `q`；
- action path 把 position target 减去同一个 bias，使 native servo 的误差等效于对
  biased encoder 闭环。

仓库来源：
`src/tasks/tracking/tracking_env_cfg.py:71-78,103-106,185-192`；
MJLab 来源：
[`observations.py#L51-L62`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/observations.py#L51-L62)、
[`actions.py#L211-L225`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/actions/actions.py#L211-L225)、
[`data.py#L366-L372`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/entity/data.py#L366-L372)。

**[判断]** 这是闭环一致的标定残差模型，明确正确。范围 ±0.01 rad（约 ±0.57°）仍应由
实际零位重复标定数据确认。

### 5.6 command delay

**[事实]** delay 单位是 physics step（本任务 5 ms），延迟的是 position setpoint；
native PD 仍读取最新 q/qdot。相同 actuator type/transmission/delay cfg 会融合，共享一个
lag buffer；环境之间仍独立。官方文档：
[`actuators.rst#L245-L276`](https://github.com/mujocolab/mjlab/blob/v1.5.3/docs/source/actuators.rst#L245-L276)，
融合实现：
[`builtin_group.py#L98-L192`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/actuator/builtin_group.py#L98-L192)。

**[事实 + 实现复核]** `hold_prob=1.0` 的含义是每个自动更新时刻都保留当前 lag，不是
“每 episode 自动采一次”。当前实现把采样直接放在 `MotionCommand` 的 motion
重采样生命周期中：

1. `MotionCommandCfg.actuator_command_lag_range` 为训练配置 0–4、play 配置 `None`；
2. 每次 `_resample_command()` 写完 root/joint state 后先 `robot.reset()`，清空上一段
   reference 的 command history；
3. reset 返回后立即调用 `randomize_actuator_command_lag()`，为相关 env 独立采样并通过
   公开 `Actuator.set_lags()` 写 buffer；
4. episode reset 和 reference motion 到末尾都走同一路径，因而每个新 reference
   片段各采一次 lag；
5. play 跳过采样，reset 后保持 nominal zero lag。

仓库来源：
`src/tasks/tracking/config/tk3/env_cfgs.py`、
`src/tasks/tracking/mdp/events.py`、
`src/tasks/tracking/mdp/commands.py`。

**[回归测试边界]** `tests/test_mjlab_compat.py` 用 fake entity/actuator 精确复现
“reset 将 lag 清零”，验证同一 MotionCommand 路径随后写入新 lag，并在第二次 reference
resample 时重新采样。测试不读取 MJLab 私有 delay buffer，也不是完整 GPU env 测试，
尚未直接观察真实 fused buffer 的最终值。

### 5.7 push 与姿态初始化

**[事实]** `push_by_setting_velocity` 是直接向 root velocity 加增量的瞬时 kick，质量无关，
不是力/冲量积分，也不会指定受力点。官方函数说明：
[`events.py#L316-L338`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/events.py#L316-L338)。

**[判断]** 对 humanoid motion tracking，用低成本速度 kick 训练恢复能力是合理 baseline；
但它不能验证抗具体牛顿级外力、受力高度产生的倾覆矩或不同质量下响应。若目标是实机推扰
可解释性，应根据实测外力另加短时 body impulse，而不是把当前范围称为“±某 N 推力”。

**[事实]** MotionCommand 初始化以 reference frame 为中心，对 root position/orientation、
root velocity 与 joint position加扰动；joint velocity保持 reference。play 将其清零。
参数均被实际消费，没有参数名误用。

## 6. 分类判断

### 6.1 明确正确

1. **版本与 API 对齐。** 配置针对实际安装的 MJLab 1.5.3 / MuJoCo 3.10 /
   MuJoCo Warp 3.10.0.3，不依赖 mujoco_playground/MJX。
2. **foot/ground 摩擦拆分。** priority 设计让脚和普通身体接触使用不同随机变量，避免
   不明确的双边组合。
3. **mass/inertia 同比例缩放。** 使用 pseudo-inertia 而非只改 `body_mass`，满足刚体
   惯性物理一致性。
4. **PD gain slots。** Kp/Kd 对 native position actuator 的 gain/bias 修改完整一致。
5. **encoder bias 闭环。** actor 与 actuator 同时消费 bias，critic 保持 clean state。
6. **play nominal 化。** physical events、observation noise、motion 初始扰动都明确关闭；
   导出 metadata 从 host nominal model 读取，不会把某个训练 env 的 startup 样本写入部署。
   来源：`src/tasks/tracking/rl/runner.py:141-170`。
7. **reference-segment command delay 生命周期。** MotionCommand 在每次 Entity reset
   清除旧 command history 后立即采新 lag；episode reset 与 reference wrap 行为一致。
8. **mass/inertia 与 pelvis COM 组合。** pseudo-inertia 先执行且仍包含 pelvis，base COM
   后执行，EventManager 最后按 strongest level 统一 recompute。

### 6.2 有条件合理

1. **startup 而非 per-reset 的物理参数。** 对大规模并行训练，每 env 固定一组“硬件”
   能减少非平稳性；但若 env 数很少或做长续训，分布覆盖会不足。不是错误，应按训练规模选择。
2. **foot µ 0.3–1.8、ground µ 0.1–1.0。** 覆盖很宽，适合 robustness；不能解释为真实
   TK3 脚垫在目标地面的均匀概率。建议用实测 nominal/P5/P95 收窄主体分布，仅留少量 tail。
3. **每 link 独立 mass/inertia ±30%。** 能制造 link-ratio 不确定性，但比常见制造误差宽；
   total mass 实际窄得多。若已有 CAD/称重数据，应按 body group 与附件相关性采样。
4. **armature ±30%、Kp/Kd ±20% 相互独立。** 鲁棒性上合理，辨识模型中应保留参数相关性。
5. **1–3 s velocity kick。** 可训练恢复，不是可解释的 physical push。
6. **初始 root velocity/姿态扰动。** 对 dance tracking 较激进但可接受；应通过失败率和
   curriculum 分布确认没有让策略主要学习 reset recovery。
7. **white observation noise。** gyro/joint 噪声范围可作 baseline；对 rotation matrix
   六个元素直接独立加噪会产生非正交输入，而实机部署由合法 quaternion/rotation 得到，
   更物理的做法是随机小旋转后再转 6D。
8. **joint friction `abs U(0.01,0.6) N·m`。** API 与单位已经明确；但范围很宽且未按
   关节分组，只有实测后才能判断是否适合训练分布。

### 6.3 高风险或待确认

1. **高风险待确认：训练/部署 PD 可能不是同一组。** 当前
   `TK3_USE_EXPLICIT_PD_GAINS=False`，显式 `900/57`、`1260/80` 等 override 不生效；
   当前 nominal 是公式值（例如 hip pitch/roll Kp≈236.87、knee≈365.18）。
   仓库近期导出的 `BeyondMimic_dance.yaml` 也是公式值，但工作区外的部署 checkout
   仍可见显式高增益版本。只有确认实机实际加载的 YAML、板端是否再缩放 Kp/Kd，才能判断
   当前 ±20% gain DR 是否围绕正确 nominal。
2. **高风险待确认：并联踝没有进入训练物理模型。** 当前 MJCF 是两个独立 serial
   ankle joints；工作区外的实机部署路径会做 serial↔parallel 状态/命令转换，并使用
   单独的 physical ankle gains。该转换的非线性、耦合、雅可比奇异区与误差未被当前
   armature/Kp/Kd 独立缩放覆盖。是否影响当前动作取决于真实踝机构和目标动作幅度。
3. **验证边界。** 新回归覆盖 delay 的 command/entity reset 后采样顺序和 COM startup
   insertion order，但使用 fake entity/配置断言；尚未在完整 GPU env 读取 fused delay
   buffer 或 startup 后真实 `body_ipos` 分布。

### 6.4 缺失但值得考虑

以下是优先候选，不是“缺少即错误”：

1. **最高优先：实测 command/observation latency。** 修复 command delay 后，再分别建模
   joint state、IMU 的采样/ROS/队列延迟；当前 observation terms 的 delay 都是 0。
2. **actuator dynamics。** 依据日志决定是否加入 torque-speed/电压/电流限制、torque
   slew、死区/回差、温度降额或 learned actuator；不要用 Kp/Kd DR 代替这些效应。
3. **effort limit DR。** 当前 force limits 固定。真实电压、温度和驱动限流若变化明显，
   可做小范围/分组随机化；若硬件限流稳定则保持 fixed 完全合理。
4. **passive viscous damping 标定。** 当前所有 joint 固定 damping=1，未随机化；对轻小
   wrist/ankle 与重 knee 使用同值是否合理需要 decay/PRBS 数据，而不是直接加宽。
5. **并联踝 transmission。** 若部署转换不可忽略，优先做相同 kinematics/dynamics 或
   trace-driven residual，而不是只扩大 ankle gain。
6. **IMU 系统误差。** 增加安装角、固定 bias、慢漂移、轴间相关噪声前先核对实机 IMU
   frame 和部署中的 roll offset；当前只有每 step gyro white noise。
7. **contact compliance / geometry。** 目前随机化 µ，不随机化 `solref/solimp`、脚垫
   几何/高度、地面微坡与弹性。对平地 dance 不必一次全加；若主要失败来自触地冲击或
   脚边缘接触，再优先加入。
8. **小概率异常。** missed deadline、hold-last、packet drop 与 lag jitter 是不同机制；
   只有实测存在时再建模，不能用每步独立 uniform lag 代替。

## 7. 修复状态与后续验证顺序

1. **[已完成实现与核心路径回归]** delay reset 顺序已修复：Entity reset 清零后立即为
   新 reference 片段采样，reference wrap 同样采新 lag。待补完整 GPU env reset 后 fused
   buffer 观测。
2. **[已完成实现并增加顺序断言]** COM event 已通过 pop/reinsert 放到 pseudo-inertia 后。
   待补完整 env startup 后
   `body_mass/body_inertia/body_ipos` 最终分布断言。
3. **[已完成 API 修正，待物理标定]** joint friction 已选用绝对
   `abs U(0.01,0.6) N·m`；下一步用实测 frictionloss 判断范围并增加编译后数值分布断言。
4. 固化一份“训练 nominal ↔ 导出 YAML ↔ 实机最终 motor command”逐关节 Kp/Kd、
   effort limit、action scale 对照，并特别记录并联踝转换后的值。
5. 用真实日志测 command/encoder/IMU latency、gain response、friction 和 torque
   saturation，再决定是否调整范围；先做消融，不要同时增加所有候选 DR。
6. 保存 nominal、现有有效 DR、修复后 DR 三组评估，至少比较 tracking error、跌倒率、
   contact slip、峰值 torque 和 recovery performance。

## 8. 尚不能确认

1. TK3 实机端到端 command latency、joint/IMU observation latency、jitter、丢帧与各总线
   间相关性没有仓库内测量数据。
2. 实机实际加载的是公式 Kp/Kd 导出 YAML，还是显式高增益 YAML；板端是否二次缩放或限制
   Kp/Kd 也未知。
3. `effort_limit` 是峰值、持续值还是与实机电流/温度限制的何种对应关系，仓库无电机
   datasheet/驱动日志可核验。
4. 真实脚垫材料及目标地面的 µ 分布、contact compliance、脚底磨损/污染没有实测。
5. 并联踝变换对当前 dance motion 的误差大小、工作区是否接近奇异位形尚未量化。
6. 默认训练使用多少并行 env 可由 CLI/GPU 运行方式改变；因此 startup 固定参数的实际
   分布覆盖率不能只从 env cfg 判断。
7. joint friction 的绝对 `0.01–0.6 N·m` 是否适合全部 29 DoFs，仓库没有 breakaway、
   低速跟踪或 decay 辨识数据。
8. **[验证备注]** compatibility unittest 13 项、修改文件编译及 `git diff --check`
   均通过，其中包含 command delay 核心生命周期、COM 配置顺序和 soft torque-limit
   reward 数值回归。尚未运行完整 GPU env 读取 fused delay buffer 或 startup 后真实
   `body_ipos` 分布，因此这些单测不能替代完整运行时验证。
9. 本审计没有复跑长周期 PPO；结论覆盖配置/API/实现链与核心回归，不等于证明这些范围能产生
   最优 sim-to-real policy。

## 9. 一手来源索引

### 仓库

- `src/tasks/tracking/tracking_env_cfg.py:30-38,40-203,283-315`
- `src/tasks/tracking/config/tk3/env_cfgs.py:27-48,51-241`
- `src/tasks/tracking/config/tk3/__init__.py:11-18`
- `src/tasks/tracking/mdp/commands.py:297-355,348-490,577-592`
- `src/tasks/tracking/mdp/events.py:26-112`
- `src/assets/robots/tiangong3/tk3_constants.py:28-72,74-227`
- `src/assets/robots/tiangong3/xmls/tiangong3.xml:11-12,58-284`
- `scripts/train.py:38-43,64-94`
- `scripts/play.py:49-84`
- `src/tasks/tracking/rl/runner.py:141-170,190-221`
- `tests/test_mjlab_compat.py:209-386`
- `tests/test_tk3_tracking.py:77-81`
- `pyproject.toml:10-14`
- `uv.lock:721-843`

### MJLab v1.5.3 官方源码/文档

- [Event modes 与语义](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/managers/event_manager.py#L100-L141)
- [Event insertion order 与 strongest recompute](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/managers/event_manager.py#L260-L333)
- [Environment startup/step/reset 顺序](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/manager_based_rl_env.py#L299-L354)
- [Environment reset manager 顺序](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/manager_based_rl_env.py#L552-L585)
- [`abs` / `scale` / `add`](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/dr/_types.py#L82-L100)
- [Friction DR 默认只改 sliding axis 0](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/dr/geom.py#L117-L146)
- [Pseudo-inertia mass/inertia/COM 语义](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/dr/body.py#L416-L540)
- [Joint armature/frictionloss/encoder bias](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/dr/joint.py#L56-L110)
- [Native position actuator PD gain DR](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/envs/mdp/dr/actuator.py#L29-L188)
- [Actuator command delay 官方说明](https://github.com/mujocolab/mjlab/blob/v1.5.3/docs/source/actuators.rst#L245-L276)
- [DelayBuffer update/hold/reset](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/utils/buffers/delay_buffer.py#L120-L300)
- [Builtin actuator delay fusion](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/actuator/builtin_group.py#L98-L192)
- [Actor noise/corruption pipeline](https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/managers/observation_manager.py#L17-L24)

### MuJoCo 官方文档

- [Contact parameter mixing：explicit pair、priority、condim、friction](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)
- [Joint armature / damping / frictionloss XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
- [Position actuator XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position)
- [Actuation model 与 gain/bias force 生成](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model)
