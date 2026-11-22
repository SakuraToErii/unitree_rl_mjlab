> 恢复说明（2026-11-22）：从 2026-09-03 的会话记录还原，并保留当天稍后对动作尺度和 PPO 标准差的修订。正文描述的是当时的设定，不是当前分支。

# 端对端 0-PD 力矩控制：同类工作与奖励设计

调研日期：2026-09-03。问题：G1 29DoF 在 `IdealPdActuator(stiffness=0, damping=0)` +
`JointEffortAction` 下，PD 步态奖励抄过来之后回合变长、回报变负。需要看文献里真正
0-PD / 纯力矩策略是怎么训起来的，而不是继续盲调 `joint_acc_l2`。

本仓库当前设定（对照用）：

- 植物：0-PD + 理想力矩，无 T–N 曲线，无 `τ₀`。动作 `τ = clip(a · 0.25 · Y1, ±Y1)`。
- 策略频率：`sim.dt = 0.005`，`decimation = 4` → **50 Hz**。
- 奖励：mjlab 位置 PD 步态表的拷贝（跟踪 1.0、`joint_acc_l2 = -2.5e-7`、
  `effort_action_rate_l2 = -0.05`、gait/clearance/slip、姿态偏离）。
- PPO：`init_std=1`，`std_type="scalar"`，`std_range=(1e-3, 2.0)`，`clip_actions=None`，`entropy_coef=0.001`。

Isaac Lab 姊妹仓 `unitree_rl_lab/.../effort_loco` 同一机器人用 **残差力矩**
`τ = τ₀ + 0.4·Y1·a`，并且带 Unitree T–N 执行器；奖励仍是 PD 表，但多了
`alive = 0.15`。

## 结论

纯 0-PD 端对端力矩控制**有**成功工作，但成功案例几乎都不是「把 PD 奖励权重微调一下」。
它们改的是下面几件事里的至少一件：

1. **策略频率提到 100–250 Hz**（Chen、SATA、Kim、DecAP；Peng 2017 证明同频下力矩最差）。
2. **给力矩空间一个会消失的先验**：重力补偿预训练（Kim）、站立 `τ₀` 残差（本仓的 Isaac 版）、
   衰减 PD 力矩偏置（DecAP）、力矩上限/频率课程（SATA）。
3. **奖励以任务项压过无界正则**，或把正则改成有界 `exp(-k‖·‖²)`（Kim）。SATA 把线速度跟踪
   放到 `10 dt`，比 Rudin 的 `1.0` 高一个数量级。
4. **人形成功案例用模仿 + 相位**，不是从零拼 PD 步态项（Kim / TOCABI）。
5. 四足成功案例是 **12 DoF**。**没有找到公开的 G1 29DoF、无残差、50 Hz、纯 0-PD 走通的论文。**

对当前训练曲线：文献预测的就是「力矩探索慢、早期乱扭、正则随存活时间积分把回报打负」。
不是 `joint_acc_l2` 这项没意义，而是 **50 Hz 人形 0-PD 还顶着 PD 无界 L2 正则**，和成功论文
不在同一个 MDP 里。

## 工作谱系（按接近程度）

下面只收**策略直接出关节力矩、训练时 Kp=Kd=0 或等价**的工作。低增益 PD、残差 PD、
WBC+RL 力矩补偿不算 0-PD，单独注明。

### 1. Chen et al., Humanoids 2023 — 最接近的「纯力矩、无 PD」四足

- 来源：[arXiv:2203.05194](https://arxiv.org/abs/2203.05194)，
  [Humanoids 会议 PDF](https://hybrid-robotics.berkeley.edu/publications/Humanoids2023_RL_TorqueControl.pdf)
- 机器人：Unitree A1，**12 DoF**。Isaac Gym，4096 环境。PPO，MLP 512-256-128。
- 动作：`τ = clip(a · 10, ±33 Nm)`。无名义力矩。10/33 ≈ **0.30 Y1**（本仓 0.25 Y1）。
- 频率：**200 Hz** 策略直出力矩。对比的位置策略是 50 Hz + 200 Hz PD。
- 奖励：Rudin 风格，**权重里带 Δt=0.005**（与 mjlab `scale_by_dt=True` 同类）：

  | 项 | 表达式 | 权重 |
  |---|---|---|
  | 线速度 xy | `exp(-‖v*-v‖²/0.25)` | `1.0 Δt` |
  | 线速度 z | `-vz²` | `-2.0 Δt` |
  | 角速度 xy | `-‖ωxy‖²` | `-0.05 Δt` |
  | 角速度 z | `exp(-(ωz*-ωz)²/0.25)` | `0.5 Δt` |
  | 姿态 | `‖gxy‖²` | `-1.0 Δt` |
  | 关节力矩 | `‖τ‖²` | `-2e-4 Δt` |
  | 关节加速度 | `‖q̇_last - q̇‖²` | **`-2.5e-7 Δt`** |
  | 腾空时间 | `Σ(tswing-0.5)` | `2.0 Δt` |
  | 碰撞 | 膝盖触地 | `-1.0 Δt` |
  | 动作变化 | `‖a-alast‖²` | `-0.01 Δt` |
  | hip / thigh 名义角 | `|q*-q|` | `-1.0 Δt` |

- 他们写明：没有参考轨迹；hip/thigh 名义角只为「看起来自然」。
- 学习曲线：力矩 **前期更慢**，后期回报可以超过位置策略。位置策略的 Kp 会改整条曲线。
- 和本仓的差：12 DoF vs 29 DoF；200 Hz vs 50 Hz；无 gait/clearance/slip 堆叠；
  `joint_acc` 权重数字相同，但控制周期短 4 倍、机器人简单得多。

SATA 论文对 Chen 的评价（[arXiv:2502.12674](https://arxiv.org/abs/2502.12674) §I）：
靠「额外奖励项 + action scale」能训，但 **对超参敏感，早期探索效率低**。

### 2. Kim et al., 2023 — 唯一公开的人形尺寸 0-PD sim-to-real

- 来源：[arXiv:2304.09434](https://arxiv.org/abs/2304.09434)
- 机器人：TOCABI，1.8 m，**~100 kg**，下肢 12 关节力矩；上身仍是 PD + 重力补偿保姿态。
- 动作：`τ_cmd = π_θ`，**无 PD**。`Σ = τ_limit / sτ`，`sτ ∈ {5,10,20}` 都能学。
- 频率：**250 Hz**（消融可到 62 Hz）。
- 关键不是调 PD 奖励，而是：
  1. **重力补偿预训练**：用接触一致全身模型采 20 万样本，先学会撑住，再 fine-tune 任务。
     他们写：随机初始化的力矩策略必须先学承重，样本效率极差。
  2. **模仿奖励**（参考来自模型控制器 / DeepMimic），不是从零拼 tracking+gait。
  3. 正则是 **有界指数**，不是无界 L2：

  | 项 | 定义 |
  |---|---|
  | 基座姿态模仿 | `0.3 exp(-13.2‖qref-q‖)` |
  | 关节角模仿 | `0.35 exp(-4.0‖qref-q‖²)` |
  | 接触相位模仿 | 匹配则 0.2，否则 0 |
  | 速度任务 | `0.3 exp(-3.0‖vcmd-v‖²)` |
  | `q̇` 正则 | `0.05 exp(-0.01‖q̇‖²)` |
  | `q̈` 正则 | `0.05 exp(-20.0‖q̈‖²)` |
  | 接触力 | `0.1 exp(-5e-4 ‖F‖)` |
  | `ΔF` | `0.1 exp(-5e-4 ‖ΔF‖)` |
  | `τ` | `0.05 exp(-0.01‖τ‖)` |
  | `Δτ` | `0.2 exp(-0.01‖Δτ‖)` |

  `q̈` 项饱和在 0.05，活得越久也不会像 `-2.5e-7 ‖q̈‖²` 那样把回合回报打穿。

- 结论对调参的含义：人形 0-PD 的公开成功路径是 **撑住先验 + 模仿 + 高频 + 有界正则**，
  不是把 `joint_acc_l2` 从 `-2.5e-7` 改成 `-1e-7`。

### 3. SATA, RSS 2025 — 力矩空间能训，但植物不是裸 0-PD

- 来源：[arXiv:2502.12674](https://arxiv.org/abs/2502.12674)，代码
  [github.com/marmotlab/SATA](https://github.com/marmotlab/SATA)
- 机器人：Unitree Go2。策略出「激活」，再过 Hill 式力–速：

  `τ = τ_limit · α · (1 - sign(α) · q̇ / q̇_limit)`，`α` 还有一阶平滑 `γ=0.6`。

  这不是 IdealPd 0 增益；等效于 **内置 T–N + 低通**，和本仓 Isaac 版的 UnitreeActuator 一类。
- 课程（Gompertz `G(t)`）：力矩上限 7.05 → 23.5 Nm，频率 **100 → 200 Hz**。
- 奖励只有 9 项，`dt=0.005`：

  | 项 | 权重 |
  |---|---|
  | 跟踪 vx `exp(-4|e|)` | **`10 dt`** |
  | 跟踪 vy / yaw | `5 dt` |
  | 高度 `min(hb, ht)`（奖励不是惩罚） | `5 dt` |
  | roll、vz²、关节限位 | `-5 dt` |
  | 疲劳 `ζ·|τ|` | `-0.05 dt` |
  | `q̈²` | `-1e-6 dt` |

  早期 `G(t)≈0` 时跟踪的是「命令区间中点」而不是精确命令；侧向/偏航跟踪随 G 打开。
- 消融：去掉生物力学模型 **学不成步态**；去掉 growth 早期慢、高速指令差。
- 对调参的含义：他们把跟踪权重拉到 Rudin 的 **10 倍**，正则很少，并且用植物本身平滑力矩。
  本仓跟踪仍是 1.0，正则项更多。

### 4. DecAP, 2023 — 力矩空间样本效率问题的直接证据

- 来源：[arXiv:2310.05714](https://arxiv.org/abs/2310.05714)
- 主张：端对端力矩 **样本效率差、不一定收敛到自然步态**；只加模仿奖励不够。
- 做法：先训位置策略 → 用仿真跟踪到的 `q̂` 做模仿 → 把
  `β = Kp(q̂-q) + Kd(-q̇)` 加到力矩动作上，再 `γ^{t/k}` 衰减到 0。
- 动作 scale：**8.0**（Go1/Cassie/Daisy），200 Hz。
- 任务奖励仍是 Rudin 表（`joint_acc` 也是 `-2.5e-7 dt`）。
- **真机部署仍叠了低增益 PID**（论文自己画了「position-assisted torque」）。严格说部署不是 0-PD。
- 对调参的含义：他们明确写「克服力矩探索差通常需要多轮参数和 shaping」，
  所以发明了动作先验，而不是继续拧奖励权重。

### 5. Peng & van de Panne, SCA 2017 — 动作空间对照（仿真角色）

- 来源：[arXiv:1611.01055](https://arxiv.org/abs/1611.01055)
- 同频对比：PD 目标角始终更快、更稳；**直接力矩最差**，角色越复杂差距越大。
- 降频时 PD/速度动作还能撑，力矩策略迅速垮。
- 这是后来所有「力矩要更高频」说法的源头之一。Chen / Kim 都引用这篇。

### 6. Haarnoja et al., RSS 2019 — 常被误当成力矩，其实是低增益 PD

- 来源：[arXiv:1812.11103](https://arxiv.org/abs/1812.11103)
- Minitaur 真机 SAC。动作是腿摆/伸，再映射成电机位置，`kp=0.3, kd=0.003`。
- 奖励几乎只有前进 `- |ä| - |roll|`。说明「极简奖励」可以走，但植物不是 0-PD。

Gym / MuJoCo Humanoid、PyBullet Minitaur 的 `|τ·q̇|` 能量项属于玩具环境，不作为 G1 权重来源。

### 7. 本仓姊妹实验 `unitree_rl_lab` `effort_loco`（G1 29DoF）

路径：`/home/ordis/projects/unitree_rl_lab/.../effort_loco/robots/g1/29dof/`。

- 植物：Unitree T–N + `stiffness=damping=0`。
- 动作：**残差** `τ = τ₀ + 0.4·Y1·a`，`τ₀` 来自位置策略站立窗口。注释写 0.4 是站立/平衡用，
  动态步态可提到 0.5–0.7。
- 时序：`decimation=4`，与本仓一样 50 Hz。
- 奖励：PD 速度任务拷贝，但有 **`alive = 0.15`**（本仓 mjlab effort 没有这项），
  跟踪 xy=1.0、yaw=0.5，`joint_acc=-2.5e-7`，`effort_action_rate=-0.05`。

这不是论文，但是目前最接近「G1 29DoF 0-PD」的实现。它已经承认：**裸 `τ=a·scale` 撑不住，
要用站立力矩当偏置**。本仓 mjlab 版去掉了 `τ₀`，探索比 Isaac 版更难。

## 和当前 mjlab effort 的对照

| | 本仓 mjlab | Chen A1 | Kim TOCABI | SATA Go2 | Isaac `effort_loco` |
|---|---|---|---|---|---|
| 形态 | 人形 29DoF | 四足 12 | 人形下肢 12 | 四足 12 | 人形 29DoF |
| 植物 | IdealPd 0 | 理想力矩 | 理想力矩 | Hill T–N | Unitree T–N |
| 偏置 | 无 | 无 | 重力预训练 | 无，但限幅课程 | **τ₀ 残差** |
| 策略频率 | **50 Hz** | 200 Hz | 250 Hz | 100→200 Hz | 50 Hz |
| 动作 scale | 0.25 Y1 | ~0.30 Y1 | `τlim/sτ` | 课程限幅 | 0.4 Y1 |
| 跟踪权重 | 1.0 / 1.0 | 1.0 / 0.5 | 模仿+0.3 | **10 / 5 / 5** | 1.0 / 0.5 |
| `q̈` | `-2.5e-7` L2 | 同左 | **有界 exp** | `-1e-6` 但跟踪×10 | `-2.5e-7` |
| 存活项 | 只有摔倒 -200 | 无显式 alive | 模仿撑姿态 | 高度正奖励 | **alive 0.15** |
| 步态先验 | phase+gait+clearance | 腾空时间 | 接触模仿 | 无 | gait+clearance |

数字要对齐：mjlab RewardManager 默认 `reward = raw * weight * dt`（`dt=0.02` 在 50 Hz）。
Chen 表里的 `weight × Δt` 是同一类时间归一化。所以 **`-2.5e-7` 这个数字 Chen 也用了**，
不能解释成「权重抄错了一个数量级」。差在频率、自由度和有没有撑住先验。

## 对「奖励到底怎么调」的文献答案

文献**没有**给出 G1 29DoF 的现成权重表。它们给出的是优先级：

1. **先改 MDP，再拧权重。** Peng / Chen / SATA / Kim 一致认为力矩策略在 50 Hz、无先验时
   探索是病态的。继续在当前表上加减 `joint_acc` 权重，是 Chen 被 SATA 批评的那条路。
2. **`joint_acc_l2` 该留，但不要让它随存活无界积分吃掉跟踪。** Kim 改成
   `0.05 exp(-20 ‖q̈‖²)`；SATA 用更强跟踪盖过它；谁都没删这项。
3. **跟踪/存活必须在早期占主导。** SATA 把 vx 放到 10；Isaac 版加了 `alive`；Kim 用
   模仿+重力预训练让「站住」一开始就有回报。本仓 `is_terminated=-200` 摊到 20 s 上大约
   恒为 -0.2，活 8 步和活 400 步这项一样，所以会出现「步数涨、回报跌」。
4. **人形不要指望 Rudin 腾空时间自发变成步态。** 公开人形力矩成功用了参考运动。四足 Chen
   可以靠 air time，因为 12 关节、200 Hz。
5. **动作先验比奖励 shaping 更有效。** Kim 的重力预训练、DecAP 的衰减 PD、Isaac 版 `τ₀`、
   SATA 的力矩课程，全是改动作/植物，不是再加一项 `joint_acc`。

若仍坚持「只动奖励、不动频率/偏置」，文献里最接近还能工作的改法是 SATA 那套：
砍掉大部分 PD 步态 shaping，把跟踪提到 ~10×，高度改成正奖励，正则留很少的有界项。
这仍然是在 12 DoF、100–200 Hz、带 T–N 的四足上验证的，搬到 G1 没有保证。

## 建议的实验顺序（未实施）

按文献成功概率，而不是按改起来方便：

1. 恢复 Isaac 版站立 `τ₀` 残差（或 Kim 式重力预训练），50 Hz 先看能不能站住。
2. `decimation` 4→2 或 1（100/200 Hz），与 Chen/SATA/Kim 对齐。
3. 植物加上 T–N（SATA / UnitreeActuator），而不是 IdealPd 饱和盒。
4. 奖励：加与时长成正比的存活项；把 `joint_acc` / `action_rate` 改成有界核或降权到
   跟踪项能付得起；不要在没步态时堆 clearance/slip。
5. 若 1–4 仍不走：Kim/DecAP 路径，用现成位置策略做模仿或衰减 PD 先验。

## 未纳入 / 证据不足

- 公开 G1 位置 PD 工作（Isaac Lab G1 locomotion 等）全部是 ImplicitActuator + 关节目标角，
  与 0-PD 无关。
- arXiv:2607.18365 *Towards Torque-Driven Reinforcement Learning for Quadruped Locomotion*
  是 2026-07 预印，奖励表更重，未做硬件；只作旁证，不当权重来源。
- 未逐行核对 SATA GitHub 与论文表是否完全一致；奖励数字以论文 Table I–III 为准。
- Haarnoja / Gym Humanoid 不应当作 G1 权重来源。

## 来源

- Chen et al., *Learning Torque Control for Quadrupedal Locomotion*, Humanoids 2023 / [arXiv:2203.05194](https://arxiv.org/abs/2203.05194)
- Kim et al., *Torque-based Deep Reinforcement Learning for Task-and-Robot Agnostic Learning on Bipedal Robots*, [arXiv:2304.09434](https://arxiv.org/abs/2304.09434)
- Li et al., *SATA*, RSS 2025 / [arXiv:2502.12674](https://arxiv.org/abs/2502.12674), [marmotlab/SATA](https://github.com/marmotlab/SATA)
- Sood et al., *DecAP*, [arXiv:2310.05714](https://arxiv.org/abs/2310.05714)
- Peng & van de Panne, *Does the Choice of Action Space Matter?*, SCA 2017 / [arXiv:1611.01055](https://arxiv.org/abs/1611.01055)
- Haarnoja et al., *Learning to Walk via Deep Reinforcement Learning*, RSS 2019 / [arXiv:1812.11103](https://arxiv.org/abs/1812.11103)（低增益 PD，对照用）
- 本仓 `src/tasks/effort/velocity_env_cfg.py`、`src/tasks/effort/config/g1/action_cfg.py`
- `unitree_rl_lab` `tasks/effort_loco/robots/g1/29dof/effort_env_cfg.py`
- mjlab `RewardManager`：`reward = raw * weight * dt`
