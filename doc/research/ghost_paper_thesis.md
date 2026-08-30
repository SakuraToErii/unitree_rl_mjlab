# Ghost 点子：优化后的论文切口与运控问题地图

调研日期：2026-08-30  
口径：本文区分 **[仓库事实]**、**[论文原文]**、**[判断]**。  
对象：`src/tasks/ghost/`（main）与 `origin/prototype/ghost-qref-residual`（q_ref 残差原型）。  
问题：这个点子有没有办法写成中科院一区/二区论文？它能对准哪些具身智能 / 人形运控难点？

---

## 0. 先给结论

**可以写成论文，但不能按现在这条 Ghost 去写。**

| 投稿目标 | 现有 Ghost 够不够 | 优化后是否现实 |
|---|---|---|
| 中科院小类机器人学 **一区**（Science Robotics） | 不够，差一个量级 | 基本不现实：需要全新能力 + 大规模真机，且 2026 年「洗参考」已被占 |
| 国内常算 **一区** 的 TRO / IJRR（大类计算机 1 区、小类 2 区 Top） | 不够 | 只有走「在线因果物理解投影 + 多接触支撑发现 + 真机」这条硬切口，才有机会；否则会被看成 OmniTrack 跟进 |
| 中科院小类 **二区** RA-L，或 ICRA / IROS + RA-L option | 不够 | **最匹配**。把 Ghost 从「OmniTrack Stage I 的 TK3 复现」改成「带硬可行性门的残差物理投影器」，补 Stage II、消融和至少一组真机/严格 sim-to-sim |
| 纯工程笔记 / 内部技术报告 | 已经够 | 质量协议、鞋底几何、TCP 允许接触，都有工程价值，但撑不起 SCI 贡献点 |

一句话：**「仿真里 rollout 一遍，把穿地/浮空洗掉再跟踪」这件事，OmniTrack 已经写成了主贡献。** Ghost 现在几乎就是这件事在天工人形 3（TK3）上的特权 PPO 实现。要投稿，必须把贡献从「我们也有 PMG」改成「PMG 还缺的那一层结构」：残差、硬约束、接触模态、在线因果性。不要声称「首次物理一致参考」「首次 \(q_{\mathrm{ref}}\) 残差」「首次在线物理化」——这三条 2026 年会被直接 desk-reject。

---

## 1. 当前 Ghost 实际是什么

### 1.1 它自己怎么定位

[仓库事实] `src/tasks/ghost/PROTOTYPE.md` 写明这是一个 **一次性、单 clip 的物理参考生成器**，不是可部署策略，也不是 teacher-action 蒸馏：

> Can a privileged PPO policy turn one noisy 100 Hz TK3 motion into a torque-limited MuJoCo rollout with no hard joint-limit violation, deep ground penetration, or unsupported static floating while preserving its global trajectory?

它明确对齐 OmniTrack Stage I（Physical Motion Generation, PMG）：特权状态、无观测噪声、无物理域随机化、参考指令噪声、放宽早停、失败驱动分段采样、真机力矩上限、把仿真 rollout 当作 Stage II 参考。

[仓库事实] Stage II 只写在文档里，仓库里的可部署跟踪仍是 `src/tasks/tracking/`（BeyondMimic 式、有 DR）。Ghost → Tracking 的闭环 **还没接上**。

### 1.2 和 OmniTrack 的重合度

OmniTrack 原文（[Li et al., arXiv:2602.23832](https://arxiv.org/abs/2602.23832)，§III-A、附录 A.5–A.7）：

- 特权 generalist 策略当滤波器，在仿真里 rollout 出动力学一致轨迹。
- Stage I **不做** domain randomization，只对参考 `q, q̇, 根位姿` 加有界噪声。
- 放宽早停；adaptive sampling（1 s bin，`α=0.001`，clamp `[0.75/N, 100/N]`）。
- 真机力矩上限；保存关节、全局位姿/速度、接触。
- 奖励：14 个 key body 的位姿/速度高斯核 + 根位姿 + 非末端接触惩罚。
- Stage II：部分可观 `(q, q̇, R, ω, a_{t-1})` + desired-contact + 全套 DR。
- 动作只写成「输出目标关节位置 \(a_t\) 给 PD」。**没有公布** 是绝对目标、BeyondMimic 式 \(q_0\) 偏置，还是 \(q_{\mathrm{ref}}+a\)。不要默认 OmniTrack 已经做了残差 PMG。
- 真机：Unitree G1、LAFAN1 零样本、一小时连续、侧空翻、VR/动捕遥操作。
- 对比停在 DAgger / AAC / OmniH2O / ExBody2 / BeyondMimic，**没有** 对 DSMS、DynaRetarget、OmniRetarget、I-CTRL、PhySINK。

[仓库事实] Ghost 复用了同一套骨架（特权 obs、指令噪声退火、adaptive sampling、undesired contact、力矩限幅、rollout 存接触掩码）。额外工程选择：

- 平台换成 TK3；仿真是 mjlab / MuJoCo Warp，不是 Isaac Lab。
- **单 clip specialist**，不是 OmniTrack 的 generalist。
- 鞋底凸包 + 橡胶接触（`tk3_constants_ghost.py`），TCP / 腕允许撑地。
- 根 XY / Z / 姿态拆开奖励；身体奖励用 root-relative、heading-aligned。
- 观测是 error-centric preview（`future_motion_goal`），不是把绝对参考塞进网络。
- `rollout.py` 的质量门比 OmniTrack 附录更严、更物理：FK 自洽、硬限位、力矩越限、穿地 >5 mm、**无接触且近似静止且不在自由落体** 的悬空（OmniTrack 用「根高 >0.8 m 连续 1 s」，会误伤跳跃、放过蹲着漂）。

这些是 **实现质量**，不是新问题定义。评审会问：去掉 TK3 外壳，贡献还剩什么？

### 1.3 q_ref 残差原型在做什么

[仓库事实] `origin/prototype/ghost-qref-residual` 把动作从

\[
q_{\text{cmd}} = q_0 + s\cdot a
\]

改成

\[
q_{\text{cmd}} = \mathrm{clip}\big(q_{\text{ref}} + s\cdot \mathrm{clip}(a, \pm 1),\; 0.98\cdot\text{joint limits}\big)
\]

并拿掉 `motion_joint_pos/vel` 奖励、放松足/腕竖直跟踪、加上穿地/自碰/关节加速度惩罚。玩具逻辑在 `qref_residual_model.py`：隔离的是 **动作中心** 和 **风格误差 vs 物理质量的优先级**。

这是 Ghost 里相对 OmniTrack **正文未写死** 的结构（OmniTrack 动作中心未发表）。它仍然不是空白：图形学 SuperTrack（2021）已用参考附近的 PD 偏置；PHC 明确 **拒绝** \(q^d=\hat q+a\)，理由是噪声参考会把残差绑死在坏轨迹上；I-CTRL / RobotDancing / DynaRetarget 的 RL 阶段都用过 \(q_{\mathrm{ref}}\) 残差。Ghost 要回答的是 PHC 的反例：坏参考上残差是否仍优于重写整段动作。

---

## 2. 相关工作已经占住的坑

按「Ghost 容易撞车」的程度排列。引用均为一手摘要/方法节。

### 2.1 物理可行参考（和 Ghost 主叙事几乎同构）

| 工作 | 做法 | 对 Ghost 的含义 |
|---|---|---|
| **OmniTrack** [2602.23832] | 特权 PMG rollout → 部分可观 GMT；真机 G1 | **主贡献已被占用。** 「解耦可行性与跟踪」不能再当 title |
| **PHUMA / PhySINK** [2510.26236] | 大规模视频 + 物理约束优化（关节限位、着地、消滑步） | 优化式物理化已有；Ghost 必须证明 **仿真交互** 优于 **约束优化**，不能只说「我们用了物理」 |
| **OmniRetarget** [2509.26633] | interaction mesh + 运动学约束，保人–物–地形接触；G1 跑酷/loco-manipulation | 场景交互的运动学保真已被占；Ghost 的增量应在 **动力学/力矩/支撑**，不是再做一遍几何 retarget |
| **I-CTRL** [2405.08726]，IEEE RAM 2025 | **有界残差 RL**，在非物理 retarget 上做 constrained refinement；5 个双足、~9k 动作、同一套奖励 | **残差物理化已经发过杂志。** Ghost 的 q_ref 残差若停在「跟踪策略输出 Δq」，会被直接对比 I-CTRL |
| **KungfuBot** [2506.12851] | 启发式接触/浮空修正 + Mink IK，再 **每 clip 一个** RL；自适应 σ | 「先改参考再跟踪」不是新故事；且 KungfuBot **并不** 用动力学 rollout 当参考 |
| **GMR** [2510.02252] | 非均匀缩放 + 两阶段 IK；BeyondMimic 下游对 retarget 伪影极敏感 | 运动学基线。Ghost 的 Stage I 对比至少要包含 GMR/Mink，不能只打 raw CSV |
| **ASAP** [2502.01143] | MaskedMimic 做 sim-to-data 清洗 + 真机 delta action 对齐动力学 | 「仿真里能跟踪」≠「真机能跟踪」；Ghost 的物理参考只对 **名义仿真** 一致 |

另有一类 **轨迹优化 / 隐式接触**，评审会拿来打「RL Ghost 并不硬约束」：

| 工作 | 做法 | 对 Ghost 的含义 |
|---|---|---|
| **Opt2Skill** [2409.20514]，RA-L | DDP 出含力矩的 Digit 参考，再 RL 跟踪 | TO-then-track 已发 RA-L；面向任务接触，不是野 mocap |
| **DynaRetarget** [2602.06827] | IK 后再做采样 TO，修 OmniRetarget 缺接触/穿地；RL 用残差 + 仿真接触。约 **1 分钟算 1 秒动作** | 离线 loco-manip 接触修复已被占；Ghost 的活口是 **因果/实时**，不是再做一个更慢的箱子 TO |
| **DSMS** [2608.03116] | 可微 MuJoCo 里 contact-implicit multiple shooting，**不预设接触日程**；真机爬行（手/肘/膝）+ 180° 跳转 | **手膝当承重支撑、从动力学里长出额外接触** 已被演示。Ghost 不能写「首次发现多接触」；只能写规模化 / 野 mocap / 在线 |
| **KDMR** [2603.09956] | 多接触全身 TO + GRF 触发的足部事件 | 需要地面反力，视觉 mocap 没有 |
| **SPIDER** [2511.09484] | 采样 + 虚接触课程，大规模运动学→动力学 | 规模靠采样，不是 generalist Ghost |

### 2.2 残差动作（和 q_ref 原型同构）

| 工作 | 残差加在哪 | 和 Ghost 的差别 |
|---|---|---|
| **SuperTrack**（TOG 2021） | 图形学角色：运动学关节上的 PD **偏置**，消融显示优于绝对目标 | 残差-绕-参考比人形 RL 潮更早；不能当 2026 新方法 |
| **PHC** [2305.06456] | 讨论过 \(q^d=\hat q+a\)，因噪声参考而改用 **绝对** PD 目标 | 对 Ghost 最硬的反例：坏 mocap 上残差可能更差 |
| **RobotDancing** [2509.20717] | **可部署跟踪器**：`q_cmd = q_ref + Δq`，常 **每条序列一个策略**；髋/膝选择性残差 | 残差在部署端。Ghost 原型放在 **特权 Stage I** |
| **I-CTRL** | 有界残差 + 约束 MDP（参考周围 hypertube） | 残差就是控制策略；无「洗完再训 GMT」；无环境交互 |
| **DynaRetarget** | TO 之后的 RL 用残差动作空间 | 残差跟在慢 TO 后面，不是特权 PMG 本身 |
| **ResMimic** [2510.05070] | GMT **动作** 上残差 \(a_{\mathrm{GMT}}+\Delta a\)，不是 \(q_{\mathrm{ref}}\) | 物体 loco-manipulation |
| **ASAP** | 真机数据学 \(\Delta a=\pi^\Delta(s,a)\)；全文 4 个踝关节残差才稳（23 维会过热、摔过两台 G1） | 残差补仿真器，不是参考库 |
| **BeyondMimic** | \(q=\bar q+\alpha a\)，绕 **默认姿态** 不是绕参考 | Ghost main 目前就是这个家族 |

[判断] 「输出 Δq 而不是绝对关节目标」本身 **不够当 2026 年二区贡献点**。能写的是：在 **特权投影器** 里用残差，并配硬可行性门，使 Stage II 不再判断「参考该不该跟」——这是 OmniTrack 的解耦 + I-CTRL/RobotDancing 的残差，组合后必须用实验证明 1+1>2，并正面回答 PHC「噪声参考不该残差」。

### 2.3 一般运动跟踪（Ghost 的下游，不是 Ghost 自己）

BeyondMimic [2508.08241]、GMT [2506.14770]、UniTracker [2507.07356]、OmniH2O、ExBody2、TWIST、AnyTrack、Sonic：都在打 **部分可观、长时程、泛化跟踪**。OmniTrack 已经用 PMG 把它们当 baseline 打过。Ghost 若只交 Stage I 指标、没有 Stage II 对比，审稿人会说缺下游证据。

---

## 3. 还能对准哪些真正没解决的运控问题

下面只列 **OmniTrack / I-CTRL / PHUMA / RobotDancing / DSMS / DynaRetarget 之后仍然空着、且 Ghost 代码已经碰到边** 的问题。每条标注：问题是否真实、Ghost 现在是否真的在解、要写成论文还缺什么。

### 3.1 跟踪精度 vs 平衡稳定的冲突（真问题，但主叙事已被占）

[论文原文] OmniTrack §I：形态差 + mocap 噪声 → 浮空/穿地；部分可观策略没有特权信息判断「这个参考该不该跟」，于是盲目追奖励、动态失败、generalist 不收敛。

Ghost 现在：用特权 PPO + 软奖励 **缓解** 冲突，没有 **消除** 冲突——穿地/悬空仍是惩罚项，质量门在 rollout **之后**（`quality_report`），训练时策略仍可在穿地轨迹上拿跟踪分。

要升级成贡献：把可行性变成 **训练中不可妥协的约束**（CMDP / Lagrangian / 门控奖励 / 拒绝采样），跟踪只在可行集上优化。OmniTrack 说「strictly adhere to dynamics」只表示 **rollout 是仿真轨迹**，力矩/接触/风格仍是软代价。这才是「解耦」的算法化，而不只是「分两阶段训练」。DSMS / Opt2Skill 已经在少数技能上做了动力学硬约束；Ghost 的空档是 **野 mocap、数据集规模、实时**，不是「RL 也能约束」。

### 3.2 风格保真 vs 物理可行的 Pareto（真问题，几乎没人报告）

OmniTrack Table I：物理化后 LAFAN1 MPJPE +21 mm、AMASS +16 mm，穿地/浮空到 0。这是 **一个工作点**。I-CTRL 用固定 hypertube \(\delta\) 限风格偏离；KungfuBot 用自适应 \(\sigma\) **放松跟踪**；SoftMimic [2510.17792] 用柔顺换刚度。仍然几乎没人报告 **用户可控的 Pareto**（风格权重或 \(\delta(t)\)）以及失败时改高度、改接触、还是拒绝 clip。

[仓库事实] q_ref 玩具原型已经把 `tracking_weight` vs `feasibility_weight` 当成核心旋钮。论文里这应升格为：

> 在机器人欠驱动动力学流形上，求与 mocap 的最小风格距离，并报告可行集上的 Pareto。

这能回答运控里一个很具体的工程问题：**这段动作根本做不到时，策略该改高度、改接触，还是拒绝执行？** 现有跟踪器 implicit 地乱选。

### 3.3 绝对动作把容量浪费在「重写动作」上（真问题，残差文献已部分回答）

RobotDancing / I-CTRL：长时程高动态上，绝对关节指令要重新合成整段动作，误差累积。残差把容量留给动力学补偿。

Ghost 的差异化：**把残差用在特权滤波器，而不是部署策略。** 理由：

- Stage I 的任务本来就是「最小修正」；`q_cmd = q_0 + s a` 强迫网络先学会「跟踪参考」，再学会「偏离参考」，和目标相反。
- 原型注释写得很清楚：再奖励 `q/q̇` 精确跟踪会抑制对坏参考的修正。
- Stage II 可以继续用绝对或残差；论文要消融的是 **PMG 动作中心**，不是再发一篇 RobotDancing。

未解决点：残差半径（`scale=0.25 rad`）能否覆盖「必须改接触模态」的大修正（摔倒爬起、手撑地）？太大则退回绝对动作，太小则洗不掉浮空。I-CTRL 的 \(\delta\) 是固定状态管；需要 **按接触相位自适应的残差半径**，并在坏参考上证明残差不比 PHC 的绝对动作更糟。

### 3.4 接触模态从 mocap 迁不过来（真问题，比舞蹈跟踪更空）

人形运控里比「跟着跳舞」更难的是：

- 手撑地、爬行、起身、膝/肘接触（OmniTrack 有 getting up，但接触奖励仍粗）。
- 人掌 ≠ 机器人 TCP；允许接触集合必须按硬件重写。
- 运动学 retarget 保了相对位置，不保接触力/摩擦锥/支撑多边形。

[仓库事实] Ghost 已把 `left_tcp_link / right_tcp_link / wrist_pitch_*` 列入允许接触，undesired contact 惩罚其余；质量报告统计 undesired frames。这是 **loco-support（用手当支撑）**，还不是 ResMimic 那种带物体的 loco-manipulation。

空档要写窄：**从野 mocap（无预设接触日程、无 GRF）里，在线或近实时地长出承重接触**，并在数据集规模上归档。已经不能写「首次手膝支撑」——DSMS 真机爬行已经演示接触发现；OmniRetarget / DynaRetarget 修的是场景/物体接触，但是离线运动学或分钟级 TO。KungfuBot 的接触掩码是踝关节阈值启发式。Ghost 若只在平地舞上允许 TCP 接触，这条贡献是空的。

### 3.5 在线遥操作的因果性与延迟（真问题，OmniTrack 只演示了管线）

OmniTrack §III-C：动捕/VR → 仿真里跑 PMG → GMT 上真机。这是 **两条策略 + 一台影子仿真器**。原文几乎不谈：

- PMG 前视窗口（Ghost 用 0/50/100 ms）在直播里是否非因果；
- 影子仿真与真机状态不同步时，投影还对不对；
- 100 Hz 残差修正 vs 整段 clip rollout 的延迟。

残差 PMG 更适合做成 **因果滤波器**：当前帧 `q_ref` 进来，输出有界 `Δq`，不需要在仿真里「演完整个未来」。DynaRetarget / DSMS 是 **离线分钟级**，不能当遥操作。OmniH2O / TWIST / CLONE 有遥操作，但没有显式 Ghost。活口是 **测过的端到端延迟、因果前视、抖动 VR**，不是再发一段遥操作视频。

### 3.6 「对哪套物理一致」（真问题，Ghost 现在完全没碰）

Ghost / OmniTrack 的「物理一致」= **名义仿真器 + 真机力矩上限**。ASAP 证明：敏捷技能的瓶颈是 sim–real 动力学差，不是参考漂不漂。

因此还存在：

- 对 Isaac 一致的参考，在 MuJoCo / 真机上仍穿地或力矩饱和；
- 用 DR 训出来的 GMT 会故意偏离「物理参考」，因为参考绑的是错的物理。

一区级问题：**物理参考应该关于 identified / residual-aligned 动力学生成，而不是关于训练用的干净仿真。** 这需要真机轨迹，工作量接近 ASAP + OmniTrack。二区论文可以先做 **sim-to-sim**（Warp ↔ 原生 MuJoCo ↔ 另一套接触参数），证明「PMG 过拟合接触模型」。Ghost 的 FK 自洽检查已经在碰仿真器差异，可以做成实验，而不是附录里的 sanity check。

### 3.7 可行性度量本身不可靠（真问题，适合当论文的 evaluation 贡献，不够当唯一贡献）

OmniTrack 浮空定义（根高 >0.8 m 且持续 1 s）对 G1 站立身高还凑合，对跳跃误报、对蹲着漂漏报。Ghost 的 unsupported static（无地面接触 ∧ |v_z|<0.1 ∧ |a_z+g|>2）更接近「仿真在吊着机器人」。

运控社区缺少 **与形态无关的参考质量协议**：穿深、支撑完整性、摩擦锥、力矩裕度、接触切换抖动。Ghost 的 `metrics.json` 是雏形。可以当论文的第三个贡献（benchmark），不能当第一个。

### 3.8 哪些难点 Ghost **解决不了**、不要写进 contribution

| 难点 | 为什么不要认领 |
|---|---|
| 部分可观 generalist 跟踪本身 | 那是 Stage II / BeyondMimic / OmniTrack GMT |
| 敏捷技能的 sim-to-real（空翻落地软硬） | ASAP；Ghost 关掉 DR 是反方向 |
| 带物体的 loco-manipulation | ResMimic / OmniRetarget / DynaRetarget；仓库没有物体 |
| 「首次」仿真可行参考或 \(q_{\mathrm{ref}}\) 残差 | PHC、I-CTRL、Opt2Skill、OmniTrack、DSMS、SuperTrack、RobotDancing |
| 视觉–语言–动作、场景理解 | 观测里没有视觉 |
| 形式化安全证书（CBF/可达集） | 仿真 rollout 只证明「这个仿真器里没炸」，不是安全 |
| 电池、热、关节温度长时约束 | 质量门没有热模型 |
| 非平地运动跟踪 | 场景是 plane |

---

## 4. 不该写的论文（避雷）

1. **「TK3 上的 OmniTrack」**  
   换机器人 + 换仿真器。二区都会嫌 incremental；会议 poster 都勉强。

2. **「我们提出残差动作」**  
   SuperTrack、I-CTRL、RobotDancing、DynaRetarget、ResMimic、ASAP 都是残差。必须写清残差加在 **特权投影器**，正面回答 PHC 的反例，并证明它改变了可行性–风格权衡。

3. **只有 Stage I、没有下游**  
   OmniTrack 的核心实验是：物理参考让 GMT 在数据变大时仍稳（Fig. 4, Tab. A.8）。Ghost 若只报穿地/浮空下降，没有「Stage II 成功率和 MPJPE」，贡献停在数据清洗。

4. **单 clip specialist 冒充 generalist**  
   原型故意 one motion at a time。OmniTrack 的卖点是 **一个** PMG 洗 LAFAN1+AMASS。要么做 generalist，要么把 specialist 当成「高质量归档器」并论证何时优于 generalist（难 clip 的可行性门通过率）。

5. **用 OmniTrack 的浮空/穿地定义却宣称全面更好**  
   度量不一致。应同时报他们的定义和 Ghost 的定义。

6. **「首次在线物理化」**  
   OmniTrack §III-C 已经把 GMR → 仿真 PMG → 真机 GMT 写成应用。没有延迟/因果消融就是跟进。

7. **「RL rollout 等于硬约束满足」**  
   评审会拿 DSMS / Opt2Skill。软 PPO + 事后 `quality_passed` 不能声称 constraint satisfaction。

---

## 5. 推荐的三条切口（按投稿目标）

### 切口 A — 二区主力（建议就写这个）

**题目方向：** Residual Physical Projection for Humanoid Motion Tracking  
**一句话主张：** 把运动学 mocap **投影** 到机器人力矩/接触可行集上；投影器是特权残差策略，可行性是硬门，风格是可行集上的目标，而不是和穿地惩罚拧在一起的软权重。

**算法要点（相对现有 Ghost 必须改的）：**

1. 动作：`q_ref` 残差（原型已有），去掉关节精确跟踪奖励（原型已有）。
2. 训练目标改成两层，而不是再堆高斯核：
   - **硬门（失败/拒绝）：** 穿深 >2–5 mm、unsupported static、硬限位、力矩越限、非允许接触持续超阈。
   - **软目标：** heading-aligned 身体形状、根 XY、允许末端水平位置；根 Z 与末端高度只在硬门满足后计分。
3. 残差半径：默认小；当接触门触发时允许更大 Δq（自适应 hypertube，区别于 I-CTRL 的固定有界残差）。
4. 输出：物理 npz + 接触掩码 + **质量证书**（现有 `quality_report`）。
5. Stage II：用证书通过的参考训 `TK3-Tracking`；对照 raw retarget / 绝对动作 PMG / 残差 PMG。

**必须有的消融：**

| 消融 | 用来挡住哪条审稿意见 |
|---|---|
| `q0` 绝对动作 vs `q_ref` 残差 | 「不就是 OmniTrack」 |
| 硬门 vs 仅软惩罚 | 「不就是多几个 reward」 |
| 有/无末端竖直放松 | 风格被物理修正毁掉 or 穿地洗不掉 |
| specialist vs 一个 generalist PMG | 规模化 |
| raw vs 物理参考训 Stage II | OmniTrack Fig. 4 的必要复现 |
| 质量定义：OmniTrack 浮空 vs Ghost unsupported static | 度量抬杠 |

**最低实验规模：**

- 数据：≥ 一个公开集（Unitree-retargeted LAFAN1 或等价 TK3 重定向）+ 一组 **接触丰富** clip（起身、手撑、跪、爬）。只有舞蹈不够。
- Stage I：穿透/悬空/力矩饱和 vs raw、vs GMR/Mink 运动学基线、vs 绝对动作 PMG。若声称多接触，必须能说明和 DSMS（慢、少技能）或 DynaRetarget（离线 TO）差在实时/规模，而不是「我们也会爬」。
- Stage II：成功率、MPJPE、Δvel，seen / hard / 若可能 unseen。
- 真机：至少 3–5 条（走、舞、起身或手撑过渡）。没有真机就走 RA-L 的 sim-to-sim 强化 + 明确写成 limitation。

**能写进 abstract 的运控难点：** 不可行参考导致的跟踪–稳定冲突；长时程误差累积（通过洗参考而不是加部署残差）；接触伪影（穿地/假支撑）。

### 切口 B — 冲 TRO / 顶会的加码版（在 A 做成之后）

**题目方向：** Causal Residual Physicalization for Online Whole-Body Teleoperation  
**额外主张：** 同一套残差投影器可以 **在线、因果** 地吃脏动捕/VR 流，有界延迟内给出力矩可行命令；并在手/脚多接触支撑切换上保持支撑完整性。

必须多出来的证据：

- 延迟扫描：0 / 20 / 50 / 100 ms 前视对稳定性与风格的影响。
- 把 PMG 当实时滤波器（逐步）vs 离线整段 rollout，对比。
- 噪声动捕：抖动、丢帧、根漂。
- 多接触日程：起身、爬行、手撑站起；接触切换相对 mocap 的编辑距离。
- 真机遥操作，而不仅是离线 replay。

这对准的难点是 **在线全身遥操作在脏指令下的稳定性**（TWIST/OmniH2O/OmniTrack 都演示过，但缺少因果投影的定量）。必须能说明为何不用 DSMS（太慢）而用残差 RL。没有真机遥操作不要投 TRO。

### 切口 C — 不要单独投，可并进 A 的第三贡献

**Humanoid Reference Quality Protocol：** 与形态无关的可行性证书（FK 自洽、支撑完整性、摩擦锥近似、力矩裕度、接触抖动）。用 G1 + TK3 两个 embodiment 证明 OmniTrack 浮空定义会误分类。这是 evaluation 论文的种子，单独不够二区，除非做成社区数据集（工作量接近 PHUMA）。

---

## 6. 现有代码到切口 A 还差什么

按「不改就写不成论文」排序。

1. **接上 Stage II。** 用 `rollout_ghost.py` 的物理 npz 训 `TK3-Tracking`，desired-contact 用 Ghost 的 `contact_mask`。没有这条，审稿人只看到数据清洗。
2. **把 `quality_report` 的门搬进训练。** 现在穿地传感器「不进策略、不当奖励」（main 的注释）；原型只加了软 `deep_ground_penetration_cost`。论文需要终止或 Lagrangian，而不是再加一个 -0.1。
3. **保留 q0 作为消融任务，不要删掉。** 原型 commit 写了「移除旧 q0 任务入口」。投稿必须能一行开关 `q0` vs `q_ref`。
4. **Generalist PMG。** 单 clip 只适合调奖励。LAFAN1 规模才和 OmniTrack 可比；至少要「多 clip 一个策略」。
5. **接触丰富数据。** 允许 TCP 接触若只在平地走走上，贡献是空的。需要起身/手撑 clip，并报接触日程，不只报 MPJPE。
6. **和绝对动作 PMG、raw tracking 的同预算对比。** 同一随机种子预算、同一 Stage II MDP。
7. **真机或严格 sim-to-sim。** 名义 Warp 上 quality_passed=true，换接触刚度/延迟后仍通过，才有资格谈「物理」。

不需要为了论文去做的：

- 再精调鞋底凸包（除非消融接触几何对 PMG 的影响）；
- 把 Ghost 做成部署策略（那是 Stage II）；
- 视觉、物体、语言。

---

## 7. 投稿路径（国内「一区/二区」怎么理解）

分区每年微调，以下按 2025–2026 中科院升级版的常见口径，**以学校人事文件为准**：

| venue | 常见口径 | Ghost 优化后 |
|---|---|---|
| Science Robotics | 小类 1 区 | 不建议。需要新能力和 OmniTrack 级以上的真机叙事 |
| IEEE T-RO | 大类 1 区 / 小类 2 区 Top，很多单位算一区 | 切口 B + 真机长时程/遥操作才够；切口 A 偏短 |
| IJRR | 同上 | 更偏完整系统与分析；周期长 |
| IEEE RA-L | 小类 2 区 | **切口 A 的主目标**；可配 ICRA option |
| IEEE RAM | 小类 2 区附近 | I-CTRL 已在此；不宜再投「有界残差模仿」 |
| RSS / CoRL | 顶会，不算 SCI 分区 | 切口 B 更合适；强调问题和实验，不强调期刊分区 |
| ICRA / IROS | 会议 | 切口 A 的保底；一区考核通常不够 |

**时间形态（按技术依赖，不按日历）：** 先做通切口 A 的消融和 Stage II，再决定是 RA-L 还是加码真机冲 TRO。不要先定一区再倒推实验——现有文献已经把「洗参考」的一区窗口关掉了。

---

## 8. 建议的贡献列表（切口 A 投稿时用）

可声称（实验成立的前提下）：

1. 把特权物理运动生成表述为 **有界残差投影**，而不是绝对动作再合成；证明动作中心改变可行性–风格权衡。
2. 用 **训练期硬可行性门 + 归档期质量证书** 替代「只靠软接触惩罚」；提供比根高阈值更物理的悬空定义。
3. 证明投影后的参考能提升下游部分可观跟踪（成功率 / 数据规模稳定性），并在接触丰富动作上优于 raw retarget。

不要声称：

- 首次提出两阶段跟踪或物理一致参考；
- 首次提出残差关节动作或在线物理化；
- 首次从动力学发现手/膝支撑（DSMS）；
- 解决了 sim-to-real；
- 通用具身智能或 VLA。

---

## 9. 一句话决策

Ghost 现在是 **OmniTrack Stage I 在 TK3/MuJoCo 上的高质量复现 + 一套更认真的质量协议 + 一个还没写进论文叙事的 q_ref 残差**。

把它写成二区论文的办法：承认 PMG 框架来自 OmniTrack，把新问题收成 **「如何把 mocap 投影到力矩与支撑可行集，并且投影器只输出相对 q_ref 的有界修正」**；用硬门、接触丰富数据、Stage II 消融证明这不是换奖励。

它能对准的运控难点，按真实性排序：

1. 运动学参考不可行 → 跟踪与稳定互相拆台（主问题，需硬约束化才有新意）；
2. 长时程跟踪误差累积（用参考层残差投影，而不是再做一个部署残差）；
3. 手脚多接触支撑无法从 mocap 直接迁移；
4. 在线脏指令的因果物理化（加码版）；
5. 参考所绑定的仿真物理 ≠ 真机物理（只有上真机/残差仿真器才能碰）。

**下一步若继续做代码：** 不要再调单 clip 奖励。打开 `q0` vs `q_ref` 双入口、把质量门接进终止、用物理 npz 训 Stage II，先打出切口 A 的消融表。没有这张表，论文叙事还是空的。

---

## 来源

- 仓库：`src/tasks/ghost/PROTOTYPE.md`，`tracking_env_cfg.py`，`config/tk3/env_cfgs.py`，`mdp/actions.py`，`rollout.py`，`origin/prototype/ghost-qref-residual`
- OmniTrack: <https://arxiv.org/html/2602.23832v1>
- RobotDancing: <https://arxiv.org/html/2509.20717v1>
- ASAP: <https://arxiv.org/html/2502.01143>
- I-CTRL: <https://arxiv.org/abs/2405.08726>
- PHUMA: <https://arxiv.org/abs/2510.26236>
- OmniRetarget: <https://arxiv.org/html/2509.26633>
- ResMimic: <https://arxiv.org/html/2510.05070>
- BeyondMimic: <https://arxiv.org/html/2508.08241v3>
- KungfuBot: <https://arxiv.org/abs/2506.12851>
- GMR: <https://arxiv.org/abs/2510.02252>
- SuperTrack: Fussell et al., ACM TOG 2021
- PHC: <https://arxiv.org/abs/2305.06456>
- Opt2Skill: <https://arxiv.org/abs/2409.20514>
- DynaRetarget: <https://arxiv.org/abs/2602.06827>
- DSMS: <https://arxiv.org/abs/2608.03116>
- SoftMimic: <https://arxiv.org/abs/2510.17792>
- SPIDER: <https://arxiv.org/abs/2511.09484>
- KDMR: <https://arxiv.org/abs/2603.09956>
