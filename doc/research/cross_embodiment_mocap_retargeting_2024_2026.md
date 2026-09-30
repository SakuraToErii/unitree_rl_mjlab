# 跨本体动捕重定向调研（2024–2026）

范围：近两年顶会顶刊，外加少数与「用幽灵策略把 TK3 动捕转成 G1」直接相关、但尚未核实录用的预印本。截止日期 2026-09-10。只采论文摘要、项目页、会议录用页，不采博客转述。

## 结论

可以，但文献里几乎没人把这件事做成「源机器人关节角 → 目标机器人关节角」的端到端策略。近两年主流是三条分开的线：

1. **运动数据重定向**：把源骨架（人 / 另一台机器人）映射成目标机器人的运动学参考。工业默认仍是约束优化（GMR、OmniRetarget、PyRoki），不是学习策略。
2. **物理可行性投影**：用特权 RL 跟踪器在仿真里把不可行参考滚成可行轨迹。这就是当前 `prototype/ghost-qref-residual` 在做的事，对应 OmniTrack Stage I、H2O 的 sim-to-data、NMR 的 CEPR。
3. **跟踪策略跨本体迁移**：已经有一台机器人的 whole-body tracker，用对齐 + LoRA 迁到另一台（Any2Any）。迁的是控制器，不是动捕文件。

TK3 → G1 最贴近文献、也最贴近现有 ghost 代码的做法是：

- 先把 TK3 表示成**关键点 / 连杆位姿**（不要关节对关节拷贝）；
- 用 GMR 或 OmniRetarget 做一次运动学重定向，得到 G1 参考；
- 再在 **G1** 上跑现有幽灵策略，把尺度差、穿模、漂浮当「糟糕动捕」修掉。

「让幽灵策略自己学会跨本体转换」在文献里最接近的是 NMR 的 CEPR：先用分簇 RL 专家把噪声演示投影到目标机器人可行流形，再用这些成对数据监督一个神经网络重定向器。那是人→G1，不是 TK3→G1，但训练配方可以直接搬。

若 TK3 和 G1 的关节语义对不齐、不想手写 body map，预印本 [UMR](https://arxiv.org/abs/2609.02134) 用稠密点云对应代替稀疏关键点；[G-DReaM](https://arxiv.org/abs/2505.20857) 用图条件扩散做异构骨架转换，作者自称是机器人里较早的统一跨本体重定向。

## 先把问题拆开

当前 ghost 分支的问题是：**同一台 TK3** 上，特权 PPO 把一条嘈杂 100 Hz 动捕滚成力矩受限、无硬限位、无深穿地的 MuJoCo 轨迹（见 `src/tasks/ghost/PROTOTYPE.md`，配方来自 OmniTrack Stage I）。

跨本体转换是另一个问题。源和目标差的是：

- 拓扑与关节数（不能 `q_tk3[i] → q_g1[i]`）；
- 肢长、骨盆高度、脚底板形状；
- 质量、惯量、执行器，所以运动学上对得上，动力学上不一定跟得住。

文献几乎都在解 **人 → 人形机器人**。真正的 **机器人 A → 机器人 B** 论文很少；OmniRetarget 用同一套交互网格把一条演示扩到 G1 / H1 / T1，Any2Any 把 G1 上的 Sonic 跟踪策略迁到 LimX。没有找到「TK3 动捕文件直接变成 G1 动捕文件」的顶会论文。

## 已核实顶会 / 顶刊

| 论文 | 会议 | 转什么 | 方法 | 和 TK3→G1 ghost 的关系 |
| --- | --- | --- | --- | --- |
| [H2O](https://human2humanoid.com/resources/H2O_paper.pdf) | IROS 2024 | 人(SMPL/AMASS) → Unitree H1 | SMPL 拟合机器人外形 + IK；特权模仿器 **过滤** 不可行动作（sim-to-data），再训可部署跟踪器 | 幽灵的近亲，但是筛选器不是转换器；本体仍是人→一台机器人 |
| [ExBody](https://www.roboticsproceedings.org/rss20/p107.pdf) | RSS 2024 | CMU MoCap → H1 | 上身跟动作、下身跟速度，故意不全身关节对齐 | 说明跨形态时「全身关节拷贝」会失败；不是数据转换 |
| [HumanPlus](https://proceedings.mlr.press/v270/fu25a.html) | CoRL 2024 | 40h 人体动捕 → 定制 33-DoF 人形 | RL 低层 shadowing + 真机 BC | 跟踪，不是重定向方法论文 |
| [OmniH2O](https://omni.human2humanoid.com/) | CoRL 2024 | 大规模人体动作 → 带灵巧手的全身遥操 | 大规模重定向 + 增强；特权教师蒸馏稀疏观测学生 | 教师–学生结构和 ghost→deploy 类似；重定向本身仍是离线运动学 |
| [HOVER](https://hover-versatile-humanoid.github.io/) | ICRA 2025 | 多种控制接口（H2O / OmniH2O / ExBody / HumanPlus 模式）→ 一个策略 | 全身运动学模仿当公共抽象，多模式蒸馏 | 接口是「运动学 pose」，不是「另一台机器人的 q」 |
| [ASAP](https://agile.human2humanoid.com/) | RSS 2025 | 已重定向人体动作 → 真机 G1 | 仿真预训练跟踪 + 真机残差动作模型对齐动力学 | 残差修的是 sim2real，不是跨本体 |
| [AMO](https://www.roboticsproceedings.org/rss21/p061.html) | RSS 2025 | VR 关键点 → IK → 29-DoF G1 | 在线 IK 全身遥操 | 遥操作接口，不是动捕库转换 |
| [TWIST](https://proceedings.mlr.press/v305/ze25a.html) | CoRL 2025 Best Paper | OptiTrack 人 → G1；论文里也迁到 Booster T1 | 运动学重定向后 RL+BC 全身跟踪 | 跨机器人只验证了 **跟踪策略** 可迁，动捕仍先重定向到目标机 |
| [VideoMimic](https://www.videomimic.net/) | CoRL 2025 Best Student Paper | 单目视频里的人+场景 → G1 | 4D 重建 + 软约束重定向 + 场景条件跟踪 | 人→机器人；重定向带地形，但是优化不是学习映射 |
| [KungfuBot](https://kungfubot.github.io/) | NeurIPS 2025 | 视频/动捕 → G1 高动态动作 | 物理量过滤 + 接触修正 + 微分 IK；自适应跟踪容差 | 重定向是 IK；物理修正在源人体模型上做，再映射到 G1 |
| [OmniRetarget](https://omniretarget.github.io/) | ICRA 2026 Best Paper（Manipulation and Locomotion） | 人/OMOMO/LAFAN1 → G1、H1、T1 | 交互网格 Laplacian 变形 + 硬约束（不穿模、脚粘地、关节限位） | **最该先读的跨本体数据论文**。一条演示可扩到不同机器人，但是优化不是策略 |
| [PyRoki](https://chungmin99.github.io/pyroki/) | IROS 2025 | SMPL 关键点 → G1 / H1 | 轨迹级运动学优化（相对关节、接触、自碰） | 开源工具箱；同一套 cost 能打 G1（127 cm）和 H1（178 cm） |
| [MaskedMimic](https://research.nvidia.com/labs/par/publication/maskedmimic.html) | SIGGRAPH Asia 2024 / TOG | 残缺运动描述 → 物理角色 | 掩码运动补全，物理跟踪 | 图形角色，不是真机；「不完整/错位参考」可被策略补全，和幽灵修噪声同构 |
| [TokenHSI](https://openaccess.thecvf.com/content/CVPR2025/papers/Pan_TokenHSI_Unified_Synthesis_of_Physical_Human-Scene_Interactions_through_Task_Tokenization_CVPR_2025_paper.pdf) | CVPR 2025 Oral | 多任务人体–场景交互 | Transformer 任务 token | 图形；跨任务不是跨机器人 |
| [CLoSD](https://proceedings.iclr.cc/paper_files/paper/2025/file/7319b7561ffe5e2f6419acd4a2f52d6b-Paper-Conference.pdf) | ICLR 2025 | 文本扩散规划 + PHC 跟踪 | 闭环扩散 | 图形；底层跟踪器是 PHC |
| [RoboGhost / From Language to Locomotion](https://arxiv.org/abs/2510.14952) | ICLR 2026 | 文本潜变量 → G1 动作 | MoE 教师 + 扩散学生，**推理时不做显式重定向** | 名字像 ghost，实际是跳过关节参考，直接从 motion latent 出动作。不产生 G1 动捕文件 |

H2O 的 sim-to-data 原文（IROS 2024）：

> To create a large-scale retargeted motion dataset of human movements for humanoid robots, we propose a scalable "sim-to-data" process to filter and pick feasible motions using a privileged motion imitator.

这是「特权策略修/筛参考」在顶会上的源头。它筛的是人→H1 之后还能不能跟，不会把 TK3 的 `q` 翻译成 G1 的 `q`。

## 和「用幽灵学会转换」最近的工作（多为预印本）

下面几篇方法上最贴，但 **2026-09-10 未能从会议页核实录用**。当方法参考，不当「已发表顶会」引用。

### 1. NMR：幽灵专家生成成对数据，再学神经网络重定向

[Make Tracking Easy: Neural Motion Retargeting](https://arxiv.org/abs/2603.22201)（Zhao et al., 2026, 南大）。人（含噪声 SMPL）→ G1。

核心是 CEPR：VAE 把动作分簇，每簇训并行 RL 专家，在仿真里把噪声演示 **投影到机器人可行流形**，得到约 3 万对人–机器人对；再用 CNN-Transformer 学这个映射，推理不再跑仿真。论文明确写 GMR 这类优化非凸、容易跳关节和自碰；CEPR 修完后自碰帧降 54%、限位违规降 61%。局限也写了：CEPR **绑定形态**，换平台要重跑数据管线。

这就是「幽灵策略当数据生成器，再蒸馏成转换网络」。源是人不是 TK3，但配方可直接用：TK3 当源、G1 当目标专家。

### 2. OmniTrack：当前 ghost 原型的出处

[OmniTrack](https://arxiv.org/abs/2602.23832)（Li et al., 2026）。Stage I 特权通才策略在仿真里滚出物理一致参考；Stage II 再训可部署跟踪器。和 `PROTOTYPE.md` 一致。它修的是 **已经重定向到目标机之后** 的漂浮/穿模，不负责换本体。

跨本体时它的正确用法：先运动学转到 G1，再 G1-OmniTrack/ghost 做 Stage I。

### 3. IKMR：共享拓扑潜空间 + 仿真跟踪微调

[Implicit Kinodynamic Motion Retargeting](https://arxiv.org/abs/2509.15443)（Chen et al., 2025）。图卷积双自编码器把人和人形机器人收到同一拓扑潜空间，再用仿真跟踪反馈微调解码器。推理 >5000 fps，并声称能抑制源数据抖动。真机验证在 G1。和 NMR 同族：学习映射，物理先验来自仿真跟踪，不是在线 IK。

### 4. AdaMorph：一个网络打 12 台人形

[AdaMorph](https://arxiv.org/abs/2601.07284)（Zhang et al., 2026）。人运动进形态无关 intent，机器人用 AdaLN / prompt 条件解码。声称 12 台异构人形、无需每台单独训。这是「一个学习器做多本体运动学重定向」最接近的陈述。**未核实会议；通讯邮箱在 HTML 版是占位符 `teacher@example.com`，引用前需要再核原文。** 而且它做的是运动学生成，不是物理跟踪。

### 5. UMR：学稠密点云对应，不手写关节 map

[Unified Motion Retargeting](https://arxiv.org/abs/2609.02134)（Cao et al., 2026, HKUST(GZ) / Noitom）。把人体和机器人外表面点云当统一接口，在规范姿态上学稠密对应，再做带约束的点云匹配；接触图随表面一起迁。源可以是 SMPL-X / LAFAN1 / SOMA 等 mesh，目标评测以 G1 为主，对照 GMR 和 OmniRetarget。作者写现有方法依赖手工稀疏关键点，换机器人就要重做 body map。

对 TK3→G1 的实际含义：如果两台机拓扑不完全对齐，UMR 比「列一张 TK3 link → G1 link 表」更可扩展。仍是「学对应 + 优化」，不是端到端策略。未核实会议。

### 6. G-DReaM：图条件扩散，异构骨架统一重定向

[G-DReaM](https://arxiv.org/abs/2505.20857)（Cao et al., 2025）。骨架做成图，扩散模型按图条件生成目标运动；缺配对数据时用重定向损失做 energy guidance；可低成本适配未见骨架，实验含非同胚（人形和下半身 biped）。作者称这是机器人里较早的统一跨本体重定向。质量和真机管线弱于 GMR 系。未核实会议。

### 7. Human2Humanoid（CycleGAN，无成对数据）

[Human2Humanoid](https://arxiv.org/abs/2606.03476)（Huang et al., 2026, 中科大）。无配对 CycleGAN + 骨架感知 GCN + T-pose 归一化末端一致 + 接触约束。人→G1。若 TK3 和 G1 各有大量 **不成对** 动作库、不想先跑 IK，这条路成立。未核实会议。注意标题容易和 IROS 2024 的 H2O（也叫 Human-to-Humanoid）撞名。

### 8. Any2Any：迁跟踪策略，不迁动捕

[Any2Any](https://arxiv.org/abs/2605.23733)（Yang et al., LimX, 2026）。先对齐观测/动作空间，再只给动力学敏感模块加 LoRA。6 组迁移：Oli ↔ G1 / H1 / Luna，以及 Sonic（G1 预训练）→ Oli / Luna / H1；约 1% 从头训练算力。作者自称是人形 WBT 跨本体迁移的系统研究。

若目标是「G1 已经会跟动作，想让 TK3 也跟同一批参考」，走这条。若目标是「TK3 动捕文件变成 G1 npz」，这条不解。

### 9. GMR / Retargeting Matters：把重定向质量从跟踪里拆出来

[Retargeting Matters](https://arxiv.org/abs/2510.02252)（Araújo, Ze, Xu, Wu, Liu, 2025）。Stanford。非均匀局部缩放 + 两阶段 IK（实现基于 Mink）。用 BeyondMimic、固定奖励，对比 PHC、ProtoMotions、Unitree 闭源 LAFAN1。结论：漂浮、自碰、速度尖峰会直接打崩下游 RL；GMR 接近 Unitree 闭源。代码：https://github.com/YanjieZe/GMR。仓库 README 标注 `[ICRA 2026]`，arxiv bibtex 仍是 preprint，IEEE 论文页未核到。

这是现在开源人形重定向的默认基线。OmniRetarget、PHUMA、NMR、SONIC、UMR 都拿它当对照。GMR 本身已支持多款目标机（G1 / H1 / T1 / N1 等），输入是人的 SMPL-X 或 BVH，不是机器人 A 的 `q`。

### 10. PHUMA：带物理约束的大规模重定向数据集

[PHUMA](https://arxiv.org/abs/2510.26236)（Lee et al., KAIST, 2025）。PhySINK = SINK（H2O/HOVER 那类外形自适应 IK）+ 限位 + 着地 + 防滑。73 小时，G1 和 H1-2 上都训过 MaskedMimic，真机 G1 用 BeyondMimic。价值是：**重定向器必须显式处理接触，视频动捕尤其如此。**

### 11. KDMR / DynaRetarget：优化器当「动力学幽灵」

[KDMR](https://arxiv.org/abs/2603.09956)（Zhang et al., 2026）把重定向做成多接触全身轨迹优化，加刚体动力学和接触互补；有地面反力时还能自动判 heel-toe。对照 GMR，下游 BeyondMimic。

[DynaRetarget](https://arxiv.org/abs/2602.06827)（Dhedin et al., 2026）用采样轨迹优化把不完美运动学轨迹修成动力学可行，增量地平，做人形–物体演示。和幽灵同目标（修不可行参考），但是优化器不是 RL 策略。

两篇都是 arXiv only。

## 跟踪侧（数据已经在目标机上之后）

这些论文不解决「TK3 文件怎么变成 G1 文件」，但决定 ghost 滚完之后 Stage II 怎么训。

| 论文 | 状态 | 要点 |
| --- | --- | --- |
| [GMT](https://arxiv.org/abs/2506.14770) | arXiv 2025，项目页未写会议 | Adaptive Sampling + Motion MoE，单一策略跟多样动作 |
| [BeyondMimic](https://arxiv.org/abs/2508.08241) | arXiv 2025；后续论文当 SOTA 跟踪器引用，PDF 首页未写会议 | 最少奖励；**参考必须干净**。GMR / OmniRetarget 都用它当下游探针 |
| [GMT 的残差后续 ResMimic](https://arxiv.org/abs/2510.05070) | arXiv 2025 | GMT 当先验，残差策略修物体交互；人动作用 GMR 转到机器人 |
| [Any2Track](https://arxiv.org/abs/2509.13833) | arXiv 2025 | 通才跟踪 + 历史自适应，抗扰动 |
| [UniTracker](https://arxiv.org/abs/2507.07356) | arXiv 2025 | AMASS→G1，CVAE 学生 |
| [SONIC](https://nvlabs.github.io/GEAR-SONIC/) | NVIDIA 2026 技术报告 / 开源栈 | 用 GMR + PyRoki 把 ~700h 动捕转到 G1；把跟踪做成 foundation task |
| [VENOM](https://arxiv.org/abs/2606.16696) | arXiv 2026 | 一个 GPT 跟踪器打 G1 / H1 / H1-2 / N1 / T1，靠多机器人数据而不是运行时重定向 |
| [ExBody2](https://arxiv.org/abs/2412.13196) | arXiv 2024；有 workshop 引用，不是正会 | 特权教师生成更符合机器人运动学的中间数据并滤掉不可行全身动作；学生 DAgger 上真机。和 H2O sim-to-data 同类 |

图形侧的 PHC（ICCV 2023）和 ProtoMotions 仓库仍是 H2O/HOVER/GMR 的 IK 基线，但已超出「近两年」窗口，这里不展开。

## 对 prototype/ghost-qref-residual 的含义

现有幽灵是 **同本体物理投影**。文献支持三种加码方式，由易到难：

**A. 优化重定向 + 目标机幽灵（最稳，也是顶会默认）**

1. TK3 动作提到共享几何：骨盆、左右脚、左右手、头、若干肢体中点。
2. GMR 或 OmniRetarget 解到 G1。OmniRetarget 适合有接触/场景；平地全身动作用 GMR 即可。
3. 在 G1 上复用现有 ghost（OmniTrack Stage I），把 IK 残渣当噪声修掉。
4. Stage II 用 BeyondMimic 那种少奖励跟踪。

这不是「幽灵学会转换」，是「转换仍用优化，幽灵继续修物理」。OmniRetarget 已经证明同一套交互网格能扩到 G1/H1/T1，只改关键点对应和碰撞模型。

**B. 幽灵当教师，蒸馏跨本体网络（NMR/IKMR）**

1. 用 A 或粗糙缩放关键点，在 G1 上训分簇 RL 专家（就是现在的 ghost，动作改成跟 TK3 关键点）。
2. 把 `(tk3_motion, g1_rollout)` 存成对。
3. 训一个非自回归网络，推理时不再开仿真。

这才是「学会转换」。NMR 写明换本体要重跑 CEPR。TK3 和 G1 都是双足人形，比人→机器人容易，但仍要自己做对应和专家。

**C. 学对应 / 无配对映射（UMR、G-DReaM、CycleGAN）**

- 关节语义对不齐：UMR 的点云对应，或 G-DReaM 的图条件扩散。
- 已有大量不成对 TK3 库和 G1 库、不想维护 IK：Human2Humanoid CycleGAN。

顶会里还没有与 C 同等设定的录用论文。物理可行性通常弱于 B。

不建议一上来就做 Any2Any 式策略迁移：那假设目标机已经有可跟的参考。现在缺的是参考本身。

## 建议阅读顺序

1. OmniRetarget（ICRA 2026）— 跨本体数据增强、硬约束、交互网格。
2. Retargeting Matters / GMR — 开源基线，以及「参考质量如何打崩 RL」。
3. OmniTrack — 和当前 ghost 对齐。
4. NMR — 唯一把「RL 专家投影 → 神经网络重定向」写完整的配方。
5. UMR — 不想手写 TK3↔G1 关节表时再读。
6. H2O（IROS 2024）— sim-to-data 的原始表述。
7. PyRoki（IROS 2025）— 若要自己写轨迹优化而不是调 GMR。
8. Any2Any — 只有在已经有 G1 跟踪器、还想迁到别的机时再读。

## 来源与不确定性

- 会议以项目页或 IEEE/PMLR/RSS 论文页为准。OmniRetarget 的 ICRA 2026 Best Paper 来自 https://omniretarget.github.io/ ；TWIST Best Paper 来自 CoRL PMLR v305；VideoMimic Best Student Paper 来自 https://www.videomimic.net/ 。
- GMT、BeyondMimic、PHUMA、NMR、AdaMorph、Any2Any、IKMR、OmniTrack、Human2Humanoid（2026 CycleGAN）、VENOM、SONIC、UMR、G-DReaM、KDMR、DynaRetarget、ExBody2：检索时只有 arXiv / 实验室页，**未当作已录用顶会**。
- GMR：https://github.com/YanjieZe/GMR 标题写 `[ICRA 2026]`，arxiv bibtex 仍为 preprint；在 IEEE 论文页核到之前不当正式录用。
- AdaMorph HTML 通讯邮箱异常，方法描述仅作方向参考。
- 没有找到 2024–2026 顶会论文直接做 Tiangong / TK3 → G1。最近的机器人–机器人证据是 OmniRetarget 的 G1/H1/T1 数据增强，以及 Any2Any / TWIST 的策略迁移。
- Mink（GMR 依赖的 MuJoCo IK 库）是工具不是论文。ProtoMotions 是 NVIDIA 的仿真/重定向仓库，不是会议论文。
)
