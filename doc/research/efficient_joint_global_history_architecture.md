# 关节交叉注意力与低成本全身历史表示

调研日期：2026-10-05。对象：本仓库 G1 0-PD actor，5 帧、每帧 63 维；IMU、25 个可见关节位置、29 个上一动作、3 维命令，隐藏 4 个踝位置并且没有关节速度。本文仅研究架构，不修改模型或训练脚本。

**结合 GPU 微基准，第一优先是保留原结构，增加一个小全身历史 MLP（方案 D），而非再增加时间 MHA。** 它把过去 4 帧完整观测映射为 64 维向量，再拼入最终动作 MLP，本地 Actor 前向加反向仅增加约 1–2% 时间。如果还需要明显提速，优先测试 4 个全身 KV、关节局部历史 residual、body Query 和共享小 head 的紧凑结构（方案 B+C 的变体）；微基准快约 27–29%，但尚未验证 PPO 奖励。

另一个较保守的注意力方案是保留 29 个关节历史 token，把独立的 IMU token 换成 4 个全身历史帧 token（方案 A）。它补充全身同步 memory，静态 MAC 仅增加约 3%，但实测 Actor 前向加反向慢约 6–7%；与共享 head 配合也未取得明显加速。本文保留这个候选作为表示消融，不把它推荐为速度优化。

这是依据本地代码和一手文献提出的候选设计，不是已经验证的 G1 训练结果。当前“前期较快、后期线速度奖励较低”的现象也没有匹配的远端旧实验记录，不能据此确认架构就是唯一原因。另一台服务器 FP32 出现相同趋势，支持优先检查表示和优化，但不构成因果证明。

## 1. 为什么不优先继续叠一个时间 Transformer

当前实现位于 [`src/tasks/effort/rl/models.py`](../../src/tasks/effort/rl/models.py)：

- `_JointCrossAttention` 为每个关节构造一个 Query：当前自身位置、上一动作、joint embedding、missing flag，再加入当前 IMU / 重力 / 命令的编码。
- 每个关节的过去 4 帧位置和动作拼接后，经一个**跨关节共享**的 `history_encoder`，生成 29 个历史 token；4 帧 IMU 单独生成第 30 个 token。
- `_JointAttentionBlock` 对 29 个 Query 做 cross-attention，随后每个 Query 都经过 `d → 2d → d` FFN。
- 输出投影成每关节 32 维，29 个输出 flatten 后与当前完整帧编码拼接，再经过 actor MLP 输出 29 个动作。

因此，现有模型已经有共享 encoder。共享参数减少了参数数量，但同一网络仍然要对 29 个关节执行，不能把“共享一份权重”理解成“只算一次”。从计算上看，29 个 Query 的 FFN、Q / K / V / output 投影，以及 `29 × 32` 特征进入大 MLP，都值得关注。

设关节数为 $J$，历史 memory 数为 $M$，宽度为 $d$。一层常规 cross-attention 的主要乘加数约为：

\[
\underbrace{(2J+2M)d^2}_{Q/K/V/output\ 投影}
+\underbrace{2JMd}_{QK^T\ 与\ AV}
+\underbrace{4Jd^2}_{d\to2d\to d\ FFN}.
\]

多头数改变的是切分方式；在总宽度 $d$ 固定时，不会把上述总投影和点积乘加数除以 head 数。softmax、LayerNorm、激活和张量搬运不在这个近似公式里。

本地小配置 $J=29,M=30,d=64$ 的静态核算中：Q / K / V / output 投影约 483k MAC，QK + AV 约 111k MAC，关节 FFN 约 475k MAC，flatten 后的动作 MLP 约 297k MAC。注意力点积只占 actor 主要乘加的一小部分，约 7%。这些是**静态计算量，不是实测耗时**；把 KV 数从 30 减到 4，不能推出整轮训练快 7.5 倍。

PPO 每轮是 8192 环境 × 24 步，再执行 5 epochs × 4 minibatches；每个 minibatch 49152 样本。整体迭代还包含仿真、critic、优势计算、优化器和日志等。应分别看 rollout 和 update 时间，不能直接把近 3 秒全部归到 attention。

## 2. 方案 A：混合 memory，优先保留局部历史

这是最保守的能力补充。现有的 29 个关节历史 token 继续生成、继续作为 K/V；历史 IMU 不再单独压成一个 token，而是融入 4 个完整全身历史帧。当前 `_FrameEncoder` 复用于当前和过去 4 帧，每帧执行一次相同映射：

\[
g_s=E(o_s),\qquad G=[g_{t-4}+p_0,\ldots,g_{t-1}+p_3].
\]

新的 memory 是 `concat(29个关节历史token, 4个全身帧token)`，数量由 30 变成 33。Q 来源、每关节 FFN、当前完整帧旁路和最终动作 MLP 全部保持原设计。这里新增的是 KV 中的信息源，**没有增加第二层 attention，没有增加时间 self-attention**。时间位置 embedding 必须保留，否则逐帧共享 encoder 无法让 attention 区分历史方向。

```text
29个关节四帧历史 → 原共享关节encoder → 29个local KV ─┐
过去4帧完整观测 → 共享当前帧encoder+时间embedding → 4个global KV ─┤
                                                        同一层cross-attn
当前自身q/u + joint ID + missing + 当前IMU/命令 → 29个Q ─┘
```

这一方案的优点是兼顾已有局部历史读取与新增全身同步历史，而非让它们二选一。每个全身帧里有所有可见关节位置、29 个上一动作、IMU / 重力和命令，4 个踝位置仍不可见；框架不会凭空恢复缺失的真实关节角。

以 d=64 计算，额外 4 帧投影约 16,128 MAC；增加 3 个 memory 的 K/V 投影约 24,576 MAC；QK + AV 增加约 11,136 MAC；删除旧 IMU encoder 省约 1,536 MAC。合计增加约 50,304 MAC，占原 actor 主要乘加约 **3.3%**。这是忽略激活、归一化、搬运后的静态估算，**不等于整轮时间增加3.3%**；实际需微基准和完整迭代计时。

局部和全身 token 仍竞争同一个 softmax。4 个全身 token 比原来单一 IMU 摘要提供更细的时间与全身信息，但是否得到足够利用需要检查和消融。若全身 token 长期权重很低，可单独测试 type embedding 或 body Query；不建议先加多级注意力或硬规定权重配额。

方案 A 的目的主要是以小成本补能力，不能期待它自身显著提速。若质量改善，后续缩小 FFN / head 时也可以继续使用这个混合 memory；不必强制放弃局部 token。

## 3. 方案 B：共享全身帧 encoder，替换 KV

适合先检查“缺少全身同步历史”的假设；保持现有 Query、FFN、输出维度和动作 MLP，降低其余变更的影响。

对当前帧和历史帧用同一个小 encoder：

\[
g_s=E(o_s),\qquad E:\mathbb{R}^{63}\to\mathbb{R}^{64}.
\]

`E` 可以先用 `Linear(63,64) → ELU → LayerNorm`，与现有 `_FrameEncoder` 规模一致；暂时没有必要新增 128 维、多层时间 MHA。共享的是每帧的映射参数，每个样本仍执行 5 个帧向量的映射，可合并为一批矩阵乘法。

历史 memory 改成：

\[
G=[g_{t-4}+p_0,\ g_{t-3}+p_1,\ g_{t-2}+p_2,\ g_{t-1}+p_3]
\in\mathbb{R}^{4\times d}.
\]

`p` 是 4 个时间位置 embedding。**不能省掉时序信息**：共享逐帧 encoder 加无位置的 attention 对 memory 的排列不敏感，会妨碍按时间方向区分位置变化。这里所有帧都在当前时刻之前，没有未来观测，不需要为了防止泄漏另加因果掩码。

| 部件 | 当前架构 | 方案 B |
|---|---|---|
| Q 来源 | 自身当前位置 / 动作、joint ID、missing、当前 IMU / 命令 | 先保持相同 |
| K 来源 | 29 个关节历史摘要 + 1 个 IMU 历史摘要，分别投影 | 4 个完整全身历史帧编码，分别投影 |
| V 来源 | 同上，经独立 V 投影 | 同上，经独立 V 投影 |
| 当前观测旁路 | 当前完整帧 encoder → 最终 MLP | 保留，可与历史帧 encoder 共享权重 |
| Query FFN 和 actor MLP | 29 个宽 FFN + flatten 大 MLP | 先保留 |

```text
过去4帧完整观测 ── 共享帧encoder + 时间embedding ── 4个KV
                                                       ↑
当前自身q/u + joint ID + missing + 当前IMU/命令 ── 29个Q
                                                       ↓
                              现有cross-attn + FFN → 关节输出
当前完整帧 ── 同一个帧encoder ────────────────────→ 拼接 → 现有MLP
```

一个历史 token 现在同时包含该时刻的髋、膝、所有可见关节、上一动作和 IMU。关节 Query 读取一个帧时，可以直接取得同一时刻的全身特征，不再必须从独立关节摘要和 IMU 摘要拼出同步关系。这个收益是**结构上的推断**，是否改善线速度仍需训练验证。

此方案保留了关节当前反馈、关节身份和每关节输出槽位，但不再保留“每个关节独立的四帧历史 memory”。如果希望连局部时间反馈也保留，可把原有共享 `history_encoder` 的结果改接到对应关节的 Query residual：

\[
r_j=L([q_{t-4:t-1,j},u_{t-4:t-1,j},e_j,m_j]),
\qquad q_j\leftarrow q_j+r_j.
\]

它不需要额外 attention，并且只读取自身历史，但也增加执行量和一个变量。建议将它作为方案 B 之后的单独消融。隐藏踝的位置仍为零，必须保留 missing flag，不能让模型把“未观测”误当成“测得零”。

替换后点积项由 (2J\times30\times d) 降为 (2J\times4\times d)，K/V 投影和 memory 激活也减少；29 个 Query 投影、FFN 和大动作 MLP 仍然存在。四帧全身 memory 不经过历史 self-attention，只经过共享逐帧 encoder，因此比“保留原分支再并行加时间 MHA”更简单。

## 4. 方案 C：同一层读出全身向量，配共享小动作 head

适合方案 A 或 B 的训练质量可以接受，但 update 仍然较慢时。它是计算与表示一起重构的候选，需要分步消融。

以下以方案 B 的 4 个全身历史 KV 展示；也可以保留方案 A 的混合 memory，单独测试 head 精简。在 29 个关节 Query 之外，加 1 个全身 Query，**合成 30 个 Query，只调用同一层 attention 一次**：

\[
q_j=Q_{local}([q_{t,j},u_{t-1,j},e_j,m_j])+g_t,\quad
q_{body}=g_t+e_{body}.
\]

这里 $g_t=E(o_t)$。与现有 Query 只加 IMU / 命令相比，每个关节 Query 现在还依据当前全身姿态选择历史；隐藏踝的 Query 因而能直接受到当前髋、膝等观测影响。

\[
[z_1,\ldots,z_{29},z_{body}]
=Q_{input}+\operatorname{MHA}(Q_{input},G,G).
\]

保留必要的 LayerNorm 和 residual，但不立即沿用 30 个 `d → 2d → d` FFN。输出改为共享的小 head：

\[
\tau_j=h([z_j,z_{body}]),\qquad h:\mathbb{R}^{2d}\to\mathbb{R}^{64}\to\mathbb{R}.
\]

joint ID 和 missing flag 已进入 $z_j$ 的 residual，使共享 head 可以学习关节差异。可按需要把 $g_t$ 加到 head 输入或改为 residual；先选择一种接法固定测试，避免反复增加拼接宽度。

```text
5帧完整观测 → 共享帧encoder ───── 过去4帧+时间embedding → KV
                      │                                ↑
                      ├─ 当前全身编码 ── 1个body Q ──────┤
自身当前q/u + ID/缺失 ─┴─ 29个joint Q ──────────────────┤
                                             单层cross-attn
                                              │         │
                                           z_joint    z_body
                                              └────┬────┘
                                               共享小head
                                                   ↓
                                              29个力矩输出
```

相比方案 B，多一个 Query 只把点积项从 $29\times4$ 增为 $30\times4$；同时所有动作都得到一个显式的 $z_{body}$。减少关节 FFN 和 flatten 大 head 会降低算术量，但实测表明保留 33 个混合 KV 时，这些精简几乎没有提高速度。KV 激活量、归一化、张量复制和 backend 同样影响耗时；不能从 FFN 或参数数量直接推断收益。共享 head 仍然执行 29 次，它应足够小，否则会把省下的计算重新花掉。

对“保留现在的优点”的准确说法是：保留局部 Query、ID、missing、residual 和关节动作结构，**尝试保留**其局部控制归纳偏置。不能保证保留当前的收敛优势。去掉 flatten 后，各动作之间的协调主要依赖共享的全身 KV 和 $z_{body}$，而非一个任意混合所有关节特征的大 MLP；表示容量和协调方式发生变化，可能降低，也可能改善最终奖励。

如果共享小 head 的容量不足，可以先留小 FFN，例如 `d → d → d`，或给 body 向量一个全局 MLP，生成 29 维全局动作，再加共享局部 head 的动作 residual。它们是容量回补方案，应在共享 head 版本出现明确质量下降后再测试。

## 5. 方案 D：小全身历史 MLP，保留原结构

这是最适合先验证表示假设的低侵入候选。过去四帧完整观测按既定顺序展平：

\[
h_g=\operatorname{MLP}_{252\to128\to64}([o_{t-4},o_{t-3},o_{t-2},o_{t-1}]),
\qquad \tau=\operatorname{Head}([latent_{original},h_g]).
\]

采用 ELU，可在最后加 LayerNorm；保留现有 29 个关节 Query、30 个历史 memory、关节 FFN、动作 head 的隐藏层尺寸。首轮只让动作 head 多接收 64 个输入维度，**不修改原 attention 的 Q/K/V**。固定顺序的展平输入用不同输入列表示不同时间，因此不需要额外时间位置 embedding。

新增主要乘加约 `252×128 + 128×64 + 64×256 = 56,832 MAC`（d=64、本地 head）；对原约 1.51M MAC 仅增加 3.75%。本地微基准前向加反向从 112.11 ms 增至 113.95 ms，约 1.6%；这个差值接近微基准波动量级，准确结论是新增开销很小，而非承诺整轮只慢固定比例。独立 128 维时间 MHA 分支则需约 150.74 ms。

如果希望共享 encoder，可复用现有 `_FrameEncoder` 对当前及过去四帧编码：`frames=E(sequence)`，然后 `h_g=MLP(flatten(frames[:,:-1]))`；当前 `frames[:,-1]` 继续进入原动作 head。这样当前与历史在同一个表示空间，但共享参数不会消除四帧的额外执行。d=64 下每帧原始宽度 63 与编码宽度 64 接近，共享并不天然更省乘加；本地该版本需 115.56 ms。选择共享应基于表示和训练结果，而非“shared 一定更快”。

首轮同时保留本关节历史和全身历史，最容易检查“全身表示是否确实改善线速度”。若有效，再单独测试 `Q_j += W_g h_g`，让历史读取依据全身历史；这与仅拼入动作 head 是不同变量。小 MLP 并不保证足以恢复缺失踝或接触状态；HIM / DreamWaQ 的观测更完整，见下节的限制。

## 6. 一手文献：哪些思想可以迁移

以下是候选排序；“适用判断”结合本仓库的 4 帧 / 30 token / PPO 场景，是本文推断。论文均未直接验证“隐藏 4 个踝、无关节速度的 G1 0-PD 控制”。

| 候选 | 一手来源的实际创新 / 目的 | 在这里能借用什么 | 这里的限制与优先级 |
|---|---|---|---|
| 共享全身帧编码 + 关节 Query 读 memory | 本文结合代码提出；output-specific Query 读取共享 latent 的思想可参考 [Perceiver IO §3.2](https://arxiv.org/html/2107.14795v3#S3.SS2) | 4 帧只编码一遍；29 个关节共享这些 KV；可与原29 local KV混合，或直接替换；加 1 个 body Query 显式读出全身历史 | **优先**。混合memory补能力成本小；仅4帧memory更省，但会替换局部历史token；二者都不是完整Perceiver IO |
| 短历史 MLP estimator | [HIM §3.2](https://junfeng-long.github.io/HIMLoco/static/pdfs/HIMLocomotion.pdf)、[DreamWaQ §II-C](https://arxiv.org/html/2301.10602v2#S2.SS3) 从本体历史估计速度和隐变量，不依赖时间 Transformer | 一个小 `Flatten(4×63) → 128 → 64` 可提供全身历史向量；可给 body 向量加显式速度监督 | **重要低成本对照**。HIM 使用更宽的 512/256/128 extractor，不能照搬为“极小 MLP 足够”；原论文含关节速度，观测比这里完整 |
| Set Transformer ISAB / PMA | [原论文 §3.1–3.2](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf)：用少量 inducing points，将 set 自注意力 $O(n^2)$ 改成 $O(nm)$；PMA 用少量 seed Query 聚合 | 保留原 30 个 memory 时，可先用 2–4 个 learned latent 读取 memory，关节再读取 latent；body Query 与其 pooling 思想接近 | **次选**。完整 ISAB 有两次 attention；目前30 token 很短，额外层和投影可能抵消收益。时间和关节身份需要显式编码，不能直接当无序集合 |
| Perceiver / Perceiver IO bottleneck | [Perceiver IO §3](https://arxiv.org/html/2107.14795v3#S3)：大量输入 → 固定 latent → output Query，主计算在小 latent 中 | 输入扩展到很多时间 / 传感器 token 时，用少量全身 latent 做信息交换；关节 ID 对应 output query 的语义 | **未来扩展**。现在 KV 可直接是 4 个帧 token，无需先加完整 encoder / latent-self-attention / decoder 三阶段 |
| MQA / GQA | [MQA](https://arxiv.org/abs/1911.02150)、[GQA](https://aclanthology.org/2023.emnlp-main.298/)：减少 KV heads，优化逐 token 解码的 KV 带宽和缓存 | 4 个 Q heads 对应 1 或 2 个 KV heads，减少 K/V 投影与激活 | **低优先**。主要收益场景是增量解码，本任务每个 PPO 样本重算短窗口；不减少 29 个 Query FFN。共享全身 encoder 与 GQA 的“跨 heads 共享 KV”是不同维度的共享 |
| DeepSeek MLA / 低秩投影 | [DeepSeek-V2 §2.1.2](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2)：joint low-rank KV compression，可在推理时吸收部分投影，降低 KV cache；也使用 Query 低秩压缩减训练激活 | 可借“低秩瓶颈”检验投影 / 全局动作 head 的秩，例如 `d → r → d`，而不是复制整套 MLA / RoPE | **低优先**。d=64 本来很小，窄矩阵和更多 kernel 未必快；这里没有长 KV cache。论文整体训练 / 推理节省包含其他架构，不能全归于 MLA |
| FlashAttention / SDPA | [FlashAttention-2](https://arxiv.org/abs/2307.08691)、[官方实现](https://github.com/Dao-AILab/flash-attention)：保持精确 softmax attention，优化 IO、并行和工作分配 | 保持数学架构不变，使用可用的 fused backend；对表示假设没有额外影响 | **可做工程检查**，但短 cross-attn 是否快要实测。CUDA FA2 官方支持 FP16/BF16；FA3 优化 Hopper，FA4 优化 Hopper/Blackwell，不能当 FP32通用替代。当前 `need_weights=False` 已具备走 PyTorch SDPA 的条件（[官方 MHA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward)），是否真正使用某个 backend 仍需 profiling |
| Gated DeltaNet / Qwen3-Next / Kimi Linear | [Gated DeltaNet](https://arxiv.org/abs/2412.06464) 用门控和 delta rule 管理 recurrent memory；[Qwen 官方说明](https://qwen.ai/blog?id=qwen3-next) 混合线性与标准 attention；[Kimi Linear §5.6](https://arxiv.org/html/2510.26692v1) 比较长序列运行时间 | 历史扩展到很长或需要持续状态时，考虑固定状态 memory；小门控可借作局部 / 全局融合，但不应宣称由此省去主要计算 | **当前不优先**。Kimi 报告4k–16k时与MLA耗时相近，128k起优势明显；这里4帧远短于它的“短上下文”。跨episode reset、PPO minibatch状态重建和训练边界处理会增加系统复杂度 |

### 6.1 这里最有价值的论文信息

[HIM](https://junfeng-long.github.io/HIMLoco/static/pdfs/HIMLocomotion.pdf) 把显式线速度估计和隐式稳定性表示分别学习，使用短历史 MLP，再把表示与当前观测输入策略。它支持“线速度估计应被明确关注”，也说明全身历史不一定需要另一层时间 attention。其默认历史参数 $H=5$，extractor 为 512/256/128，不能把论文结论等同于本文的 64 维小 encoder。

[DreamWaQ](https://arxiv.org/html/2301.10602v2#S2.SS3) 的 CENet 也从历史本体观测估计机体速度和隐式环境表示，用模拟器真值监督速度。两者都支持把 **64 维 body 表示加一个 `64 → 3` 速度预测 head** 作为后续消融；这主要增加训练目标，而不是把实际线速度直接加入部署观测。是否与 PPO 联合更新、如何设置 loss weight，还需要单独处理，不能在架构消融首轮一起改变。

[Perceiver IO](https://arxiv.org/html/2107.14795v3#S3.SS2) 支持按输出元素构造不同 Query，输入特征和位置 / 类别身份可共同形成 Query。可以借这个接口：关节 Query 保留自身反馈和身份，body Query 请求全身表示；这里直接读取 4 帧 memory 即可，完整 Perceiver 的多阶段网络并不必要。

### 6.2 大模型速度数字不能直接套用

[MQA](https://arxiv.org/abs/1911.02150) 和 [GQA](https://aclanthology.org/2023.emnlp-main.298/) 的主要问题是增量生成时反复读取长 KV。4 / 30 个 KV 的 PPO 网络没有同样的缓存压力；减少 KV heads 也不能把宽 FFN 或动作 MLP 的计算消掉。

[MLA](https://arxiv.org/html/2405.04434v5#S2.SS1.SSS2) 的关键是低秩 joint KV compression 和推理时投影吸收，而不是“使用 latent 就一定更省训练计算”。若只把 K/V 写成两层投影却没有小秩或吸收，反而可能增加算子。

[FlashAttention 官方 README](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support) 写明 CUDA FA2 的 FP16 / BF16 支持；FA3 针对 H100 / H800。当前 README 还提供面向 Hopper / Blackwell 的 FA4。它们都是实现层优化，不会自行补齐全身历史表示。特别是FP32服务器，不能默认安装一个库就直接沿用原精度获得FA2/3加速。

[Kimi Linear](https://arxiv.org/html/2510.26692v1) 的报告区分了训练 / prefill 与 decoding：在4k–16k长序列时与MLA效率相近，128k及以上加速明显，1M context 的6倍数字是 decoding。其“RL scaling”属于语言模型后训练实验，不能当作本任务 PPO 机器人控制的验证。

## 7. 建议的消融顺序

所有对照固定观测项、奖励、随机化、PPO 设置、种子和训练步数；先在同一机器比较。至少同时记录线速度误差 (v_x,v_y,v_z)、线速度奖励、总奖励、episode length、rollout/update 分别耗时。仅看总奖励或整轮时间，难以判断表示与速度是否兼顾。

| 顺序 | 唯一主要改动 | 要回答的问题 |
|---|---|---|
| 0 | 当前分支基线 | 当前奖励和耗时能否稳定复现 |
| 优先 | 方案 D：原结构追加小全身历史 MLP，Q/K/V不变 | 几乎不增加执行量时，全身表示是否改善线速度 |
| 1 | 方案 A：保留29个local KV，IMU token替换为4个全身KV，其他模块尺寸固定 | 小成本加入同步全身历史能否改善后期线速度 |
| 2 | 在 A 上，只去掉29个local KV，得到方案 B | 全身KV是否足以替代局部历史，以及能省多少执行量 |
| 3 | 在选定memory方案上，把当前完整帧编码加入每个关节 Query | 当前全身姿态 conditioning 是否改善隐藏踝的历史读取 |
| 4 | 加 1 个 body Query，先保留现有动作 head | 显式全身历史向量是否有额外收益 |
| 5 | 删除 / 缩小关节 FFN，动作 head 暂保持 | 主要执行量能否减少，同时保住训练质量 |
| 6 | 以共享小动作 head 替换 flatten 大 MLP，接入 body 向量，得到方案 C | 容量与协调方式变化是否值得其速度收益 |
| 可选 A | 本关节4帧q/u encoder作为自身residual | 丢掉独立局部历史memory是否损害早期反馈优势 |
| 可选 B | body表示加显式速度监督 | 瓶颈是否主要是状态估计的学习目标 |
| 低成本对照 | 小全身历史MLP替换attention历史表征 | 短窗是否真的需要动态attention选择 |

方案 C 可以作为一个独立的整体候选快速测速度，但奖励比较仍应按上述步骤拆解。快速微基准只能筛查执行效率，不能替代 PPO 训练验证；不同 dtypes、TF32 设置、batch size、warmup、编译选项和 GPU 都要写明。部署 encoder 的缓存也只适合参数固定后的推理，不能在 PPO 多个 epochs 中直接复用旧参数生成的 latent，否则改变更新梯度。

## 8. 实测补充

### 8.1 日志中的整轮时间

本地最近 JointXAttn 的 TensorBoard 保存配置使用 BF16、d=64、head=(256,128,64)，稳定阶段的 1000 个采样记录均值：rollout 0.821 s、learning 1.789 s，合计约 2.611 s；learning 约占 69%。这是本地实验，不能当作另一台 FP32 服务器的分项时间。旧 MHA 日志的配置、观测和训练状态不同，不用于证明同条件的速度或奖励优势。

### 8.2 GPU 微基准条件

- RTX 4090；PyTorch 2.13.0+cu130；CUDA 13.0。
- FP32 参数、输入和输出；按项目默认设置启用 TF32 矩阵乘。没有 BF16 autocast，**也不是所有矩阵乘严格使用 IEEE FP32 尾数**。
- d=64 配 `(256,128,64)` head；d=128 配默认 `(512,256,128)` head；4 heads，J=29，H=5，F=63。
- Actor 编码器与动作 head；随机输入、随机初始化；不含观测 normalizer、Gaussian 分布、critic、PPO loss、optimizer、仿真和日志。反向目标是输出平方均值，只用于执行路径与张量规模相近的计时。
- rollout batch=8192，仅 inference forward；update batch=49152，forward+backward。每种形状先 warmup 3 次，随后 CUDA events 测 7 次，表中为中位数；未使用 `torch.compile`。
- 原实现直接复用现有 `_JointCrossAttention` 与 actor MLP。候选为临时结构原型，未改仓库训练实现，也未训练候选或比较奖励。

| Actor 结构 | d=64 rollout ms | d=64 fwd+bwd ms | d=128 fwd+bwd ms | d=64 相对基线 |
|---|---:|---:|---:|---:|
| 原结构，复测 | 6.71 | 112.11 | 184.20 | 基线 |
| 原结构 + 独立128维时间MHA分支 | 8.20 | 150.74 | 230.07 | 慢34.5% |
| D：原结构 + 全身历史MLP 252→128→64 | 7.08 | 113.95 | 185.42 | 慢1.6% |
| D共享版本：复用每帧encoder + 历史MLP | 6.86 | 115.56 | 188.98 | 慢3.1% |
| A：29 local + 4 global混合KV | 7.19 | 119.95 | 195.62 | 慢7.0% |
| B变体：仅4 global KV，Q加入当前全身编码 | 4.91 | 84.82 | 138.96 | 快24.3% |
| B+C：4 global KV + body Q + 共享小head | 4.28 | 74.05 | 118.03 | 快34.0% |
| B+C + 本关节历史残差加入Query | 5.10 | 82.03 | 131.14 | 快26.8% |
| A+C：33混合KV + body Q + 共享小head | 6.58 | 109.47 | 184.87 | 快2.4%，d=128无改善 |

注意区分描述中的控制变量：正文 B 可先保持原 Query 进行奖励消融；表中的 B 原型已把当前全身帧编码加入 Query。因此它的速度与正文严格“只改 KV”版本未必一致。B+C 的 body Query 使用当前帧编码，未额外加 learned body embedding；共享 head 为 `2d→d→1`，所以 d=128 时隐藏层为128。加本关节历史残差的版本仍由共享 local encoder 生成残差，但它进入对应关节 Query，而非保留为29个独立KV。

基线初测与复测的 update 中位数分别为112.57/112.11 ms（d=64）、184.65/184.20 ms（d=128），差约0.4%/0.2%。小比例差异只能说明低开销。更多参数共享、更少参数、更低 MAC 并不自动对应更短实测时间；混合memory+共享head就是反例。

### 8.3 Profiling 与 backend 检查

FP32 d=64、batch49152 的原 Actor 前向加反向 trace 已确认使用 PyTorch `aten::_scaled_dot_product_efficient_attention`，不是手写的未融合 softmax attention。单次 trace 中 attention forward/backward 对应 CUDA 执行约6.52/17.08 ms；`aten::copy_`约16.95 ms、矩阵乘`mm/addmm`约16.83/10.12 ms、LayerNorm forward/backward约6.53/8.81 ms。这些来自 profiling 而非稳态 CUDA-events 中位数，不能把嵌套 aten 事件与底层 CUDA kernel重复加总。

这也解释了为什么“QK/AV静态 MAC 仅7%”不能等同于“attention只花7%时间”。宽度很小、batch很大时，显存搬运、切分/复制、归一化和短序列kernel效率都重要。4KV版本减少的不仅是点积，也包括memory投影、激活与搬运。

另用当前安装版本的 FP32 Q=4 heads、KV=1 head、长度29/30进行 GQA backend 探测：强制 `EFFICIENT_ATTENTION` 报 `No available kernel`，math backend成功。因此在此版本上，直接换GQA可能失去当前 fused backend；未对其进行速度承诺。[当前在线2.14文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)已列出更广的GQA backend支持，不能把本地2.13探测推广为所有版本都不支持。

### 8.4 对当前问题的执行建议

先训练 D，检查用小全身历史MLP补信息能否改善线速度；这样保留当前架构的执行与局部历史路径。若有效且仍需要提速，再训练 B+C+局部历史Query残差作为第二候选，它保留自身短历史反馈和共享全身表示，微基准比原Actor快约27–29%。减少独立local KV和大head仍可能改变训练质量，需要与D、原结构分别比较。

整轮近3秒不能按Actor微基准比例直接缩放，尤其本地日志是BF16而此基准是FP32/TF32。候选仍须在目标服务器测完整rollout/update时间；此报告未声称已经实现奖励提升或整轮提速。
