# 部分可观测与历史注意力（AME、RMA、HIM、Transformer）

> 本仓库在 mjlab 上用 PPO 训练 Unitree G1 的 0-PD 残差力矩行走。Actor 是 4+1 的 `ResidualMhaModel`：过去 4 帧做自注意力，last-token 读出，当前帧 $s_t$ 单独编码后拼接，再进 MLP。学习得到的历史 pooling 已经实证为负。下一步想做部分可观测：砍掉 IMU / 角速度 / 投影重力 / 关节速度，或再砍掉部分关节，希望时间注意力能补回缺失物理量（用位置差分出速度、用历史估重力、用耦合补关节）。问题是：历史上的双向自注意力是否符合直觉？因果注意力行不行？$s_t$ 当 query、历史当 KV（再残差接回 $s_t$）会不会更好？AME 用本体状态当 Q、地形特征当 KV，有哪些可以迁移？

只依据一手论文。本地 actor：对 $H=4$ 的过去帧做双向自注意力，last-token 或 pooling，再拼接编码后的 $s_t$。本文不涉及实现代码。

## 1. AME 关注的是空间，不是时间

AME（He et al.）是在机体系高程图上做空间选择。CNN 保留 $L \times W$ 个点；本体状态是一个 query；地图点是 K/V：

> “we use the proprioception embedding as the query and point-wise local features as the key-value pairs.” $n=1$，$d=64$，$h=16$。地图尺寸：ANYmal $26 \times 16$，GR-1 $17 \times 11$。注意力会落到未来落脚点上。

策略仍然能看到当前的 $v_b,\omega_b,g_b,q,\dot q,a_{t-1}$，再加上这段地图编码。没有时间 Transformer。

AME-2 仍是空间 MHA。Query = MLP(全局地图池化, 本体状态)；局部点仍是 K/V。教师策略含 $v_b$。学生策略**砍掉 $v_b$**，叠 20 步（仍留 $\omega,g,q,\dot q$），经 LSIO 编码后，把该向量当作查地图的 query。理由是：“a 1 cm position drift over a 20 ms timestep can translate to a velocity error of 0.5 m/s.” 注意力索引的仍是地形。

## 2. Q=本体、KV=外感知 vs Q=$s_t$、KV=历史

代数上类似：一个 query、一袋上下文、再把 query 和 attended 拼回去。功能上不是一回事。AME 的 KV 是几百个无序地形点，多数与当前步无关，Q 用来挑落脚点。$T=4$ 的历史又短又有序，差分时每一帧都有用。LocoTransformer 把一个本体 token 和视觉 patch 做联合自注意力，也不是 AME 那种 $n=1$ 的 cross-attn。

可以迁移的是保留当前帧残差 $s_t$（AME、HIM、DreamWaQ、LSIO 都把当前与上下文拼接）。不能迁移的是「注意力会自动忽略垃圾 token」。

## 3. 用历史恢复 $v_b$ / IMU

可部署的 actor 通常看不到实测 $v_b$。短历史 **MLP** 就够：

| 工作 | $H$ | 编码器 | 恢复什么 | Actor 仍保留 |
|---|---|---|---|---|
| RMA | 50（0.5 s） | 一维 CNN | 外源 $z$，不是 $v$ | $q,\dot q$，IMU 横滚/俯仰，接触 |
| Ji et al. | $t-20/40/60$ ms | MLP；LSTM「没有实质差别」 | 显式 $v$、足端 $z$、接触 | IMU $+q,\dot q$ |
| DreamWaQ | 5 | MLP CENet | 显式 $v$ + VAE 隐变量 | $\omega,g,cmd,q,\dot q,a_{t-1}$ |
| HIM | 5 | MLP 512-256-128 | $\hat v$ + 对比学习隐变量 | 同上；$\hat v$ 进策略 |
| AME-2 学生 | 20 | LSIO | 隐式 $v$ | $\omega,g,q,\dot q$；没有 $v_b$ |
| LSIO | 4（约 0.1 s）+ 66（2 s） | 短→MLP；长→一维 CNN | 系统辨识；EKF 的 $v$ 已在 $o$ 里 | IMU + 电机 $q,\dot q$ |

编码器里的 $\dot q$ 经典做法是 $\Delta q / \Delta t$。重力是陀螺加加速度计的互补滤波，不是 Transformer。$T=4$ 已经对齐 HIM / DreamWaQ / Ji。**砍掉 $v_b$ 并不需要注意力。** 这些论文真正对照的是显式 $\hat v$ 头。

砍 IMU（$\omega$ 和 $g$）更难。MSTA：$\dot q$、接触、$\omega$、$a_{t-1}$「相当冗余」；$q$ 和 $g$「更关键」。AME-2 学生仍保留 $\omega,g$。

## 4. 因果 vs 双向（窗口已经 $\le t-1$）

因果注意力用在**当前** token 还在序列里的时候（Radosavovic：因果 Transformer，上下文 16，「只看前面的 token」）。更长上下文在斜坡上有帮助。

本仓库只编码过去帧，$s_t$ 从 MHA 外面注入，没有未来可漏。双向自注意力更适合平滑的有限差分。$T=4$ 上再加因果，会让早期 token 变瞎；last-token 本来就会丢掉它们，除非做 pooling。因果是可选项，不是默认。

## 5. Cross-attn vs last-token vs pooling

- 时间维自注意力，取最后 / 因果 token：Radosavovic，TERT 一类时间步 token。
- CLS / 可学习 pooling：MSTA（对掩码后的传感器-时间 token 做状态嵌入）。
- Q=当前，KV=大的**空间**集合：AME / AME-2。
- 拼接当前 + 编码后的历史（不再另做 Q）：HIM、DreamWaQ、LSIO 短流、本仓库 4+1。
- 传感器维（不是时间步）自注意力加掩码：MSTA。

Q=$s_t$、KV=历史是成本很低的消融。要保留残差 $s_t$。在 $T=4$ 上，它不该被理解成「一定能打赢 last-token / pooling」；pooling 在本仓库已经失败。

## 6. 砍关节不等于砍速度

- **速度是时间量。** $\dot q \approx \Delta q / \Delta t$；$v_b$ 来自运动学 + IMU + 接触。HIM / DreamWaQ / Ji 用剩余传感器约 $H \approx 3 \sim 5$ 就能恢复。MSTA：$\dot q$ 和 $\omega$ 最好砍。
- **关节是空间量。** 缺 $q_j,\dot q_j$ 会在运动链上打洞。MSTA：编码器侧丢关节「影响可以很大」；丢几个关节仍可能走。Kwon et al.（RJM）用 $H=30$ 蒸馏受损关节隐变量，这是 RMA 式系统辨识，不是有限差分。耦合补不出 IMU 重力。

## 7. 不要从 AME 做类比

地形 KV 是 $O(10^2)$ 个无结构点；$T=4$ 又短又有序。AME 的注意力是落脚点选择器，不是微分器。AME / AME-2 **并不**砍 IMU；它们砍 $v_b$，另开时间编码器（MLP / LSIO），再把结果当作查地图的 Q。AME 靠忽略无关**空间**来泛化；历史重建需要邻近的**时间**。

**对本 actor：** 在 $H=4$ 上砍 $v_b$（也许再砍 $\dot q$）是文献默认；若要对齐他们的消融，加一个显式 $\hat v$ 头。砍 $g$ / $\omega$ 或关节是 MSTA / RJM 的掩码问题，不是 AME 迁移。过去窗上的双向是默认。Q=$s_t$、KV=历史且**保留残差 $s_t$** 只适合当消融，不要因为 AME 就认定必须这样。

## 来源

- [AME, He et al., arXiv:2506.09588](https://arxiv.org/html/2506.09588) / [abs](https://arxiv.org/abs/2506.09588)
- [AME-2, Zhang et al., arXiv:2601.08485](https://arxiv.org/html/2601.08485)
- 非官方 AME G1：[SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion)；AME-2：[Kitjesen/ame2](https://github.com/Kitjesen/ame2)
- [RMA, Kumar et al., RSS 2021](https://arxiv.org/abs/2107.04034) / [PDF](https://roboticsproceedings.org/rss17/p011.pdf)
- [Ji et al., RA-L 2022](https://arxiv.org/abs/2202.05481)
- [DreamWaQ, Nahrendra et al., ICRA 2023](https://ar5iv.labs.arxiv.org/html/2301.10602)
- [HIM / HIMLoco, Long et al., ICLR 2024](https://arxiv.org/html/2312.11460) / [code](https://github.com/OpenRobotLab/HIMLoco)
- [LSIO, Li et al., IJRR 2024](https://arxiv.org/html/2401.16889v2)
- [Radosavovic et al., Sci. Robotics 2024](https://arxiv.org/html/2303.03381)
- [LocoTransformer, Yang et al., ICLR 2022](https://arxiv.org/html/2107.03996)
- [MSTA, Liu et al., arXiv:2409.03332](https://arxiv.org/html/2409.03332v2)
- [Random Joint Masking, Kwon et al., arXiv:2403.00398](https://arxiv.org/html/2403.00398v1)
- [GaRLILEO，有限差分 $\dot q$](https://arxiv.org/html/2511.13216)
- IMU 重力互补滤波：[two-stage proprioceptive SE](https://discovery.ucl.ac.uk/id/eprint/10216087/1/Two-Stage%20Proprioceptive%20State%20Estimation%20with%20Stability%20Guarantee%20for%20Legged%20Robots.pdf)
