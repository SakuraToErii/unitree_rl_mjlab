# `nn.MultiheadAttention` 的 `need_weights` 在这条任务上选反了

PyTorch 文档说 `need_weights=False` 才会走优化过的 `scaled_dot_product_attention`。在 GroupTime 这条 fp32、当前帧做 Query 的时间注意力上，实测相反：`False` 更慢，`True` 更快。

关节编码器和 IMU 编码器的训练前向都固定为 `need_weights=True`，返回的权重丢掉，不参与损失。不要为了“符合文档”改回 `False`，也不要强制切到 FlashAttention。

## 原因

环境是 PyTorch 2.13、RTX 4090、TF32。`need_weights=False` 时 `F.multi_head_attention_forward` 调用 `scaled_dot_product_attention`。这个形状是 fp32、head dim 16 或 8、Query 长度 1、Key/Value 长度 5：

- FlashAttention 要求 fp16 或 bf16，直接拒绝。
- cuDNN attention 没有可用 kernel。
- 落到 Cutlass memory-efficient kernel：`fmha_cutlassF/B_f32_aligned_64x64_sm80`。

`need_weights=True` 不走 SDPA，改走 `bmm` / `gemv` 的 math 路径。Query 长度是 1，这条路径更合适。两边输出最大差约 `1.2e-4`，梯度最大差约 `1e-11`。

## 实测

历史 5 帧，update batch 24576。关节路径的序列数是 `24576×29 = 712704`。IMU 路径的序列数就是 24576。

| 路径 | `need_weights=False` 前向+反向 | `True` 前向+反向 | 一次 PPO update（×20） |
|---|---:|---:|---:|
| 关节，64 维，Q `[712704, 1, 64]` | 418 ms | 100 ms | 8.36 s → 2.00 s |
| IMU，64 维，Q `[24576, 1, 64]` | 13.72 ms | 2.92 ms | 0.274 s → 0.058 s |

同一任务、4096 环境、history 5 的整轮 PPO：原来关节 `False` 时 update 8.72 s、整轮 9.68 s。关节改成 `True` 且内部 64 维后，update 2.35 s、整轮 2.97 s。关节再降到内部 32 维后，update 1.58 s、整轮 2.14 s。最后这组数里的 IMU 仍是 `False`。

## 还没改的地方

`_SummaryFusion` 仍是 `need_weights=False`。那是关节摘要之间的完整自注意力，不是 Query 长度 1 的时间检索，上面的计时不包括它。默认训练任务没有打开这层。
