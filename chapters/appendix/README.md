# 附录：深入延伸篇

主线（[00–10 章](../)）逐节对照 MiniMind 源码，把它真有的代码和设计选择读透。这里是**深入卷**：从主线讲到的某个组件或某个真实痛点出发，顺到它背后完整的技术脉络——有些 MiniMind 源码里有、只是没展开，有些 MiniMind 完全没涉及、但进阶绕不开。

**怎么读**：只想读懂源码，主线读完即可，不必进这里。想再深一层，就顺着主线里的跳转链接进来；每篇开头都标了前置（读完哪节即可），彼此大多独立，可按需挑读。

## 长上下文

| 篇目 | 从哪切入 | 讲什么 |
|---|---|---|
| [03 · RoPE 长度外推](03-rope-length-extrapolation.md) | v2 那个默认关闭的 `rope_scaling` 开关 | PI→NTK→YaRN 谱系，逐行对应 `precompute_freqs_cis` |
| [04 · KV cache 压缩](04-kv-cache-compression.md) | v2 的 GQA（8 头 Q / 2 组 KV） | MHA→MQA→GQA→MLA，怎么围绕 KV cache 一步步改 |

## 模型结构

| 篇目 | 从哪切入 | 讲什么 |
|---|---|---|
| [02 · 归一化的演进](02-normalization-evolution.md) | v2 的 RMSNorm、v3 的 QK-Norm | BatchNorm→LayerNorm→RMSNorm→QK-Norm，每步解决什么 |
| [09 · 位置编码的演进](09-positional-encoding-evolution.md) | v2/v3 都用的 RoPE | 绝对→相对→RoPE/ALiBi，为什么最后是 RoPE |
| [12 · MoE 的演进](12-moe-evolution.md) | v2/v3 的 MoE 配置默认值 | GShard→Switch→Mixtral→DeepSeekMoE，每个配置来自哪一段 |

## 对齐与 RL

| 篇目 | 从哪切入 | 讲什么 |
|---|---|---|
| [05 · GRPO 变体家族](05-grpo-variants.md) | v2 的 GRPO、v3 的 CISPO | Dr.GRPO / DAPO / GSPO 各自改了什么 |
| [08 · DPO 变体家族](08-dpo-variants.md) | v2 的原始 DPO（`train_dpo.py`） | SimPO/IPO/KTO/DPOP… 各自在补 DPO 的哪个洞 |
| [14 · RL 对齐算法全景](14-rl-alignment-landscape.md) | MiniMind 简化掉的那些 RL 零件 | RLHF 三步 / RLAIF / REINFORCE 家族 / VAPO / TTRL 全景地图 |

## 推理与部署

| 篇目 | 从哪切入 | 讲什么 |
|---|---|---|
| [06 · 模型量化](06-quantization.md) | 26M 模型想塞进端侧 | 位宽/粒度/阶段三旋钮，PTQ 难点在激活离群值（有损） |
| [07 · 投机解码](07-speculative-decoding.md) | MiniMind 逐 token 生成的基线 | 先猜后验、为何 lossless、接受率与加速上限（无损） |
| [10 · FlashAttention](10-flash-attention.md) | v2 Attention 的 Flash / 标准两条路径 | IO-aware 分块 + online softmax，同结果更少显存往返 |

## 训练方法

| 篇目 | 从哪切入 | 讲什么 |
|---|---|---|
| [11 · PEFT 全家族](11-peft-family.md) | v2 的 LoRA（`model_lora.py`） | soft prompt 线（Prompt/Prefix/P-Tuning）+ Adapter + LoRA 变体，各自省什么改什么 |

## 模型家族

| 篇目 | 从哪切入 | 讲什么 |
|---|---|---|
| [13 · DeepSeek 谱系](13-deepseek-lineage.md) | MiniMind 借来的那些设计 | V1→V3.2 时间线，每个设计出自哪一代、MiniMind 用没用 |

## 进阶入口（点到为止）

| 篇目 | 内容 |
|---|---|
| [01 · 进阶方向](01-advanced-pointers.md) | Flash Attention / LoRA / 知识蒸馏 / Agent RL 的最小原理 + 源码入口 |

---

> 深入篇随学习持续补充。
