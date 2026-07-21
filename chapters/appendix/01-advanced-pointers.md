# 进阶入口（从源码接点到深入篇）

主线聚焦 MiniMind 有的代码：模型结构、预训练、推理、SFT、对齐（DPO/PPO/GRPO/SPO）、训练机制、版本对照、实验。下面几个方向 MiniMind 源码里也有，但属于进阶分支，主线只点到为止。它们大多已各自成篇（收在本附录），这张卡只做一件事：从主线的源码接点，把你导到对应的深入篇。

## Flash Attention

主线 [02-model/02-attention](../02-model/02-attention.md) 里那条 `F.scaled_dot_product_attention` 分支就是它的入口——和标准 `softmax(QK^T/√d)V` **数学等价**，靠分块计算 + 不显式存完整注意力矩阵省显存。触发条件（v2 源码）：`self.flash and seq_len > 1 and past_key_value is None and 无复杂 mask`；增量推理带 KV cache 时回退标准路径。

- IO-aware 分块、online softmax 怎么做到等价却省显存 → 附录 [10 · FlashAttention](10-flash-attention.md)。

## LoRA

主线 [05-sft](../05-sft/01-assistant-only-supervision.md) 提过 `train_lora.py` 和 Full SFT 共用 `SFTDataset`、标签逻辑一致，区别只在更新哪些参数：Full SFT 更新全部，LoRA 冻结主干、只训注入的低秩 adapter（`model/model_lora.py` 的 `apply_lora`、`save_lora`/`load_lora`）。

- 它在参数高效微调全谱里的位置（soft prompt 线 / Adapter / LoRA 变体） → 附录 [11 · PEFT 全家族](11-peft-family.md)。

## 知识蒸馏

源码 `trainer/train_distillation.py`：用大 teacher 指导小 student，核心是 `distillation_loss`——student 除了学硬标签（CE），还用 KL 对齐 teacher 的**软标签分布**，温度 `T` 放大 logits 差异、`T²` 补偿梯度尺度，teacher 冻结。总 loss 常是 `α·CE + (1-α)·蒸馏KL`。它和 DPO/RL 的 reference 一样是「冻结一个模型当参照」，但目标不同：蒸馏让 student 逼近 teacher 分布，DPO/RL 是约束 policy 别漂移。

- 白盒/黑盒/R1 蒸馏、以及它在模型压缩三条线里的位置 → 附录 [16 · 知识蒸馏](16-knowledge-distillation.md)。

## Agent RL（v3 新增，暂无独立深入篇）

[09-minimind2-vs-3/05](../09-minimind2-vs-3/05-thinking-scale-removals.md) 提过，MiniMind-3 新增 `train_agent.py` + `rollout_engine.py` 和 `AgentRLDataset`，v2 没有。方向是多轮 rollout（模型与环境/工具多轮交互）+ 延迟 reward（reward 在多步之后才给），比单轮 PPO/GRPO 复杂。这是 v3 的进阶分支、涉及 v3 专属代码，主线未覆盖，本附录也暂未单独成篇——要学时直接读 v3 的 `rollout_engine.py`，看多轮采样怎么组织、reward 怎么延迟分配。对齐算法的整体版图可先看附录 [14 · RL 对齐算法全景](14-rl-alignment-landscape.md)。

---

> 其余主线只点到为止的细节（如 Tokenizer 的 BPE 训练细节），同样属于按需延伸：有源码 / 实操支撑的地方写深，其余标清边界、给出入口。完整的深入篇目录见[附录导航页](README.md)。
