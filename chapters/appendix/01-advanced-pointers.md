# 进阶方向（点到为止）

本书正文聚焦 MiniMind 主线：模型结构、预训练、推理、SFT、对齐（DPO/PPO/GRPO/SPO）、训练机制、版本对照、实验。下面四个方向 MiniMind 源码里都有，但属于进阶分支，本书不展开成正文深度，只给最小原理 + 源码/外链入口，供你按需延伸阅读。

## Flash Attention

正文 [02-model/02-attention](../02-model/02-attention.md) 提过 Attention 有两条路径：标准实现和 `F.scaled_dot_product_attention`（`model_minimind.py` Attention.forward）。后者就是 Flash Attention 的入口——PyTorch 2.0+ 内置，和标准 `softmax(QK^T/√d)V` **数学等价**，但通过分块计算 + 不显式存完整注意力矩阵，省显存、提速度。

- 触发条件（v2 源码）：`self.flash and seq_len > 1 and past_key_value is None and 无复杂 mask` 才走这条；增量推理带 KV cache 时回退标准路径（[04-inference/01](../04-inference/01-kv-cache-and-generate.md) 讲过原因）。
- 深入：读 FlashAttention 原论文（Dao et al.）了解 IO-aware 分块和 online softmax；本书只需知道「它是标准 attention 的等价高效实现，由 PyTorch 提供」。

## LoRA

[05-sft](../05-sft/01-assistant-only-supervision.md) 提过 `train_lora.py` 和 Full SFT 用同一个 `SFTDataset`、标签逻辑一致，区别只在**更新哪些参数**：Full SFT 更新全部，LoRA 冻结主干、只训练注入的低秩 adapter。

- 源码：`model/model_lora.py` 有 `LoRA` 模块、`apply_lora(model, rank=8)`（给线性层注入低秩旁路）、`save_lora`/`load_lora`（只存/load adapter 权重）。`train_lora.py` 里 `apply_lora` 后只把 `'lora' in name` 的参数交给 optimizer，主干 `requires_grad=False`。
- 原理：用两个低秩矩阵 `A·B`（rank≪hidden）近似权重增量 `ΔW`，训练参数量降几个数量级，adapter 可单独保存/切换（如 `lora_identity` / `lora_medical`）。
- 深入：LoRA 原论文（Hu et al.）；本书只需知道「冻结主干 + 训练低秩旁路」。

## 知识蒸馏

源码 `trainer/train_distillation.py`：用一个大 teacher 模型指导小 student 模型。核心是 `distillation_loss`：

```python
teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
return (temperature ** 2) * kl
```

- 原理：student 不只学硬标签（CE），还学 teacher 的**软标签分布**（KL 对齐），温度 `T` 放大 logits 差异、`T²` 补偿梯度尺度。teacher 冻结（`requires_grad_(False)`）。训练时总 loss 常是 `α·CE + (1-α)·蒸馏KL`。
- 它和 [DPO 的 ref_model](../06-dpo/01-preference-optimization.md)、RL 的 reference 一样，都是「冻结一个模型当参照」，但目标不同：蒸馏是让 student 逼近 teacher 分布，DPO/RL 是约束 policy 别漂移。
- 深入：Hinton 的 KD 原论文。

## Agent RL（v3 新增）

[09-minimind2-vs-3/05](../09-minimind2-vs-3/05-thinking-scale-removals.md) 提过，MiniMind-3 新增 `train_agent.py` + `rollout_engine.py` 和 `AgentRLDataset`，v2 没有。

- 方向：多轮 rollout（模型与环境/工具多轮交互）+ 延迟 reward（reward 在多步之后才给），比单轮 PPO/GRPO 复杂。
- 本书定位：这是 v3 的进阶分支，且涉及 v3 专属代码，正文未覆盖。要学时直接读 v3 的 `rollout_engine.py` 看多轮采样怎么组织、reward 怎么延迟分配。

## 本书范围说明

以上四项，加上正文只点到为止的一些主题（如 Tokenizer 的 BPE 训练细节），都属于按需延伸的方向：有源码 / 实操支撑的地方写深，其余只标清边界、给出入口，供你顺着往下查。
