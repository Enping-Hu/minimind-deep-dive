# A. 模型结构差异：QK-Norm / shared expert / head_dim

模型结构有三处改动，都对齐 Qwen3 / Qwen3-MoE。逐条对照两版 `model/model_minimind.py` 源码。

## A1. QK-Norm（v3 新增，Qwen3 标志）

v3 在 Attention 里对 Q、K 各加一个 per-head 的 RMSNorm（v3 `Attention` 的 `__init__` 加 norm 层、`forward` 里应用）：

```python
# v3 Attention.__init__
self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
# v3 forward：view 成多头后、应用 RoPE 前
xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
xq, xk = self.q_norm(xq), self.k_norm(xk)   # ← v2 没有这一步
```

- 作用对象：每个 head 的 `head_dim` 维向量，先做 RMSNorm 再进注意力。
- 直觉：把 Q/K 的尺度归一，稳定 `QK^T` 打分，避免深层/长训时注意力 logits 爆炸、注意力熵塌缩。这是 Qwen3 的标志改动。
- v2 对照（`minimind-master` 的 `Attention.forward`）：Q/K 投影后直接 RoPE + 打分，**没有** q_norm/k_norm。

顺序上，QK-Norm 在 view 成多头之后、[`apply_rotary_pos_emb`](../02-model/03-rope.md) 之前——先归一化、再旋转。

## A2. MoE 移除 shared expert

回顾 v2 的 MoE（[02-model/06-moe](../02-model/06-moe.md)）：routed experts（router top-k 选）**加** shared experts（`n_shared_experts=1`，所有 token 都过）。v2 `MOEFeedForward` 里有 `self.shared_experts`，forward 末尾对所有 token 叠加。

v3 的 `MOEFeedForward` **只有 routed experts，没有 shared experts**——`gate` + `experts` 两样，config 里连 `n_shared_experts` 都去掉了：

```python
# v3 MOEFeedForward.__init__
self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size)
                              for _ in range(config.num_experts)])
# 没有 self.shared_experts
```

含义：v3 MoE 更贴近 Qwen3-MoE（同样相对 DeepSeek-MoE 去掉了 shared expert）。router 的**选择写法**（gate→softmax→topk→norm）两版同形，但除了 shared expert，还有两处实打实的差异：默认 `num_experts_per_tok` 从 v2 的 2 降到 v3 的 1（top-2 → top-1）；aux_loss 从 v2 的序列级（`seq_aux=True`、系数 `aux_loss_alpha=0.01`）改成 v3 的 batch 级（`(load * scores.mean(0)).sum() * num_experts`、系数 `router_aux_loss_coef=5e-4`），公式和粒度都变了。

> 顺带一提，v3 还改了几个 config 名：`n_routed_experts → num_experts` 是纯改名；`aux_loss_alpha → router_aux_loss_coef` 不只是改名——如上所述，aux_loss 的算法本身也从序列级换成了 batch 级。命名整体更贴近 HuggingFace 习惯。

## A3. head_dim 解耦

- v2（`Attention.__init__`）：`head_dim = hidden_size // num_attention_heads`，**计算得出，不能单独设**。
- v3（`MiniMindConfig.__init__`）：`head_dim = kwargs.get("head_dim", hidden_size // num_attention_heads)`，**可独立配置**。

含义：v3 允许 `head_dim × num_heads ≠ hidden_size`，和 Qwen3 一样把注意力头维度与隐藏维解耦，更灵活。默认不传时两版算出来一样（如 768/8=96 或 512/8=64），但 v3 多了一个旋钮。

## 不是差异的点（防误记）

写这一节时反复确认过，下面几项两版一致，别误写成 v3 才有：

- **tie_word_embeddings**：两版都绑定（v2 在 `MiniMindForCausalLM.__init__` 硬编码，v3 同一位置走 config flag）。
- **rope_theta**：两版都 1e6。
- **is_causal**：v3 把 SDPA 的 `is_causal` 做成可配属性（`Attention.__init__` 里的 `self.is_causal`），v2 硬编码 `True`。仅工程细节，因果掩码行为一致。

## 练习

1. QK-Norm 加在哪、作用对象是什么、解决什么问题？它在 RoPE 之前还是之后？
2. v2 和 v3 的 MoE 差在哪一条支路？router top-k 和 aux_loss 变了吗？
3. v2 和 v3 的 `head_dim` 定义有什么区别？默认不传 `head_dim` 时两版结果一样吗？

<details>
<summary>参考答案</summary>

1. 加在 Attention 里对每个 head 的 Q、K 各做一个 `RMSNorm(head_dim)`，作用对象是每个 head 的 head_dim 维向量，稳定 `QK^T` 打分、防 logits 爆炸/熵塌缩；在 view 成多头之后、apply_rotary_pos_emb 之前。
2. 差在 shared expert——v2 有「所有 token 都过的共享专家」支路，v3 去掉只留 routed experts；router top-k 和 aux_loss 机制两版一致。
3. v2 是 `hidden_size // num_attention_heads`（不可单独设），v3 可通过 `head_dim` kwarg 独立配置；默认不传时两版算出来一样。
</details>
