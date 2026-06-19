# MoE：router、expert 与 aux_loss

MoE（Mixture of Experts）把 block 后半段的单个 `FeedForward` 换成「多个 FeedForward + 一个路由器」。每个 token 只走其中少数几个专家，于是模型能堆很多参数，但单个 token 的计算量不跟着专家总数线性涨。这一节讲 MiniMind 的 MoE 实现：router 怎么选专家、aux_loss 为什么必须有、它怎么一路加到训练 loss。

源码：`model/model_minimind.py`，`MoEGate`（L349–424）、`MOEFeedForward`（L427–514）。由 `config.use_moe` 开关（dense 时这一节整段不生效）。

## 配置先看懂

```python
use_moe = False             # 是否启用 MoE
n_routed_experts = 4        # 可路由专家总数
num_experts_per_tok = 2     # 每个 token 选 top-k 个
n_shared_experts = 1        # 所有 token 都过的共享专家
aux_loss_alpha = 0.01       # 辅助损失权重
```

默认：4 个路由专家，每个 token 选 2 个；另有 1 个共享专家所有 token 都走。

## MoEGate：给每个 token 选专家

`MoEGate.forward`（L384–424）：

```python
hidden_states = hidden_states.view(-1, h)          # [B*T, H]
logits = F.linear(hidden_states, self.weight)      # weight: [n_experts, H] → logits [B*T, n_experts]
scores = logits.softmax(dim=-1)                    # 每个 token 对各专家的概率
topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
if self.top_k > 1 and self.norm_topk_prob:
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)  # 重新归一化
```

举例：某 token 对 4 个专家的概率是 `[0.10, 0.55, 0.05, 0.30]`，top-2 取出 `idx=[1,3]`、`weight=[0.55,0.30]`，归一化后 `[0.647, 0.353]`。含义：这个 token 送进 Expert 1 和 Expert 3，输出分别乘 0.647、0.353 再相加。

## 每个 expert 就是一个 FeedForward

```python
self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])
```

所以 MoE 不是全新计算，而是「多个 [FeedForward](05-swiglu.md) 并排放着，router 决定每个 token 走哪几个」。训练时（L465–475）按 expert 分发 token：

```python
x = x.repeat_interleave(num_experts_per_tok, dim=0)   # 一个 token 复制 top-k 份
for i, expert in enumerate(self.experts):
    expert_out = expert(x[flat_topk_idx == i])        # 每个 expert 只算分给它的 token
y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)  # 按权重合并
```

token 复制 top-k 份，是因为它要同时送进 top-k 个专家。

## routed experts vs shared experts

```python
if config.n_shared_experts > 0:
    for expert in self.shared_experts:
        y = y + expert(identity)     # 所有 token 都额外过共享专家
```

- **routed experts**：router 选，每个 token 只走 top-k 个，不同 token 走不同专家——偏「分工」。
- **shared experts**：不经路由，所有 token 都走——偏「通用能力兜底」。

> 版本差异：MiniMind-3 **移除了 shared expert**，只保留 routed experts，更贴近 Qwen3-MoE。详见 [第 9 章](../09-minimind2-vs-3/02-architecture-diffs.md)。

## aux_loss：防止专家塌缩

MoE 有个典型风险：router 总把 token 分给少数几个专家，其余专家学不到东西，参数白堆。这叫路由不均衡 / 专家塌缩。`aux_loss` 是一个辅助损失，鼓励整体路由别长期挤向少数专家。

默认 `seq_aux=True`（L407–414），核心是结合两个量：每个专家**实际被选中的频率** `ce`，和 router 给每个专家的**平均概率**。两者乘起来求和，当某专家既被频繁选中、平均概率又高时，惩罚变大，从而把路由推平。注意目标不是让每个 token 平均用所有专家，而是让**整体**负载均衡。

![MoE router 与 aux_loss 源码链](../../images/moe-router-aux-loss-source-chain.svg)

## aux_loss 怎么加到训练 loss

aux_loss 从 MoE 层一路冒泡到训练脚本：

1. `MoEGate.forward` 算出 `aux_loss`，`MOEFeedForward.forward` 存到 `self.aux_loss`（L483）。
2. `MiniMindModel.forward`（L619）汇总所有 MoE 层：
   ```python
   aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
                  hidden_states.new_zeros(1).squeeze())
   ```
   dense 模型没有 MoE 层，结果就是 0。
3. `MiniMindForCausalLM.forward`（L671）挂到 `output.aux_loss`。
4. 训练脚本把它加到语言模型 loss：`loss = res.loss + res.aux_loss`。

所以总目标是 `语言模型 loss + MoE 辅助损失`。即使是 dense 模型 aux_loss 也是 0，训练脚本统一写 `res.loss + res.aux_loss` 不出错。

## 训练路径 vs 推理路径

`MOEFeedForward.forward` 里训练和推理走不同实现（L465 / L478）：训练路径直接按 expert 分发、便于反向传播；推理用 `moe_infer`（L487，`@torch.no_grad()`）按专家分组批量处理 token，避免逐 token 调用专家的低效。`moe_infer` 是推理工程优化，不影响理解 MoE 机制，这里不展开。

## 练习

1. MoE 替换的是 block 里的哪一部分？每个 routed expert 本质是什么？
2. `MoEGate` 输出的 `topk_idx` 和 `topk_weight` 分别是什么？
3. 为什么 MoE 能增加参数容量，但单 token 计算量不按专家总数线性增长？
4. 没有 `aux_loss` 会出什么问题？它在哪里被加进训练 loss？
5. routed experts 和 shared experts 的区别是什么？

<details>
<summary>参考答案</summary>

1. 替换 block 后半段的 FFN/MLP（不是 Attention）；每个 routed expert 就是一个 `FeedForward`。
2. `topk_idx` 是每个 token 选中的专家编号，`topk_weight` 是合并这些专家输出时的权重（可选归一化到和为 1）。
3. 模型可以有很多专家参数，但每个 token 只激活 top-k 个 routed expert，不计算全部专家。
4. router 可能长期偏向少数专家，导致负载不均、部分专家学不到东西（专家塌缩）；aux_loss 在 `MiniMindModel.forward` 汇总、挂到 `output.aux_loss`，训练脚本用 `res.loss + res.aux_loss` 加进总 loss。
5. routed experts 由 router 按 token 选 top-k，不同 token 走不同专家；shared experts 不经路由，所有 token 都走。
</details>
