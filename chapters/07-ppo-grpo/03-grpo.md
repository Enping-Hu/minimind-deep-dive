# GRPO：组内相对 reward，不用 critic

PPO 要训练一个 critic 来估 baseline。GRPO（Group Relative Policy Optimization）省掉 critic：**让同一个 prompt 生成多条回答，在这一组内部比较谁更好**，用组内均值当 baseline。这一节看它怎么构造组内相对 advantage。

源码：`trainer/train_grpo.py`，`grpo_train_epoch`。相比 PPO，脚本里**没有** `CriticModel`、`old_actor_model`、`value_loss`，只留 policy / ref / reward 三个模型。

> 版本差异：MiniMind-3 GRPO 默认走 **CISPO** 变体，不是本节的经典 clipped surrogate，见 [第 9 章](../09-minimind2-vs-3/04-grpo-cispo.md)。

## 同一个 prompt 生成多条回答

```python
outputs = model_for_gen.generate(**prompt_inputs, num_return_sequences=args.num_generations, ...)  # 默认 8
completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]   # 只留 response，[B*G, R]
```

`num_generations`（默认 8）让每个 prompt 生成 G 条回答，batch 维从 `B` 变成 `B*G`（PPO 默认每 prompt 1 条）。这是 GRPO 多出来的采样成本，也是「组」的来源。

## 组内相对 advantage

每条回答打分得到 `rewards: [B*G]`，reshape 回组结构再算组内均值/标准差：

```python
grouped_rewards = rewards.view(-1, args.num_generations)                       # [B, G]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)   # [B*G]
std_r  = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)      # batch 内再标准化
```

核心是 `advantage = (reward − 组均值) / 组标准差`：这条回答相对**同一 prompt 的其他回答**好了几个标准差。高于同组平均→正→增强，低于→负→削弱。`repeat_interleave` 把每组的均值复制 G 份对齐扁平的 `rewards`，`+1e-4`/`clamp` 防数值爆炸。

为什么在同组内比？不同 prompt 难度不同（简单事实题 vs 复杂推理题），直接跨 prompt 比 reward 不公平。GRPO 只问「同一个问题下，模型自己采样的这些候选里哪个更好」，用组内均值替代了 PPO critic 的「这个 prompt 正常能得几分」。

![GRPO 组内相对优势流程](../../images/grpo-group-relative-flow.svg)

## policy loss 与 ref KL

GRPO 没有 old actor，但保留 ref_model 做漂移约束：

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1                       # k3 KL 估计，token 级
per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
                   - args.beta * per_token_kl)                       # beta 默认 0.02
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
```

几个点：

- `advantages.unsqueeze(1)` 把**回答级** advantage 广播到该回答的每个 token——同一条 response 的所有有效 token 共享一个 advantage。
- `completion_mask` 只统计 EOS 之前的有效 token（延续 SFT/DPO 只监督有效区域的思想）。
- `args.beta * per_token_kl` 是 ref 约束，平衡「追 reward」和「别偏离原模型」。
- `torch.exp(per_token_logps - per_token_logps.detach())`：前向值≈1（同一个量相减），但梯度仍流经当前 `per_token_logps`。**别把它误读成 PPO 的新旧 policy ratio**——GRPO 没有 old_actor，这只是「让当前 token log-prob 的梯度承载 advantage 信号」的工程写法。

## GRPO vs PPO

| 维度 | PPO | GRPO |
|---|---|---|
| 每 prompt 生成 | 1 条 | 多条（默认 8）|
| critic | 需要 | 不需要 |
| baseline | critic value | 同组 reward 均值 |
| advantage | `reward − value` | `(reward − 组均值)/组标准差` |
| old actor | 需要 | 不需要 |
| ref model | 有 | 有 |
| 主要代价 | 多训一个 critic | 多生成、多打分 |

一句话：**PPO 用 critic 学 baseline，GRPO 用同一 prompt 的多条回答现场算 baseline。** GRPO 不是免费——省掉 critic 训练，成本转移到多样本生成和 reward 评估。

## 练习

1. GRPO 为什么不需要 critic？baseline 从哪来？
2. `num_return_sequences=num_generations` 把 batch 维变成什么？为什么要 reshape 成 `[B, G]`？
3. `advantage = (reward − mean_r) / (std_r + 1e-4)` 表示什么？为什么在同组内比而不是全 batch 比？
4. `torch.exp(per_token_logps - per_token_logps.detach())` 是 PPO 的 ratio 吗？为什么？

<details>
<summary>参考答案</summary>

1. 用同一 prompt 多条回答的组内 reward 均值当 baseline、组内相对表现构造 advantage，不需训练 critic；baseline 是组均值。
2. 从 `B` 变成 `B*G`；reshape 成 `[B, G]` 让每行对应同一 prompt 的 G 条回答，才能在组内比较。
3. 这条回答相对同组平均好几个标准差；同组比能消除不同 prompt 难度差异带来的 reward 不可比。
4. 不是。GRPO 没有 old_actor；该式前向≈1、只让梯度流经当前 log-prob 承载 advantage，不是新旧 policy 概率比。
</details>
