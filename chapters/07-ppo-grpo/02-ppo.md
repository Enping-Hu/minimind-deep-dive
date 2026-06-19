# PPO：ratio / clip / critic / old actor

PPO（Proximal Policy Optimization）是这个项目里最完整的在线 RL，也是理解 GRPO/SPO 的参照系。它的关键不只是「往哪个方向改」（advantage 决定），还要控制「这一步改多大」（ratio/clip 决定）。这一节讲 PPO 的五个模型角色，重点拆 `ratio / clip / min` 这套稳定化机制。

源码：`trainer/train_ppo.py`，`ppo_train_epoch`（L119）、`CriticModel`（L29）。

> 本节讲的是 MiniMind2 的 PPO（五模型、简化优势）。MiniMind-3 已重写为标准 PPO2（四模型、token-level GAE），见 [第 9 章](../09-minimind2-vs-3/03-ppo-rewrite.md)。

## 五个模型角色

| 模型 | 作用 | 是否训练 |
|---|---|---|
| `actor_model` | 当前 policy，要优化的模型 | 训练 |
| `old_actor_model` | 旧 policy 快照，提供更新前的概率参照 | 周期同步，不反传 |
| `critic_model` | value model，估计 baseline | 训练 |
| `ref_model` | 冻结参考，约束行为漂移（KL） | 冻结 |
| `reward_model` | 冻结奖励模型，给 response 打分 | 冻结 |

分工要记清：**reward_model 给分，critic 估 baseline，ref_model 防漂移，old_actor 量更新幅度。** `CriticModel`（L29）继承自 `MiniMindForCausalLM`，把语言模型头换成输出标量 value 的头。

## advantage = reward − critic value

```python
rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)
values = critic_model(input_ids=gen_out, ...)[..., last_indices]
advantages = rewards - values.detach()
```

critic 估「这个 prompt 正常能拿几分」，实际 reward 减去它就是 advantage：比预期好为正、差为负（[01-rl-overview](01-rl-overview.md) 的 baseline 思想，PPO 的 baseline 来自 critic）。`.detach()` 是因为这一步只用 value 当基准、不让 advantage 的梯度流回 critic（critic 另有自己的 value_loss 回归 reward）。

## ratio：新旧 policy 的概率比

PPO 不只看当前 actor 输出了什么，还看它相比更新前改了多少。`old_logp` 是旧 policy 对同一条 response 的对数概率，`actor_logp` 是当前 policy 的：

```python
ratio = torch.exp(actor_logp - old_logp)
```

为什么这是概率比？因为 `log a − log b = log(a/b)`：

$$\text{actor\_logp} - \text{old\_logp} = \log\frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)} \;\Rightarrow\; \text{ratio} = \frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)}$$

`ratio > 1` 新 policy 更喜欢这条回答，`< 1` 更不喜欢，`= 1` 基本没变。所以 **ratio 表达「这一步改了多少」，advantage 表达「该不该强化」。**

## clip：限制单步幅度

只用 `surr1 = ratio * advantages` 会有问题：只要方向对，就可能一步把概率推得很猛，训练不稳、policy 偏离旧分布太快。PPO 加 clip：

```python
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

`clip_epsilon`（v2 默认 0.1）把 ratio 截在 `[0.9, 1.1]`。**clip 不改更新方向，只限更新幅度。**

`min(surr1, surr2)` 是关键保险丝：取更保守（更小）的那个目标。态度是「如果不 clip 时本来能拿更大收益，但收益来自改得太猛，那我宁可只认 clip 后更保守的收益」。一旦更新想冲出去，目标函数就不再奖励它。

分两种情况看：

- **advantage > 0**（该强化）：ratio 略大于 1 是好事；但若远超 `1+ε`，说明强化过猛，clip+min 截住。
- **advantage < 0**（该削弱）：若 ratio 还很大，等于把不该强化的回答反而推高了，更危险，同样被 clip+min 限制。

不管正负优势，目标一致：**允许朝对的方向改，但不允许一步改过头。** 末尾负号同前——想最大化 surrogate objective，写成 loss 加负号。

![PPO ratio/clip 流程](../../images/ppo-ratio-clip-flow.svg)

## 完整 loss

PPO 的总损失还包括 critic 的 value_loss（回归 reward）和对 ref_model 的 KL 惩罚（防漂移）：`policy_loss + value_loss + KL`。`ppo_train_epoch`（L119）把生成、打分、算优势、算三部分 loss、更新串起来。`old_actor_model` 按 `update_old_actor_freq` 周期同步成当前 actor，提供下一轮的 `old_logp`。

## 常见误区

- **「ratio 越大越好」**——不对，ratio 太大说明更新过猛，正是 PPO 要限制的。
- **「clip 改变更新方向」**——不，clip 只限幅度，方向由 advantage 决定。
- **「必须先推完 policy gradient 数学」**——不必，先把 ratio/clip/min 三者和代码对齐即可。

## 练习

1. PPO 的五个模型各是什么角色？哪些训练、哪些冻结？
2. `ratio = exp(actor_logp - old_logp)` 为什么等于新旧 policy 的概率比？ratio 和 advantage 各表达什么？
3. 只用 `ratio * advantages` 会有什么问题？clip 和 `min(surr1, surr2)` 共同在防什么？
4. advantage 算式里的 `values.detach()` 为什么要 detach？

<details>
<summary>参考答案</summary>

1. actor（训练的 policy）、old_actor（旧快照，量更新幅度，不反传）、critic（估 baseline，训练）、ref（冻结，KL 防漂移）、reward（冻结，打分）。
2. `log(π_θ) − log(π_old) = log(π_θ/π_old)`，取 exp 得概率比；ratio 表达这一步改了多少（幅度/方向），advantage 表达这条回答该不该强化。
3. 只用 surr1 会让更新可能一步冲太远、训练不稳；clip 把 ratio 限在 `1±ε`、min 取更保守的目标，共同防单步更新过大。
4. advantage 只把 critic value 当基准，不应让 advantage 的梯度流回 critic（critic 由自己的 value_loss 训练），所以 detach。
</details>
