# C. GRPO：默认 CISPO

[第 7 章的 GRPO](../07-ppo-grpo/03-grpo.md) 讲的 clipped surrogate（`min(ratio*adv, clamp(ratio)*adv)`）在 v3 里仍然存在，但**不再是默认**。v3 GRPO 加了 `--loss_type`，默认走 **CISPO** 变体。对照 `trainer/train_grpo.py`。

## 两个 loss 分支

v3 `train_grpo.py:135-142`：

```python
ratio = torch.exp(per_token_logps - old_per_token_logps)        # 逐 token
if args.loss_type == "cispo":                                    # 默认
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()   # 截上界 + 停梯度，当权重
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
else:  # "grpo"，经典分支
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss = -(torch.min(ratio * advantages.unsqueeze(1),
                                 clipped_ratio * advantages.unsqueeze(1)) - args.beta * per_token_kl)
```

默认值（`:226-228`）：`loss_type="cispo"`、`num_generations` 8→6、`beta` 0.02→0.1，另有 `epsilon_high=5.0`。**组内相对 advantage 算法不变**：仍是 `(reward − 组均值) / (组标准差 + eps)`（第 7 章的核心），CISPO 只改了 advantage 之后怎么进 loss。

## CISPO 和经典 GRPO 的关键区别

经典 GRPO（`else` 分支，[第 7 章](../07-ppo-grpo/03-grpo.md) 学的）裁剪的是**目标**：用 `min(ratio*adv, clip(ratio)*adv)`，ratio 越界的 token 会被 clip 卡住、梯度归零。

**CISPO（Clipped IS-weight）裁剪的是权重**：把重要性比值 `ratio` 截一个上界（`epsilon_high`）、`detach` 成固定权重，乘到 `logp × advantage` 上（REINFORCE 式），再减 k3 KL。关键差别：

- 经典 GRPO clip 会让越界 token **梯度归零**（被 clip 的项不再贡献梯度）。
- CISPO 把 ratio 截上界后 `detach` 成权重，**被截的 token 仍贡献梯度**（梯度走 `per_token_logps` 那一路，权重只是常数系数）。

所以 CISPO 的动机是「别让越界 token 完全失去梯度信号」。这个做法来自 MiniMax-M1 系列。

注意 v3 GRPO 仍是 on-policy 的近似：它也用 rollout 时的 `old_per_token_logps`（`:90`，detach）算 ratio，和 v3 PPO 用 logp 快照同源——不是第 7 章 v2 GRPO 里那个「前向≈1 的 `exp(logp − logp.detach())`」写法。

## 对第 7 章的影响

[第 7 章的 GRPO](../07-ppo-grpo/03-grpo.md) 讲的 clipped surrogate 对应 v3 的 `--loss_type grpo` 分支。读 v3 GRPO 时记住：**默认是 CISPO，经典 GRPO 要显式传 `--loss_type grpo`**。

## 练习

1. v3 GRPO 默认 `loss_type` 是什么？组内 advantage 的算法变了吗？
2. CISPO 和经典 GRPO 在「越界 token 的梯度」上有什么关键区别？
3. CISPO 里 `torch.clamp(ratio, max=epsilon_high).detach()` 的 `detach` 起什么作用？

<details>
<summary>参考答案</summary>

1. 默认 `cispo`（经典 GRPO 要显式 `--loss_type grpo`）；组内相对 advantage `(reward−组均值)/(组标准差+eps)` 不变，CISPO 只改 advantage 之后怎么进 loss。
2. 经典 GRPO 用 min-clip 裁目标，越界 token 梯度归零；CISPO 把 ratio 截上界并 detach 成权重，被截 token 仍通过 `per_token_logps` 贡献梯度。
3. `detach` 让截断后的 ratio 变成不带梯度的常数权重，乘到 `logp×advantage` 上；梯度只走 `per_token_logps`，所以越界 token 不会因 clip 而梯度归零。
</details>
