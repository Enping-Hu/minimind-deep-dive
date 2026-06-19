# B. PPO 重写：从简化版到标准 PPO2

这是 v2→v3 改动最大的一块。[第 7 章的 PPO](../07-ppo-grpo/02-ppo.md) 讲的「五模型、`advantage = reward − value`」是 **MiniMind2** 的简化实现；MiniMind-3 换成了更标准的 PPO2。逐条对照 `trainer/train_ppo.py`。

## 七个维度的对照

| 维度 | v2 PPO（第 7 章讲的） | v3 PPO |
|---|---|---|
| 模型数 | **5**：actor / old_actor / critic / ref / reward | **4**：actor / critic / ref / reward |
| old policy | 独立 `old_actor_model`，`update_old_actor_freq` 周期同步 | **rollout 时记下的 logp 快照** `old_resp_logp`，无独立网络 |
| advantage | `reward − value.detach()`（整条序列一个标量） | **token-level GAE**（`gamma`、`lam`，末 token 给 reward 再反向 bootstrap）|
| advantage 归一 | 无 | **whitening**（减均值除标准差）|
| value loss | `mse_loss(values, rewards)` | **clipped value loss**（`cliprange_value`，取 max 两项）|
| ref/KL | `(actor_logp − ref_logp).mean()` | **k3 无偏 KL 估计** `exp(d) − d − 1`，逐 token |
| 经验复用 | 一遍 | `ppo_update_iters` 遍 × mini-batch 切片，带 `early_stop_kl` |

## 五模型为什么变四模型

v2 有独立的 `old_actor_model` 网络，靠 `update_old_actor_freq` 周期同步，用来算 `ratio`。v3 不再单独建这个网络：**rollout（生成）时就把当时 policy 的 token logp 存成快照** `old_resp_logp`（v3 `train_ppo.py:100`），后面直接 `ratio = exp(new_logp − old_logp)`。少了一个常驻网络，所以五模型变四模型。

注意 v3 用了 `rollout_engine`（生成与采样的封装），它不是「第五个模型」，而是 actor 生成 rollout 的执行器 + logp 快照。剩下的 critic / ref / reward 三个模型职责和 v2 一致。

## token-level GAE 替代标量优势

v2 是「一条回答一个 reward 减一个 value」，整条序列共享一个标量 advantage。v3 把 reward 只挂在最后一个 response token 上，再用 critic 的逐 token value 做 GAE 反向 bootstrap，得到**每个 token 的 advantage**（v3 `train_ppo.py:140-147`）：

```python
gen_len = old_resp_values.size(1); lastgaelam = 0; advs_rev = []
for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advs_rev.append(lastgaelam)
advantages = stack(advs_rev[::-1])           # [B, R]
returns = advantages + old_resp_values        # critic 的回归目标
```

`gamma`（默认 1.0）折扣未来，`lam`（默认 0.95）在偏差/方差间插值。GAE 更细粒度、方差更低。之后还做 whitening（`:149-150` 减均值除标准差）让 advantage 尺度稳定。

## value loss 和 KL 都更标准

- **clipped value loss**（`:201-202`）：取「未裁剪平方误差」和「裁剪到 `old_value ± cliprange_value` 的平方误差」的 max，对应 PPO2 的 value clipping。v2 只是 `mse_loss(values, rewards)`。
- **k3 KL**（`:194`）：`exp(d) − d − 1`（`d = ref_logp − new_logp`）逐 token，是 KL 的无偏估计。v2 用 `(actor_logp − ref_logp).mean()` 的简单差。
- **多轮复用**：v3 把一批 rollout 数据用 `ppo_update_iters` 遍、按 mini-batch 切片重复优化（`:164`），并在 `approx_kl > early_stop_kl` 时提前停（`:188`），更接近标准 PPO 的 sample efficiency 做法。

默认超参也跟着调大（[规模/默认值](05-thinking-scale-removals.md)）：actor/critic lr 8e-8/8e-8 → 3e-7/5e-7，clip_epsilon 0.1 → 0.2。

## 对第 7 章的影响

[第 7 章的 PPO](../07-ppo-grpo/02-ppo.md) 的「五模型框架」「`advantage = reward − value.detach()`」要标注为 **v2 实现**。`ratio`/`clip`/`min(surr1,surr2)` 的核心思想（[第 7 章](../07-ppo-grpo/02-ppo.md) + [08-clipping](../08-training-mechanics/06-clipping.md)）在 v3 仍然通用——v3 `:197` 同样是 `min` 两项 clip——只是 advantage 来源从标量变成了 token-level GAE。

## 练习

1. v2 PPO 几个模型？v3 几个？少掉的那个角色在 v3 靠什么替代？
2. v2 和 v3 的 advantage 怎么算？GAE 的 `gamma`、`lam` 各调什么？
3. v3 的 value loss 和 KL 相比 v2 各「标准」在哪？
4. v3 PPO 里 `ratio`/`clip`/`min` 的思想还在吗？变的是什么？

<details>
<summary>参考答案</summary>

1. v2 五个（actor/old_actor/critic/ref/reward），v3 四个（去掉独立 old_actor）；v3 用 rollout 时记下的 token logp 快照 `old_resp_logp` 当 old_logp。
2. v2：`reward − value.detach()`，整条序列一个标量；v3：token-level GAE，reward 挂末 token、用 critic 逐 token value 反向 bootstrap。gamma 折扣未来、lam 在偏差/方差间插值。
3. value loss：v3 用 clipped value loss（裁到 `old±cliprange_value` 取 max），v2 只是 mse；KL：v3 用 k3 无偏估计 `exp(d)−d−1` 逐 token，v2 用简单差的 mean。
4. 还在，v3 `:197` 仍是 min 两项 clip；变的是 advantage 来源（标量 → token-level GAE），以及多了 whitening、clipped value、k3 KL、多轮复用。
</details>
