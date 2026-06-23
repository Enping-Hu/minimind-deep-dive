# DPO 的损失函数：为什么是 −logsigmoid(β·logits)

上一节拿到了 chosen/rejected 的逐 token log-prob。这一节看 `dpo_loss` 怎么把它们变成一个标量损失，并讲清那句最容易卡住的 `loss = -F.logsigmoid(beta * logits)` 到底在做什么。

一句话先记住：**DPO 不是在问「chosen 概率大不大」，而是在问「当前 policy 相对 reference，是否更站在 chosen 这一边」。**

源码：`trainer/train_dpo.py`，`dpo_loss`。

## 完整的 dpo_loss

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)            # 防零长度除零
    ref_log_probs    = (ref_log_probs    * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs    = ref_log_probs[:batch_size // 2];   reject_ref_log_probs    = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]; reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios  = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs   - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

## 第一步：mask 平均成序列分数

```python
policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths
```

逐 token log-prob 先乘 assistant mask（只保留回复区域），求和再除以 mask 长度，得到每条回答在 assistant 区域上的**平均** log-prob。为什么平均而不是求和？因为 chosen/rejected 长度可能不同，平均能减少长度差异对分数的直接干扰。`clamp_min(1e-8)` 防止空 mask 除零成 NaN。

batch 前半是 chosen、后半是 rejected（[上一节](01-preference-optimization.md) 的 `cat([x_chosen, x_rejected])`），所以按 `batch_size // 2` 切开。

## 第二步：两层「偏好差」相减

```python
pi_logratios  = chosen_policy_log_probs - reject_policy_log_probs   # policy 的偏好差
ref_logratios = chosen_ref_log_probs   - reject_ref_log_probs       # reference 的偏好差
logits = pi_logratios - ref_logratios
```

- `pi_logratios`：当前 policy 对 chosen 相对 rejected 偏爱多少。
- `ref_logratios`：原始 reference 本来对 chosen 相对 rejected 偏爱多少。
- `logits = pi − ref`：**policy 相对 reference 的「偏好领先量」**。

读法：`logits > 0` 表示 policy 比 reference 更偏向 chosen；`= 0` 表示和 reference 差不多；`< 0` 表示还不如 reference 偏向 chosen、甚至站错边。这也是 DPO 必须保留冻结 reference 的原因——没有它就没有参照系，无法判断「相对原模型变好了多少」。

注意这里的 `logits` 不是分类模型那种类别分数，而是一个相对偏好量。

## 第三步：−logsigmoid 把「越大越好」变成可最小化的 loss

目标是让 `logits` 越大越好。`sigmoid(z)` 在 `z` 大时趋近 1、`z` 小时趋近 0，所以让 `sigmoid(beta * logits)` 趋近 1 就对应「logits 越大越好」；取 log 让目标更平滑，得到 `logsigmoid(beta * logits)`。训练要最小化 loss，而我们想最大化这个量，所以前面加负号：

```python
loss = -F.logsigmoid(beta * logits)
```

单调关系：

| logits | sigmoid(β·logits) | logsigmoid | −logsigmoid = loss |
|---|---|---|---|
| 大（policy 更偏 chosen） | →1 | →0 | →0（loss 小）|
| 负（policy 偏 rejected）| →0 | 很负 | 很大（loss 大）|

一句话：**policy 相对 ref 越偏向 chosen，loss 越小；越偏向 rejected，loss 越大。** 相比「直接判断 chosen 分数是否更大」这种硬判定，`−logsigmoid` 可导、平滑，适合做梯度优化。

`beta`（默认 0.1）是偏好差的缩放系数：越大，`logits` 的变化越强地影响 loss、更新越激进；越小越温和。它对应 DPO 论文里的 KL 约束强度，本书不展开，把它当「更新有多激进」的旋钮即可。

总 loss 仍统一加 MoE 辅助损失：`loss = dpo_loss_val + outputs.aux_loss`（dense 时 aux_loss=0，见 [02-model/06-moe](../02-model/06-moe.md)）。

![DPO 损失源码链](../../images/dpo-loss-source-chain.svg)

## 怎么读 dpo_loss：它是个弱指标

训练时容易盯着 `dpo_loss` 下降，但它不能像 pretrain/SFT 的 loss 那样读。SFT 的 loss 是预测下一个 token 的交叉熵，降了就是预测更准；`dpo_loss` 不是——由上面的推导，它只关心 `logits =（policy 的 chosen−rejected 差）−（ref 的 chosen−rejected 差）` 这个**差值**。

差值变大有两条路：抬高 chosen，或压低 rejected。优化器常走后一条（对数函数的梯度性质让「降」比「升」更省力），极端时 chosen 概率自己也在降，只要 rejected 降得更快，`dpo_loss` 照样往下走。所以：loss 降不一定是模型变好（可能只是在压 rejected，即下文常见误区会强调的那种概率位移），loss 平不一定是没学，低 loss 甚至不保证学到了正确的偏好排序。

那该看什么？业界标准是盯 **reward margin**（chosen 比 rejected 的隐式 reward 高多少）和 **reward accuracy**（chosen reward 大于 rejected 的样本比例，应从 0.5 往 1 爬），再配合 KL（离 ref 漂多远）和下游固定 prompt eval。

这里有个 MiniMind 的现实局限：`train_dpo.py` 只 log 了 `dpo_loss`，没记 margin、accuracy、chosen/rejected 分项。也就是说，单看 MiniMind 给的这条曲线，判断不了 DPO 学得好不好——要判断得回到固定 prompt eval（见 [10-experiments/03](../10-experiments/03-eval-conclusions-sft-vs-rl.md)）。想自己补诊断，可在训练循环里记录 chosen/rejected 的 reward 及其差。

还有个常被误读的现象：`dpo_loss` 往上走。健康的 DPO loss 应缓降；上行通常是 lr 偏大、训练不稳的信号——这正是 MiniMind 把 DPO 默认 lr 钉到 `4e-8`、注释写「建议 ≤5e-8 避免遗忘」的原因（见 [08-training-mechanics/05](../08-training-mechanics/05-optimizer-adamw-scheduler.md)）。但训练不稳不等于模型废了：把 lr 调到 1e-6 重训，loss 明显上漂、方差变大，可固定 prompt eval 里模型照样连贯应答、事实错误也和调前没两样——曲线难看和能力受损是两件事。

## 常见误区

- **「DPO 在最大化 chosen 的概率」**——不准确。它最大化的是 chosen 相对 rejected 的偏好差，而且是**相对 reference** 看的。policy 完全可以让 chosen 概率略降，只要 rejected 降得更多，偏好差仍变大。
- **「logits 是分类分数」**——不是，它是 policy 相对 ref 的偏好领先量。
- **「必须先吃完论文推导」**——不必。先把 chosen/rejected、policy/ref、logits、−logsigmoid 这四层关系对齐就够用。

## 练习

1. `dpo_loss` 第一步为什么对 log-prob 做 mask **平均**而不是求和？`clamp_min(1e-8)` 防什么？
2. `pi_logratios`、`ref_logratios`、`logits` 三者分别表示什么？`logits > 0` 意味着什么？
3. 为什么 `logits` 大时 `-logsigmoid(beta*logits)` 会小？为什么不用「chosen 分数 > rejected 分数」的硬判定？
4. 「DPO 就是最大化 chosen 概率」错在哪？
5. 为什么说 `dpo_loss` 是「弱指标」？健康的 DPO 训练该监控哪些量？MiniMind 的 `train_dpo.py` 在这点上有什么局限？

<details>
<summary>参考答案</summary>

1. chosen/rejected 长度可能不同，平均能减少长度差异对序列分数的干扰；`clamp_min(1e-8)` 防止空 mask（长度 0）除零产生 NaN。
2. `pi_logratios` 是 policy 对 chosen 相对 rejected 的偏好差，`ref_logratios` 是 reference 的同款偏好差，`logits = pi − ref` 是 policy 相对 ref 的偏好领先量；`>0` 表示 policy 比 reference 更偏向 chosen。
3. logits 大 → `sigmoid(β·logits)`→1 → `logsigmoid`→0 → 加负号后 loss→0；硬判定不可导不平滑，无法做梯度优化，`-logsigmoid` 可导平滑。
4. DPO 最大化的是 chosen 相对 rejected、且相对 reference 的偏好差；只要 rejected 概率降得比 chosen 多，偏好差也增大，并不要求 chosen 绝对概率上升。
5. 因为 `dpo_loss` 只看 `（policy 的 chosen−rejected 差）−（ref 的同款差）` 这个差值，优化器可靠压低 rejected（而非抬高 chosen）来降 loss，所以 loss 降不一定变好、loss 平不一定没学、低 loss 不保证学到正确偏好排序。健康与否该看 reward margin、reward accuracy（chosen>rejected 比例往 1 爬）、KL 及下游固定 prompt eval。MiniMind 的局限是 `train_dpo.py` 只 log `dpo_loss`，未记 margin/accuracy/chosen-rejected 分项，单看曲线无法判健康，需以 eval 为准。
</details>
