# PPO：ratio / clip / critic / old actor

PPO（Proximal Policy Optimization）是这个项目里最完整的在线 RL，也是理解 GRPO/SPO 的参照系。它的关键不只是「往哪个方向改」（advantage 决定），还要控制「这一步改多大」（ratio/clip 决定）。这一节讲 PPO 的五个模型角色，重点拆 `ratio / clip / min` 这套稳定化机制。

源码：`trainer/train_ppo.py`，`ppo_train_epoch`、`CriticModel`。

> 本节讲的是 MiniMind2 的 PPO（五模型、简化优势）。MiniMind-3 已重写为标准 PPO2（四模型、token-level GAE），见 [第 9 章](../09-minimind2-vs-3/03-ppo-rewrite.md)。

## 五个模型角色

| 模型 | 作用 | 是否训练 |
|---|---|---|
| `actor_model` | 当前 policy，要优化的模型 | 训练 |
| `old_actor_model` | 旧 policy 快照，提供更新前的概率参照 | 周期同步，不反传 |
| `critic_model` | value model，估计 baseline | 训练 |
| `ref_model` | 冻结参考，约束行为漂移（KL） | 冻结 |
| `reward_model` | 冻结奖励模型，给 response 打分 | 冻结 |

分工要记清：**reward_model 给分，critic 估 baseline，ref_model 防漂移，old_actor 量更新幅度。** `CriticModel` 继承自 `MiniMindForCausalLM`，把语言模型头换成输出标量 value 的头（见下方折叠的源码细节）。

## advantage = reward − critic value

```python
rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)
values = critic_model(input_ids=gen_out, ...)[..., last_indices]
advantages = rewards - values.detach()
```

critic 估「这个 prompt 正常能拿几分」，实际 reward 减去它就是 advantage：比预期好为正、差为负（[01-rl-overview](01-rl-overview.md) 的 baseline 思想，PPO 的 baseline 来自 critic）。`.detach()` 是因为这一步只用 value 当基准、不让 advantage 的梯度流回 critic（critic 另有自己的 value_loss 回归 reward）。

## ratio：新旧 policy 的概率比

PPO 不只看当前 actor 输出了什么，还看它相比更新前改了多少。`old_logp` 是旧 policy 对同一条 response 的 log-prob，`actor_logp` 是当前 policy 的：

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

`clip_epsilon`（v2 默认 0.1）把 ratio 截在 `[0.9, 1.1]`。clip 不改更新方向，只限更新幅度。

但为什么取 `min`，不直接拿夹好的 surr2 当目标？关键在 clamp：ratio 一出界，输出就成了常数，梯度归零，而且两边都归零。advantage > 0（想推高概率）时，出界的两种情形该区别对待：

- ratio 冲过头（远大于 1）：概率推够了，梯度归零正合适；
- ratio 反而掉到 1 以下（好回答的概率被压低了）：这才是要救的情况，得靠梯度把它拉回来。

只用 surr2，这两种都被一刀切成零梯度，第二种就卡死了。`min` 取两者中更不利的那个，刚好把它们分开：冲过头时夹住的 surr2 更小，选它，梯度归零；掉错方向时没夹的 surr1 更小，选它，梯度还在，ratio 被拉回信任区。

| ratio（advantage > 0） | 只用 surr2 | min(surr1, surr2) |
|---|---|---|
| 掉到 1−ε 以下 | 梯度 0，卡死 | 选 surr1，梯度活，拉回 |
| 区间内 | 梯度活 | 梯度活 |
| 冲过头（> 1+ε） | 梯度 0 | 选 surr2，梯度 0 |

差别只在第一行。advantage < 0 完全对称，错方向变成 ratio > 1+ε，同样靠 min 选 surr1 拉回。

所以 clip 限单步幅度，min 让裁剪只发生在朝有利方向冲过头那侧，往回修正始终放行。末尾负号同前：要最大化 surrogate objective，写成 loss 就加负号。

![PPO ratio/clip 流程](../../images/ppo-ratio-clip-flow.svg)

## 完整 loss

PPO 的总损失还包括 critic 的 value_loss（回归 reward）和对 ref_model 的 KL 惩罚（防漂移）：`policy_loss + value_loss + KL`。critic 自己靠 `value_loss = F.mse_loss(values, rewards)` 训练：让 value 去回归 reward；回归得越好，value 越接近「这个 prompt 大致能拿几分」，正好当 advantage 的 baseline（前面 advantage 一节减掉的就是它）。`ppo_train_epoch` 把生成、打分、算优势、算三部分 loss、更新串起来。`old_actor_model` 按 `update_old_actor_freq` 周期同步成当前 actor，提供下一轮的 `old_logp`。

<details>
<summary>源码细节：rollout 到 advantage 的张量链</summary>

上面是机制骨架。下面补几个读 `ppo_train_epoch` 时容易卡住的张量级细节，都贴真实片段。

**1. CriticModel：复用基座 + value_head，不是 lm_head**

`CriticModel` 继承 `MiniMindForCausalLM`，但 forward 不走 lm_head（那是投到词表），而是复用基座的最终 `norm`、再接一个输出标量的 `value_head`：

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)   # 替换 lm_head：hidden → 标量 value

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])           # 注意：outputs[0] 已 norm 过，这里又 norm 一次
        values = self.value_head(hidden_states).squeeze(-1)   # [B, P+R]：每个位置一个 value
```

两个读源码才会注意到的点：(1) `self.model`（即 `MiniMindModel`）的 forward 返回的是**三元组** `(hidden_states, presents, aux_loss)`，所以要取 `outputs[0]` 才是隐藏状态；(2) `outputs[0]` 其实**已经在 `MiniMindModel` 内部过了一次最终 `norm`**（[03-pretrain/02](../03-pretrain/02-forward-to-loss.md) 里基座 forward 的最后一步就是 `self.norm`），这里再写 `self.model.norm(outputs[0])` 等于**又归一化了一次**，value_head 吃的是这个二次归一化后的 hidden。别误以为「基座没 norm、critic 来补」——基座 norm 过了，这是第二次。

注意它输出的是**逐位置** value `[B, P+R]`，不是一个标量——怎么压成每条回答一个 baseline，见第 2 点。

**2. last_indices：为什么取「最后一个非 pad 位置」的 value**

critic 给了每个位置的 value，但 advantage 只需要每条回答一个标量 baseline。代码取每条序列**最后一个有效（非 pad）token** 的 value：

```python
full_mask = (gen_out != tokenizer.pad_token_id).long()   # [B, P+R]，非 pad 为 1
values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
values = values_seq[torch.arange(values_seq.size(0)), last_indices]     # [B]
```

`full_mask * arange` 把非 pad 位置替换成它的列索引、pad 位置压成 0，`argmax` 就取到**最后一个非 pad 的列号**。为什么取最后一个？自回归模型里，最后一个 token 的隐藏状态浓缩了整条序列的上下文，用它的 value 当「这条回答的预期得分」最合理。这也是 v3 用 token-level GAE 替代它的动机（[第 9 章](../09-minimind2-vs-3/03-ppo-rewrite.md)）：v2 整条只取一个 value，粒度粗。

**3. left padding：为什么 prompt 左填充**

tokenize prompt 时用了 `padding_side="left"`：

```python
enc = tokenizer(prompts, ..., padding=True, padding_side="left")  # input_ids: [B, P]
prompt_length = enc.input_ids.shape[1]
```

batch 里 prompt 长短不一，左填充让所有 prompt 的**右端对齐**，于是 `generate` 从同一列开始续写、`prompt_length` 对整个 batch 是同一个值。若用右填充，短 prompt 后面跟 pad，生成会接在 pad 之后，response 区间逐条不同，后面 `resp_mask` 就没法用一个 `prompt_length` 统一切。

**4. final_mask 与 labels 错位：只统计 response 区的有效 token**

actor_logp 要的是「response 部分」每个 token 的 log-prob 之和，不含 prompt、不含 pad。先 [shift](../08-training-mechanics/02-logits-to-logprob.md) 取 token log-prob，再用两层 mask 筛：

```python
labels = gen_out[:, 1:].clone()                              # [B, P+R-1]，错位一位：位置 t 预测 t+1
logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
seq_len = gen_out.size(1) - 1
resp_mask = torch.arange(seq_len).unsqueeze(0) >= prompt_length - 1   # 只保留 response 区
final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))        # 再去掉 pad
actor_logp = (logp_tokens * final_mask).sum(dim=1)          # [B]，response log-prob 求和
```

拿一条样本走一遍最直观——左填充后 `prompt_length=5`，response 是 `r1 r2` 后面跟了个 pad。先记住 `labels = gen_out[:, 1:]` 把整条**左移一位**（为和 `logits[:, :-1]` 的 next-token 对齐），所以下面都按 labels 的下标看：

```text
labels   : PAD  p1  p2  p3  r1  r2  PAD     (= gen_out[:,1:]，整条左移一位)
resp_mask:  0    0   0   0   1   1   1      (下标 ≥ prompt_length−1 = 4)
~pad     :  1    1   1   1   1   1   0      (非 pad)
final    :  0    0   0   0   1   1   0      (相与 → 只剩 r1 r2)
```

阈值写 `prompt_length - 1` 而不是 `prompt_length`，就是因为这一位左移：response 起点从 `gen_out` 里的 `prompt_length` 跟着前移到 labels 里的 `prompt_length - 1`，不是随手减一。`final_mask` 再 `& ~labels.eq(pad)` 去掉 response 里的 pad（例中末位）。最后 `(logp_tokens * final_mask).sum` 把留下的 token log-prob 相加 = 这条 response 的总 log-prob；用 `.sum` 而非 mean，对应 [08-training-mechanics/03](../08-training-mechanics/03-token-to-sequence-objective.md)：PPO 要整条 response 的序列 log-prob。

**5. kl_ref 是简化约束，不是严格 KL**

PPO 对 ref_model 的「KL 惩罚」其实是 actor 与 ref 的 log-prob 差的均值：

```python
kl_ref = (actor_logp - ref_logp).mean()   # scalar
```

这只是 `E[log π_actor − log π_ref]`，是真实 KL 散度的一个简化代理（严格 KL 还要对分布求期望）。够用、便宜，但有偏。对比 [第 9 章](../09-minimind2-vs-3/03-ppo-rewrite.md)：v3 改用 k3 无偏估计 `exp(d) − d − 1`，更接近真实 KL。读 v2 时别把这个 `kl_ref` 当成严格 KL。

</details>

## 常见误区

- **「ratio 越大越好」**——不对，ratio 太大说明更新过猛，正是 PPO 要限制的。
- **「clip 改变更新方向」**——不，clip 只限幅度，方向由 advantage 决定。
- **「必须先推完 policy gradient 数学」**——不必，先把 ratio/clip/min 三者和代码对齐即可。
- **「critic 学准了（value≈reward），advantage 就归零、没法学了」**——不会。critic 估的是「这个 prompt 平均拿几分」的**期望**，advantage 是「这条回答比平均好/坏多少」，信号活在偏差里；真要处处归零，也只是收敛停更、不是错误更新。不过 v2 这套确实偏弱：它的 critic 吃整条 `gen_out`、`value_loss = mse(value, reward)` 直接拟合本条的分（不是只看 prompt 的状态值），逼近完美时 advantage 会塌（实际靠 critic 小、在线滞后、reward 有噪声才没塌）。**这也是 v3 把 PPO 重写成 token-level GAE、以及 GRPO 改用组内均值当 baseline 的原因之一**（见 [第 9 章](../09-minimind2-vs-3/03-ppo-rewrite.md)、[03-grpo](03-grpo.md)）。

## 练习

1. PPO 的五个模型各是什么角色？哪些训练、哪些冻结？
2. `ratio = exp(actor_logp - old_logp)` 为什么等于新旧 policy 的概率比？ratio 和 advantage 各表达什么？
3. 只用 `ratio * advantages` 会有什么问题？为什么用 `min(surr1, surr2)` 而不是直接拿 clamp 后的 surr2 当目标？
4. advantage 算式里的 `values.detach()` 为什么要 detach？
5.（源码细节）critic 输出逐位置 value，为什么 advantage 只取「最后一个非 pad 位置」的 value？prompt 为什么要 left padding？

<details>
<summary>参考答案</summary>

1. actor（训练的 policy）、old_actor（旧快照，量更新幅度，不反传）、critic（估 baseline，训练）、ref（冻结，KL 防漂移）、reward（冻结，打分）。
2. `log(π_θ) − log(π_old) = log(π_θ/π_old)`，取 exp 得概率比；ratio 表达这一步改了多少（幅度/方向），advantage 表达这条回答该不该强化。
3. 只用 surr1，单步可能冲太远、训练不稳。若只用 clamp 后的 surr2，ratio 一旦出界梯度两边都为 0，跑到错方向（好回答概率反被压低）时也卡死、回不来。min 永远取更小（更不利）的目标，让裁剪只在「朝有利方向冲过头」那侧把梯度归零，在「错方向」那侧选中没夹的 surr1 保住梯度、把 ratio 拉回信任区。
4. advantage 只把 critic value 当基准，不应让 advantage 的梯度流回 critic（critic 由自己的 value_loss 训练），所以 detach。
5. 最后一个非 pad token 的隐藏状态浓缩了整条序列上下文，用它的 value 当「这条回答的预期得分」最合理（v2 整条只取一个，粒度粗，v3 改 token-level GAE）；left padding 让 batch 内所有 prompt 右端对齐，generate 从同一列续写、`prompt_length` 全 batch 统一，response 区间才能用一个阈值切。
</details>
