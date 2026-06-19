# RoPE：旋转位置编码

self-attention 本身不带顺序——打乱 token 顺序，`QK^T` 的结果不变。位置信息得额外注入。RoPE（Rotary Position Embedding）的做法不是给 embedding 加一个位置向量，而是**按位置把 Q、K 向量旋转一个角度**。这一节讲清楚它怎么旋转、为什么这样能让注意力感知到「相对位置」。

源码：`model/model_minimind.py`，`precompute_freqs_cis`（L124–178）、`apply_rotary_pos_emb` / `rotate_half`（L181–211）。本节没有配图，用公式说明（见仓库 `chapters/图待补.md`）。

## 为什么作用在 Q/K，不在 embedding，也不在 V

注意力分数来自 $\text{scores} = QK^\top/\sqrt{d_k}$。位置信息要影响「关注谁」，就得直接进入这个匹配打分。

- 加在 **embedding** 上：位置信息要先穿过很多层线性变换，到了某一层算 `QK^T` 时未必还以直接的形式存在。
- 加在 **V** 上：V 是被加权汇总的「内容」，改它只改「取出来什么」，不改「关注谁」。
- 加在 **Q、K** 上：直接改变每一对 query-key 的点积，位置立刻参与匹配。

所以 RoPE 选择改 Q、K。

## 二维旋转：RoPE 的核心

把 Q（或 K）向量的维度两两配对，每一对 $(x, y)$ 当成平面上的二维向量。把它旋转角度 $\theta$，标准旋转公式是：

$$
(x', y') = (x\cos\theta - y\sin\theta,\; x\sin\theta + y\cos\theta)
$$

源码不显式构造旋转矩阵，而是用一个技巧凑出来。把上式拆成两项：

$$
(x, y)\cdot\cos\theta \;+\; (-y, x)\cdot\sin\theta
$$

第一项是原向量乘 $\cos\theta$，第二项是「把后半维取负挪到前面」得到的 $(-y, x)$ 乘 $\sin\theta$。后者正是 `rotate_half`（L196–203）：

```python
def rotate_half(x):
    # [x1..x_{d/2}, y1..y_{d/2}]  ->  [-y1..-y_{d/2}, x1..x_{d/2}]
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
```

于是 `apply_rotary_pos_emb`（L209）就是旋转公式的逐元素版：

```python
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

`q * cos` 对应 $(x,y)\cos\theta$，`rotate_half(q) * sin` 对应 $(-y,x)\sin\theta$，相加即旋转后的向量。**`rotate_half` 不是随便换维度顺序，它在构造旋转公式里的 $(-y, x)$ 那一项。**

## precompute_freqs_cis：每个位置的旋转角提前算好

不同维度对用不同的旋转频率（L150）：

$$
\theta_i = \text{rope\_base}^{-2i/\dim},\quad i = 0,1,\dots,\dim/2
$$

`dim` 是单个 head 的维度（默认 `512/8 = 64`）。靠前的维度对频率高、对短距离敏感；靠后的频率低、变化平缓、表达长距离。再把位置索引和频率做外积（L172），得到「每个位置在每个频率下的旋转角」，最后转成 cos/sin 表（L176–177），形状 `[max_position_embeddings, dim]`：

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim//2].float() / dim))  # [dim/2]
freqs = torch.outer(torch.arange(end), freqs)                                   # [end, dim/2]
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)             # [end, dim]
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
```

这套表在 `MiniMindModel.__init__` 里算一次、注册成 buffer（L579–580），前向时按当前位置 `start_pos:start_pos+seq_len` 切片取用（L599–601），不必每步重算。`rope_base` 默认 `1e6`。

## 为什么能体现「相对位置」

关键在于旋转的一个性质：query 在位置 `m` 被转了 `mθ`，key 在位置 `n` 被转了 `nθ`，两者做点积时，旋转带来的影响只剩下角度差 `(m−n)θ`。也就是说，旋转后的 `q_m · k_n` 只依赖**它们相隔多远**，与它们在序列里的绝对位置无关。

这正是 RoPE 的好处：模型在算注意力时，天然感知到「这两个 token 相隔几步」，而不是「这个 token 在第几位」。相对位置通常比绝对位置更有用——同样的语法关系，在句首和句中应该一样成立。

## 长上下文与 YaRN（点到为止）

RoPE 把不同维度绑定到不同频率，序列远超训练长度时，高频部分变化过快、位置模式会失真。长上下文方法（如 YaRN）就从这里入手：对原始频率做缩放/插值，让旋转在更长上下文下更稳。源码 `precompute_freqs_cis` 里 `rope_scaling`（`factor` / `beta_fast` / `beta_slow` / `type: "yarn"`）就是这条路径，默认关闭。细节超出本书范围，知道「长上下文改的是 RoPE 频率、不是 attention 主体」即可。

## 练习

1. RoPE 为什么作用在 Q/K 上，而不是加到 embedding 或 V 上？
2. `q * cos + rotate_half(q) * sin` 在数学上等价于什么操作？`rotate_half` 起什么作用？
3. 为什么说 RoPE 编码的是相对位置而非绝对位置？
4. 不同维度对的旋转频率一样吗？这样设计有什么用意？

<details>
<summary>参考答案</summary>

1. 位置要影响 `QK^T` 的匹配打分（关注谁），所以加在参与打分的 Q/K 上；embedding 加位置会被后续线性层稀释，V 只是被汇总的内容、改它不影响关注谁。
2. 等价于把每一对维度当二维向量按角度 θ 旋转：`q*cos` 是 `(x,y)cosθ`，`rotate_half(q)*sin` 是 `(-y,x)sinθ`，相加即旋转公式 `(x cosθ − y sinθ, x sinθ + y cosθ)`。`rotate_half` 负责构造 `(-y, x)` 那一项。
3. 因为 query 在位置 m 转 mθ、key 在位置 n 转 nθ，点积后只剩角度差 (m−n)θ，结果只依赖两者间距，与绝对位置无关。
4. 不一样，`θ_i = rope_base^(−2i/dim)` 随维度递减：靠前维度对频率高、敏感短距离，靠后频率低、表达长距离，让模型同时建模远近不同尺度的位置关系。
</details>
