# 延伸：KV cache 压缩——从 MiniMind 的 GQA 往两头看

[04-gqa](04-gqa.md) 讲了 MiniMind 为什么用 GQA、`repeat_kv` 怎么把 KV 头广播成 Q 头，[04-inference/01](../04-inference/01-kv-cache-and-generate.md) 讲了 KV cache 为什么能省重复计算。这一节把这两件事接起来，放进一条更大的脉络：从 MHA 到 MQA、GQA 再到 MLA，注意力结构这些年怎么围绕 KV cache 一步步改，每一步在解决前一步的什么问题。

本节是**延伸 survey**。它的锚点很硬：MiniMind v2 就是这条线上 GQA 那一档的活标本——`num_attention_heads=8`、`num_key_value_heads=2`、`n_rep=4`（`model_minimind.py:20–22`），源码里能精确看到「共享多少」这个旋钮拧到了哪。读它能回答一个部署时绕不开的问题：**上下文越来越长时，为什么先爆的常常不是模型参数，而是 KV cache？该怎么把它压下去？**

一句话主线：**这条演进线一直在调同一个旋钮，即允许多少个 query 头共享同一份 K/V。共享越多缓存越省，共享越狠表达力越容易受损。**

## 为什么 KV cache 会成为瓶颈

自回归生成每步只新增一个 token，但当前 query 要读**全部历史** K/V（[04-inference/01](../04-inference/01-kv-cache-and-generate.md) 讲过：因为 causal attention 下历史 K/V 不会被未来改写，可以缓存复用）。于是历史 K/V 必须一直留在显存里，越攒越大。

单层 cache 的典型形状是 `[batch, kv_heads, seq_len, head_dim]`，K、V 各一份。FP16 下峰值显存约：

$$2 \times 2 \times b \times l \times n_{kv\_head} \times d_{head} \times \text{seq\_len}$$

（第一个 2 是 K/V 两份，第二个 2 是 FP16 字节数，`l` 是层数）。序列一长、batch 一大，这个量会迅速逼近甚至盖过参数显存。所以有个重要结论：**训练时注意力更像算力问题，推理时它常常更像缓存问题。**

公式里 `n_kv_head` 那一项是关键——它直接乘在缓存大小上。MHA→MQA→GQA→MLA 干的事，本质就是想办法把这一项（或它背后的东西）压小。

## MHA：表达力拉满，缓存最贵

MHA（原始多头注意力）里每个 head 有独立的 Q/K/V 投影，可以分别学局部搭配、句法关系、远距离依赖、格式 token 等不同模式。强在每个 head 都够自由，弱点也来自这里：每个 head 都要存自己那份 K/V，`n_kv_head = n_head`，推理时 KV cache 最贵。

## MQA：所有 Q 头共享一份 K/V

MQA 是最直接的降缓存办法：保留多个 query 头，但让它们**共用同一组 K/V**，相当于很多人提问、查的是同一本资料册。`n_kv_head = 1`，缓存从「每个 head 各存一份」变成「全体共享一份」，显存和带宽压力明显下降。

代价是所有 head 读到的候选信息源过于一致，表达空间被压窄，效果容易掉。所以 MQA 偏强效率导向，工业界更常用的是下面的折中。

## GQA：分组共享，MiniMind 就停在这

GQA 把 query 头分成若干组，**每组共享一套 K/V**。缓存成本显著低于 MHA，又保留了组间的多样性，是 MHA 和 MQA 之间的折中，也是当前多数开源 LLM（含 MiniMind）的选择。

MiniMind v2 的实现把这一档讲得很具体。看投影层维度（`model_minimind.py:248–250`）：

```python
self.q_proj = nn.Linear(hidden, num_attention_heads * head_dim)     # 8 头
self.k_proj = nn.Linear(hidden, num_key_value_heads * head_dim)     # 2 头
self.v_proj = nn.Linear(hidden, num_key_value_heads * head_dim)     # 2 头
```

Q 投出 8 个头，K/V 只投出 2 个头——**缓存里只存 2 组 K/V，不是 8 组**，这就是省的地方。那 8 个 Q 头怎么和 2 组 K/V 对齐？靠 `repeat_kv`（`model_minimind.py:214–220`）：

```python
def repeat_kv(x, n_rep):  # n_rep = 8 // 2 = 4
    # [b, seq, 2, d] → [b, seq, 8, d]，每组 K/V 复制 4 份
    return x[:, :, :, None, :].expand(...).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
```

关键在于 `repeat_kv` 是在**读缓存时**才把 2 组广播成 8 份，缓存本身只存 2 组。`n_rep = n_local_heads // n_local_kv_heads = 4`（`:244`）就是「每组 K/V 被几个 Q 头共享」。把这几个数连起来：`num_key_value_heads` 从 8 调到 2，KV cache 直接降到 1/4，而 Q 仍是 8 头、组间保留多样性。**这就是 GQA 那个旋钮的真实刻度。**

（顺带看两个极端：`num_key_value_heads = 8` 退化成 MHA，`= 1` 退化成 MQA。GQA 是中间地带，MiniMind 取了 2。）

## MLA：不再问共享几组，改问缓存里存什么

MHA→MQA→GQA 一直在调「共享多少组 K/V」，MLA（Multi-head Latent Attention，DeepSeek-V2 提出）换了个问法：既然完整多头 K/V 太贵，就**不缓存完整 K/V，而是先把它们压成更小的低秩 latent**，参与注意力时再上投影展开。缓存成本不再跟「完整每头 K/V」绑定，而取决于压缩后的 latent 维度。

这里最难的是 RoPE。低秩压缩想把若干投影矩阵吸收到一起、少算显式步骤，但 RoPE 会把内容信息和位置信息耦合起来，不额外设计就很难既省 cache 又保住位置编码。MLA 的解法是**解耦**：把 Q/K 拆成内容分支和位置分支，内容分支走低秩压缩，位置分支单独施加 RoPE，最后拼回 attention（本质是 content score 加上 RoPE score）。

有两个容易被忽略的现实：

- **MLA 省的主要是 cache，不一定省计算。** 推理时为了真做 attention，压缩 latent 仍要上投影回各 head 的 K/V。它的收益主要在 Generation 阶段——单步 FLOPs 不高、更常被内存带宽和 cache 访问拖住，所以哪怕算子更复杂，只要显著减小每步读写的缓存量就更快。MLA 更像「用适度额外计算换掉更贵的内存流量」。
- **论文的理论 cache 优势不一定在每个实现里原样落地。** 某些 `DeepSeekV2Attention` 代码路径会先把 `compressed_kv` 展开成标准 K/V 再写进 cache，那样缓存里存的仍是展开后的多头 K/V。把「算法设计上的优势」和「当前代码路径有没有真落地」分开看，是必要的。

## 落回 MiniMind

MiniMind v2 和 v3 都停在 **GQA**，没上 MLA。放进这条谱系看，它的选择很清楚：GQA 是「缓存省 4 倍、表达力几乎不损、实现只多一个 `repeat_kv`」的甜点档，对一个 26M 的教学模型完全够用；MLA 那套低秩压缩 + RoPE 解耦的复杂度，只有在 hidden 很大、上下文很长、缓存真成瓶颈时才划算。

要在 MiniMind 上验证这条线，其实不用改代码，直接调 `num_key_value_heads`：设成 8（=`num_attention_heads`）就退回 MHA，设成 1 就变 MQA，看 KV cache 显存和生成质量怎么随之变化，就是这条谱系最小的动手实验。

## 一张表看懂演进

| 方法 | 缓存的 K/V 组数 | 相对前一步的关键改动 | 代价 | 代表 |
|---|---|---|---|---|
| MHA | `n_head`（每头一组） | —（基线：表达力最强） | 缓存最贵 | 原始 Transformer |
| MQA | 1（全体共享） | 把共享**拉满** | 表达力被压窄、效果易掉 | PaLM 等 |
| GQA | `n_kv_head`（分组共享） | 折中：**按组**共享 | 需定组数 | **MiniMind**、LLaMA2/3 |
| MLA | 压缩 latent（非完整 K/V） | 改**缓存里存什么**，低秩压缩 | RoPE 要解耦、算子更复杂 | DeepSeek-V2/V3 |

读这张表的方式：前三行是同一个旋钮的不同刻度（共享几组 K/V），MLA 换了维度（不存完整 K/V 而存压缩 latent）。MiniMind 把旋钮拧在 GQA（8 头 Q、2 组 KV），是缓存与表达力的甜点。

## 常见误区

- **「GQA 会把 8 组 K/V 都存下来再合并」**——不。缓存只存 2 组，`repeat_kv` 是读缓存时才广播成 8 份，省的正是缓存那一份。
- **「MLA 既省缓存又省计算」**——不一定。它省的是缓存（内存流量），推理时 latent 仍要上投影展开，算子甚至更复杂；收益主要在被带宽卡住的 Generation 阶段。
- **「MiniMind 用 MHA」**——不。v2/v3 都是 GQA（`num_key_value_heads=2 < num_attention_heads=8`），`repeat_kv` 就是 GQA 的标志。

## 练习

1. KV cache 的显存公式里，哪一项是 MQA/GQA 直接压小的？为什么压它就能省缓存？
2. MiniMind 的 `num_key_value_heads=2`、`num_attention_heads=8`，`n_rep` 是多少？它的含义是什么？
3. `repeat_kv` 在缓存的哪个环节起作用？为什么说它不增加缓存开销？
4. MLA 和 GQA 压缩缓存的思路有什么本质不同？MLA 处理 RoPE 时为什么要做解耦？
5. 想在 MiniMind 上把 GQA 退化成 MHA 或 MQA，该改哪个配置、分别设成几？

<details>
<summary>参考答案</summary>

1. `n_kv_head`（缓存的 K/V 头/组数）。缓存大小正比于它，MHA 里它等于 `n_head`，MQA 压到 1、GQA 压到 `n_kv_head`，所以直接减小缓存体积。
2. `n_rep = n_head // n_kv_head = 8 // 2 = 4`，含义是每组 K/V 被 4 个 Q 头共享（读缓存时复制 4 份匹配 Q）。
3. 在**读缓存做 attention 之前**，把缓存里的 `n_kv_head` 组 K/V 广播成 `n_head` 份匹配 Q。它不增加缓存开销，因为缓存里始终只存 `n_kv_head` 组，广播是临时展开、不写回缓存。
4. GQA 在「共享几组完整 K/V」上做文章，缓存的仍是完整 K/V；MLA 改「缓存里存什么」，存低秩压缩 latent、用时再上投影。RoPE 要解耦是因为它把内容和位置耦合，低秩压缩想合并投影矩阵就会和 RoPE 冲突，所以拆成内容分支（走压缩）和位置分支（单独施加 RoPE）再拼回。
5. 改 `num_key_value_heads`：设成 8（等于 `num_attention_heads`）退化成 MHA，设成 1 退化成 MQA。
</details>
