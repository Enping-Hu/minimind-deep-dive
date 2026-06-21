# GQA 与 repeat_kv

上一节的 attention 里，Q 是 8 个头，K/V 只有 2 个头。这就是 GQA（Grouped Query Attention）：让 query 头数多于 key/value 头数。这一节讲它为什么能这么做、`repeat_kv` 怎么把头数对齐、以及它省在哪。

源码：`model/model_minimind.py`，`Attention.__init__`、`repeat_kv`。

![MHA / GQA / MQA 对比](../../images/gqa-diagram.svg)

## 头数怎么设

`Attention.__init__`：

```python
self.n_local_heads = args.num_attention_heads          # Q 头数，默认 8
self.n_local_kv_heads = self.num_key_value_heads       # KV 头数，默认 2
self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 KV 头被几个 Q 头共享 = 4
self.head_dim = args.hidden_size // args.num_attention_heads  # 64
```

要求 `num_attention_heads % num_key_value_heads == 0`。这直接体现在三个投影层的输出维度上：

```python
self.q_proj = nn.Linear(hidden_size, num_attention_heads  * head_dim, bias=False)  # 8×64=512
self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)   # 2×64=128
self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)   # 2×64=128
```

Q 投影到 512 维（8 头），K/V 只投影到 128 维（2 头）。

## MHA / GQA / MQA 是一条谱

按 Q 头数和 KV 头数的关系：

| | 关系 | 特点 |
|---|---|---|
| MHA | Q heads = KV heads | 表达最强，KV 开销最大 |
| GQA | Q heads > KV heads > 1 | 折中（MiniMind 默认 8 : 2） |
| MQA | KV heads = 1 | KV 开销最省，共享最强 |

GQA 处在中间：保留较多 query 视角，但让多个 query 头共享同一组 K/V。

## repeat_kv：共享后展开，不是造新信息

K/V 只有 2 个头，但算 `QK^T` 时要和 8 个 Q 头对齐。`repeat_kv`负责把 2 个头复制成 8 个：

```python
def repeat_kv(x, n_rep):  # x: [B, T, num_kv_heads, head_dim]
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))  # [B, T, 8, head_dim]
```

`n_rep=4` 时，`[B, T, 2, 64] → [B, T, 8, 64]`，每个 KV 头被复制 4 份。注意它用 `expand`（只建视图、不真正占内存）再 `reshape`，本质是**共享**——不会凭空学出新的 K/V 表示。

## 共享 K/V 为什么不退化成单头

因为参与匹配的不只有 K/V，还有 Q。即使 4 个 Q 头共享同一组 K/V：

- 这 4 个 Q 头的 `q_proj` 不同，算出的 query 不同；
- 对同一组 K/V，它们会给出不同的打分和加权。

也就是说，**共享的是「被查询的对象」（K/V），不共享的是「查询视角」（Q）**。所以多头注意力的差异性大部分还在。

## 省在哪：KV cache

GQA 最直接省的是 K/V 侧：

- `k_proj` / `v_proj` 参数和中间激活更少（128 维 vs 512 维）；
- 推理时 KV cache 只需缓存 2 个头的 K/V，而不是 8 个。

第三点最关键。长上下文推理时，KV cache 占用随序列长度线性增长，是显存大户（见 [04-inference/01-kv-cache-and-generate](../04-inference/01-kv-cache-and-generate.md)）。KV 头数减到 1/4，这部分占用也降到约 1/4。这就是为什么现代大模型几乎都用 GQA——它主要是**推理部署**的优化，不只是训练效率。

<details>
<summary>源码细节：None 索引、expand 的 stride、reshape 何时物化</summary>

正文说 `repeat_kv` 用 `expand` 不占内存，这里把「不占内存」的边界讲精确——`expand` 确实不占，但紧跟的 `reshape` 会物化（贴真实片段）。

```python
return (x[:, :, :, None, :]                                  # [B,T,kv,d] → [B,T,kv,1,d]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)  # → [B,T,kv,n_rep,d]
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))  # → [B,T,kv*n_rep,d]
```

三步拆开：

1. **`x[:, :, :, None, :]`**：`None` 索引等价 `unsqueeze`，在第 3 维插一个长度 1 的新维 → `[B, T, kv, 1, d]`。
2. **`.expand(..., n_rep, ...)`**：把那个长度 1 的维扩成 `n_rep`。`expand` 不复制数据——它把这一维的 **stride 设为 0**，即 `n_rep` 份「视图」都指向同一块底层内存。这一步零拷贝、零额外显存。
3. **`.reshape(..., kv*n_rep, ...)`**：把 `kv` 和 `n_rep` 两维合并成一维。问题是 expand 出来的张量内存**不连续**（stride 0 的维没法直接展平），所以 `reshape` 这一步会触发**实际复制**，把数据物化成连续的 `[B, T, kv*n_rep, d]`。

所以精确的说法是：**`expand` 阶段省内存，`reshape` 合并维度时才真正复制出 `n_rep` 份 K/V**。比「先 unsqueeze 再 `repeat`」省的是中间步骤的临时显存，但最终参与 `QK^T` 的 K/V 仍是物化的完整 `[B, 8, T, d]`（毕竟矩阵乘法需要连续内存）。这也是为什么说 GQA 省的是 **KV cache 和投影**（K/V 只存/投 2 头），而不是省 attention 计算那一步——算分时该有的 8 头还是得展开。

（v3 的 `repeat_kv` 与此一字不差，GQA 机制两版相同；v3 只是在 Q/K 投影后多了 QK-Norm，见 [第 9 章](../09-minimind2-vs-3/02-architecture-diffs.md)。）

</details>

## 练习

1. `num_attention_heads=8`、`num_key_value_heads=2` 时，`q_proj` 和 `k_proj` 的输出维度各是多少？`n_rep` 是几？
2. `repeat_kv` 是在学习新的 K/V 吗？它做了什么？
3. 多个 Q 头共享同一组 K/V，为什么不会退化成单头注意力？
4. GQA 最关键省的是什么开销？为什么对长上下文推理尤其重要？
5.（源码细节）`repeat_kv` 用 `expand` 真的全程不占内存吗？`expand` 和紧跟的 `reshape` 各自做了什么？

<details>
<summary>参考答案</summary>

1. `q_proj` 输出 `8×64=512`，`k_proj` 输出 `2×64=128`；`n_rep = 8/2 = 4`。
2. 不是。它用 `expand`+`reshape` 把已有的 2 个 KV 头各复制 4 份，凑到 8 个以对齐 Q 头，不产生新表示。
3. 因为每个 Q 头的 `q_proj` 不同、query 不同，对同一组共享 K/V 仍会给出不同打分；共享的是被查询对象，不是查询视角。
4. KV cache。推理时每个 token 要缓存每层 K/V，GQA 把 KV 头数减少，缓存随之成比例下降，长上下文/大模型部署时显存收益最明显。
5. 不是全程不占。`expand` 把长度 1 的维 stride 设为 0、零拷贝建视图；但 expand 后内存不连续，`reshape` 合并 `kv×n_rep` 维时会触发实际复制、物化成连续的完整 K/V。省的是 KV cache 与投影，不是 attention 计算。
</details>
