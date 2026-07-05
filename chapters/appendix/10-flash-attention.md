# 延伸：FlashAttention——同样的结果，更少的显存往返

主线 [02-attention](../02-model/02-attention.md) 里 MiniMind 的 Attention.forward 有两条路径并排写着：一条是 `F.scaled_dot_product_attention`（Flash 路径），一条是显式的 `QK^T/√d → mask → softmax → @V`（标准路径）。附录 [01 进阶入口](01-advanced-pointers.md) 说了 Flash 是 SDPA 的底层、和标准实现**数学等价**但更省显存，并把 IO-aware 分块、online softmax 的细节留给了这一篇。

这一篇就补那两个细节：**为什么它更快，以及它凭什么在分块之后还能算出和整行 softmax 一模一样的结果。** 读完你会明白，MiniMind 源码里那两条并排的路径，其实是「同一个数学、两种计算顺序」。

一句话主线：**FlashAttention 不减少任何一次乘加，它减少的是把中间大矩阵在显存和片上缓存之间来回搬的次数——把 attention 从「算得起但搬不动」变成「搬得动」。**

## 瓶颈不在算力，在显存往返

先纠一个直觉。大家常把 attention 慢归到「计算量是 `O(N²)`」，但在现代 GPU 上，attention 的不少子步骤其实是**带宽受限**（memory-bound）：速度先被显存带宽卡住，而不是先被算力卡住。大矩阵乘法确实是算力受限，但 softmax、mask、dropout、reduction 这些步骤，瓶颈往往是数据搬运。

关键在 GPU 的两级存储：片上 **SRAM** 很小但极快，片外 **HBM**（显存）很大但慢得多。标准 attention 的问题是：它把 `S = QK^T`、mask 后的 score、softmax 后的概率矩阵，一个个**完整算出来、写回 HBM，下一步再读回来**。序列长 `8K` 时，score 矩阵是 `8192×8192`、数千万元素，单是把这块大对象反复搬进搬出 HBM 就很贵。

所以 FlashAttention 的切入点不是「我少算了多少」，而是「能不能少去 HBM 来回搬同一批中间结果」，这就是它名字里 **IO-Aware** 的含义。它反复强调自己是 **exact attention**：不像稀疏注意力删连接、也不像低秩近似换目标，而是把**完全等价**的 attention 重排成更适合 GPU 内存层级的计算顺序,同样的结果，用更低的 IO 代价算出来。

这也修正一个常见误解：不是所有「理论计算更少」的 attention 变体都会在真实硬件上更快；反过来，一个方案哪怕乘加没少多少，只要砍掉了中间矩阵反复写回 HBM 的次数，在现代 GPU 上完全可能更快。

## 难点：softmax 的分母不天然可分块

要把大矩阵留在 SRAM 里算完、不落回 HBM，就得把 Q/K/V 切成小块逐块处理。但这里卡着一个数学难点：**softmax 的分母依赖同一行的全部元素**。

把一整行 score 切成多个 block，每个 block 局部都能算 exponent，但真正的归一化分母 `Σexp` 需要整行的全局信息。只要 softmax 还是「必须拿到整行分母才能继续」，attention 就很难改成 tile-by-tile 的 kernel。FlashAttention 能成立，全靠两层铺垫解决这个问题。

**第一层:safe softmax（数值稳定）。** softmax 常用写法先减掉行最大值 `m`：

$$\operatorname{softmax}(x)_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}$$

减最大值后指数恒 ≤ 0、不会溢出（`exp(x)` 在 `x` 超过约 89 时对 fp32/bf16 就变 inf；fp16 阈值低得多、约 11 就溢出，这也是 fp16 训练更依赖 loss scaling 的原因之一），而结果不变。

**第二层:online softmax（分块可组合，这是关键）。** 只要能在遍历 block 的过程中一直维护「到当前为止的全局最大值」和「对应的指数和」，就不必先把整行 materialize 再做 softmax。

举个例子：第一块 `[10, 11]`、第二块 `[3, 20]`。只看第一块会以为最大值是 `11`，看到第二块才发现全局最大值是 `20`——这时第一块之前基于 `11` 算出的指数和，必须整体乘上 `exp(11−20)` 重新对齐尺度。**分块 softmax 不是「每块各算各的再拼起来」，而是「每块都要在同一个最终最大值坐标系里被重新对齐」。**

两层要分清:safe softmax 解决「exp 会不会炸」，online softmax 解决「不把整行放内存里还能不能得同样结果」。后者才是 tile 式计算成立的前提。

## forward：外层扫 K/V，内层扫 Q，边扫边修正

有了 online softmax，主体流程就顺了：把 Q/K/V 切成能塞进 SRAM 的小块，在块内把「矩阵乘、缩放、mask、softmax、乘 V」尽量一口气做完，**不实例化完整 `N×N` 矩阵**。

对每一行额外维护两个统计量:当前块为止的 `rowmax`（`m`）和 `rowsum`（`l`）；输出 `O` 也随块迭代更新。块内只先算到**未归一化**的概率，真正除以全局 rowsum 的归一化推迟到最后一步——这正是分块能等价于整行 softmax 的关键。

合并统计量的最小公式:某一行，上一轮维护 `m_old`、`l_old`，当前块得到 `m_blk`、`l_blk`，则

$$m_{\text{new}} = \max(m_{\text{old}}, m_{\text{blk}})$$
$$l_{\text{new}} = \exp(m_{\text{old}} - m_{\text{new}})\, l_{\text{old}} + \exp(m_{\text{blk}} - m_{\text{new}})\, l_{\text{blk}}$$

输出 `O` 同理:旧贡献和新贡献先各自对齐到 `m_new`，再按新归一化和重组。物理意义是:**后面出现更大的 score 时，前面算出的概率没有作废，而是被重新缩放到新的全局尺度里。**

具体走一遍（Q 切 3 块、K/V 切 4 块，FlashAttention-1 的顺序:外层 K/V、内层 Q；FlashAttention-2 把外层换成 Q 更利于并行）:K/V 第 0 块进 SRAM，Q0/Q1/Q2 依次和它算局部 score 与局部输出贡献;K/V 第 1 块进来时，同样的 Q0/Q1/Q2 再各算一遍，若新局部最大值更大，就把旧统计量整体 rescale 到新坐标系;等 K/V 全部扫完，O 被逐步修正成最终正确答案。

两个实现细节:causal mask 不是最后统一补的，而是在每个 tile 内就处理掉「看未来 token」的非法位置，否则局部 softmax 就已经错了（对应 MiniMind Flash 路径里的 `is_causal=True`）;真正需要长久保存的不是整块 score 或概率矩阵，而只是每行的 `rowmax`、`rowsum` 和当前输出 `O`——这就是它把显存从「存整张注意力图」降到「只存每行少量统计量 + 输出向量」的原因。

## 三种资源，各自变了什么

FlashAttention 的效果落到三种资源上，别混:

- **计算量**:大阶**没变**，仍是 `O(N²)`。它不是靠少算很多来赢。
- **显存**:不需再显式存完整 `N×N` attention 矩阵，从「必须扛住 `N²` 中间矩阵」降到更接近 `O(N)`。
- **HBM 往返**:明显减少 SRAM 与 HBM 之间的大规模搬运——这是 wall-clock 时间下降的**主因**。

所以它不是「超长上下文的万能解药」:没消掉二次计算关系，序列极长时总算量依然很大;它修的是「算这些东西时显存和 IO 浪费太夸张」。IO 压下来后，系统瓶颈往往重新暴露到别处（MLP、通信、KV cache、调度）。FlashAttention 不是终点，是把一部分主要瓶颈移开后的新起点。

## 落回 MiniMind

回到主线那两条并排的路径，现在能看清它们的关系了:

```python
# Flash 路径：SDPA 底层就是 IO-aware 分块 + online softmax
if self.flash and seq_len > 1 and past_key_value is None and 无复杂 mask:
    output = F.scaled_dot_product_attention(xq, xk, xv, ..., is_causal=True)
else:
    # 标准路径：显式 materialize 完整 N×N score 矩阵
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores[..., -seq_len:] += torch.triu(全 -inf, diagonal=1)   # causal mask
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = scores @ xv
```

- **标准路径**就是本篇开头说的那个「痛点」:显式算出完整 `N×N` 的 `scores`、整行做 softmax、再 `@V`。中间那个大矩阵会落回 HBM。
- **Flash 路径**一行 `F.scaled_dot_product_attention` 背后，就是本篇讲的整套 IO-aware 分块 + online softmax——PyTorch 2.0+ 内置，自动选后端（FlashAttention / Memory-Efficient / 数学后端）。

两条路径**数学上完全等价**（Flash 是 exact attention），差别只在计算顺序和 IO。这也解释了 MiniMind 那个触发条件:`seq_len > 1 and past_key_value is None`——Flash 走的是「一次并行处理一整段序列」的 prefill 式计算，最能发挥分块优势;而带 KV cache 的增量解码每步 `seq_len=1`（只有一个新 token），没有大矩阵可分块，回退标准路径反而简单（主线 [04-inference/01](../04-inference/01-kv-cache-and-generate.md) 讲过增量解码为什么每步只算一个 token）。

## 常见误区

- **「FlashAttention 把 attention 复杂度降到线性」**——没有。计算量仍是 `O(N²)`，它降的是显存占用（→`O(N)`）和 HBM 往返，不是算术复杂度。
- **「它是一种近似 attention」**——不。它是 exact attention，结果和标准实现逐位相同，只是计算顺序不同。
- **「Flash 靠少算 FLOPs 变快」**——不。乘加次数没少，快在减少了内存搬运（IO-aware）。
- **「带 KV cache 的推理也走 Flash」**——MiniMind 里不走。增量解码每步 `seq_len=1`、没有大矩阵可分块，回退标准路径。

## 练习

1. 为什么说 attention 的瓶颈常常是「带宽受限」而不是「算力受限」？SRAM 和 HBM 各扮演什么角色？
2. FlashAttention 自称 exact attention，这和稀疏注意力、低秩近似有什么本质区别？
3. 为什么 softmax 的分母让 attention 难以直接分块？safe softmax 和 online softmax 各解决什么？
4. online softmax 遇到「后面的块出现更大最大值」时怎么处理已经算过的前面块？写出统计量合并的思路。
5. 对照 MiniMind 源码，为什么 Flash 路径要求 `seq_len > 1 and past_key_value is None`？带 KV cache 的增量解码为什么回退标准路径？

<details>
<summary>参考答案</summary>

1. 因为 softmax、mask、dropout、reduction 这些步骤的瓶颈是数据搬运而非计算，标准实现把中间大矩阵反复写回 HBM、再读回来，时间耗在搬运上。SRAM 小但极快（片上）、HBM 大但慢（片外），FlashAttention 尽量把计算留在 SRAM、减少 HBM 往返。
2. 稀疏注意力删掉部分连接、低秩近似换掉计算目标，都改变了结果（是近似）；FlashAttention 结果和标准 attention 逐位相同（exact），只重排计算顺序以降低 IO。
3. 因为 softmax 分母 `Σexp` 依赖整行全部元素，分块后单块拿不到全局分母。safe softmax 减行最大值解决数值溢出（exp 不炸）；online softmax 在遍历块时在线维护全局最大值和指数和，解决「不存整行也能得到同样结果」，是分块可组合的前提。
4. 用新的全局最大值 `m_new = max(m_old, m_blk)` 重标定:旧的指数和乘 `exp(m_old − m_new)`、新块乘 `exp(m_blk − m_new)` 后相加得 `l_new`；输出 O 的旧贡献和新贡献也各自对齐到 `m_new` 再按 `l_new` 重组。前面的概率不作废，只是被 rescale 到新尺度。
5. Flash 走「一次并行处理整段序列」的计算，`seq_len > 1` 才有大矩阵可分块、发挥 IO 优势，`past_key_value is None` 对应 prefill（没有历史缓存）。带 KV cache 的增量解码每步只有一个新 token（`seq_len=1`），没有大矩阵可分块，走标准路径反而简单直接。
</details>
