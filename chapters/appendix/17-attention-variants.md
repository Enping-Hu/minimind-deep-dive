# 延伸：长上下文 Attention 变体——改连接、改缓存之外还能改什么

MiniMind 的 attention 是标准的 causal self-attention(主线 [02-attention](../02-model/02-attention.md)),序列一长,`QK^T` 的计算和显存都随长度平方增长。前面两篇讲了长上下文降本的两条路:附录 [03 RoPE 长度外推](03-rope-length-extrapolation.md) 改**位置编码**让模型认得更长,附录 [04 KV cache 压缩](04-kv-cache-compression.md) 压**缓存**(MHA→MLA)。

这一篇讲第三条,也是最直接的一条:**改 attention 本身的连接方式或计算形式**。DCA、S2-Attention、Gated Attention、NSA、DSA 这一组名字看着杂,但都在回答同一个问题:序列很长时,是不是每个 token 都真的要和所有其他 token 精细交互?MiniMind 没用这些(小模型、上下文不长,标准 attention 够),但想理解现代长上下文模型怎么做到 128K、1M,绕不开它们。

一句话主线:**长上下文 attention 优化都在「保住多少信息」和「省下多少计算」之间重画边界;区别只在:改连接还是改信息流、稀疏模式写死还是动态、从训练就参与还是推理时才打补丁。**

## 先立两根坐标轴

标准 attention 的复杂度问题,底层有三类应对(主线提过):**稀疏化**(不全连接,只留一部分)、**局部化**(计算压到窗口附近)、**改计算形式**(不显式构造 `n×n` 矩阵,如 Linear Attention、Flash)。附录 [10 FlashAttention](10-flash-attention.md) 属「改计算形式」但**数学等价、不近似**;附录 [04](04-kv-cache-compression.md) 的 MLA 是独立的「压缓存」维度。

要把下面 5 个变体讲清,叠加第二根轴会更顺——**它们在模型生命周期的哪个阶段介入**:

- **推理补丁**(training-free,不重训原模型)
- **训练期技巧**(微调阶段用,推理时未必保留)
- **训练原生**(预训练就参与,训练推理一致)

两根轴一搭,这组变体就各归其位了。

## DCA:分块——先局部精算,再跨块沟通

**DCA(Dual Chunk Attention)**(*Training-Free Long-Context Scaling*,ChunkLlama)解决的是:长文里既有近距离细节、又有跨段依赖,纯滑动窗口丢全局,全 dense 顶不住。

机制一句话:把长序列切成 chunk,**块内做细粒度自注意力**(保局部细节),**块间补一条粗粒度信息流**(传跨段线索)。不是两套模型,而是同一序列放进两种粒度。关键是它**显式给远程依赖留了一条通道**,区别于只砍远连接的窗口法。

它是**推理补丁**(training-free):不重训,就能把 Llama2 70B 这类模型推到 100k+。代价:效果高度依赖 `chunk_size`(太小频繁跨块补救,太大算力回升);跨块通信比块内粗,对「任意远距离 token 精细逐点对齐」的任务吃亏。擅长长文检索/问答/长代码/论文回指这类「细节多为局部、偶尔接回远线索」的场景。

## S2-Attention:移位——稀疏但可连通

**S2-Attention(Shifted Sparse Attention)**(*LongLoRA*)解决的是:长上下文**微调**时 dense 成本平方级涨,承受不起;但稀疏分组后又怕窗口变成信息孤岛。

机制一句话:分组做组内局部 attention 降本,再对**一部分 head** 错位半个 group,让相邻组逐层交换信息。精髓不是「稀疏」,而是「**稀疏但可连通**」。两个关键:① 只 shift 一半 head(一半忠于本地窗口保精度,一半负责把边界信息带出去);② shift 会引入信息泄漏,必须配套调 attention mask 保因果约束,不是简单 `roll`。

它是**训练期技巧**:LongLoRA 语境下主要为便宜地做长上下文微调,推理时常回到标准 full attention。代价:远程信息靠多层逐步扩散、非一层恢复,对「一开始就要对很远 token 高精度对齐」的任务吃亏。

## Gated Attention:门控——不改连接,改信息流

**Gated Attention**(*Gated Attention for LLMs*,Qwen 团队)是这组里的异类:它**不碰复杂度、不改连接**,而在 attention 输出后乘一个输入相关的 **gate**——attention 负责匹配,gate 负责筛选与抑制。

它想剥离「门控本身的贡献」,顺带解决 attention sink(模型把大量注意力莫名压到序列开头某 token)、massive activation、长上下文退化和训练不稳。最优配置很具体:在 **SDPA 输出后、输出投影前(G1 位置)**,用 `head-specific + element-wise + sigmoid` 的 gate(`gate_mlp(ln(x))→sigmoid→attn_out*gate`)。效果惊人:首 token 注意力占比从 46.7%(峰值 83%)降到 0.048。

它是**训练原生**的结构增强件,对「大 batch + BF16 + 高学习率」的训练稳定性尤其友好。代价:理论机制(非线性如何改变注意力动力学、sink 与长上下文的因果)尚未讲清,实验规模只到 15B。

## NSA / DSA:动态稀疏——稀疏从训练就参与

前面几个的稀疏模式基本**预先写死**(分块、分组)。最新一代把稀疏做成**动态、且从训练就参与**。

**NSA(Native Sparse Attention)** 的关键词是 `Native`:不做「先 full attention 训、推理时裁」的后处理稀疏(那会造成训练-推理分布错位),而**从预训练就在稀疏结构下学**。机制是动态分层稀疏——粗粒度压缩(保全局)+ 细粒度选择(补关键远程)+ 局部高精度,三路合并。它还强调 **Hardware-Aligned**:把稀疏连接组织成 GPU 喜欢的规则成块访存,避免优势耗在不规则 gather/scatter 上。

**DSA(DeepSeek Sparse Attention)**(附录 [13 DeepSeek 谱系](13-deepseek-lineage.md) 讲过)是另一条动态稀疏:用轻量 `lightning indexer` 给历史 token 打分、只保留 top-k 送进精算,先 dense warm-up 再切 sparse。

两者最接近、方向却相反:**NSA 重写底层范式(默认计算图就稀疏),DSA 在现有 attention 上外挂一个 query-aware 选点器**。但都比 DCA/S2 前进一步:**稀疏由 query 动态决定,而非预先写死**。

## 一张表看懂这组变体

| 变体 | 改什么 | 何时介入 | 稀疏模式 | 出处 |
|---|---|---|---|---|
| 标准 Attention | 全连接(基线) | — | — | Transformer |
| Flash Attention | 计算形式/访存(等价) | 训推通用 | — | 附录 [10](10-flash-attention.md) |
| DCA | 局部化 + 分块补全 | **推理补丁** | 分块预设 | ChunkLlama |
| S2-Attention | 稀疏 + 局部 + shift 连通 | **训练期技巧** | 分组+shift 预设 | LongLoRA |
| Gated Attention | **信息流(输出加门控)** | 训练原生 | 不适用(软筛选) | Qwen 团队 |
| NSA | 稀疏(压缩+选择+局部) | **训练原生** | 动态,训练学出 | NSA 论文 |
| DSA | 稀疏(query-aware top-k) | 训练参与(dense→sparse) | 动态,query 依赖 | DeepSeek-V3.2 |

读这张表:纵向是一条时间线——早期靠**推理补丁/微调技巧**(DCA、S2)便宜地把已有模型推长,后来发现「训练时假设全连接、推理时砍边」会分布错位,于是走向**训练原生**(NSA、DSA);横向 Gated Attention 是个提醒——长上下文优化不只有「砍连接」一条路,在信息流上加门控也能同时治稳定性和 attention sink。

## 落回 MiniMind

MiniMind **一个都没用**——它是标准 causal attention + 可选 Flash 路径(主线 02-attention)。这不是缺陷,而是匹配:MiniMind 上下文不长、模型只有 26M,标准 attention 的平方复杂度完全扛得住,上这些变体反而徒增复杂度、还未必有收益(它们的收益都在长序列才显著)。

把这组变体放进 MiniMind 的坐标:它们是「**当上下文长到标准 attention 扛不住时**」才需要的下一步。理解它们的价值,是看清 MiniMind 那个简单的 `scores = QK^T/√d → softmax → @V`(主线九步)在真实长上下文场景会撞上什么墙、业界又怎么绕:改连接(DCA/S2)、改信息流(Gated)、动态稀疏(NSA/DSA),或者前两篇的改位置编码(RoPE 外推)、压缓存(MLA)。这些路线正交、常组合,共同把 attention 从「算得起短序列」推到「扛得住 128K/1M」。

## 常见误区

- **「稀疏 attention 一定更快」**——不一定。理论少算不等于实际更快,稀疏模式若导致不规则访存,收益会耗在 gather/scatter 上——这正是 NSA 强调 Hardware-Aligned 的原因。
- **「这些变体和 Flash Attention 是竞品」**——不。Flash 是等价的计算形式优化(不改结果),这些变体多是稀疏近似(改了连接、结果不同),两者不同层、可叠加。
- **「Gated Attention 也是稀疏 attention」**——不。它不改连接、不改复杂度,而在输出侧加门控改信息流,顺带治 attention sink 和训练不稳。
- **「NSA 和 DSA 是一回事」**——方向相反。NSA 重写底层原生稀疏范式,DSA 在现有 attention 上外挂 query-aware 选点器;都动态、都训练参与,但一个改骨、一个加件。
- **「MiniMind 该上这些提速」**——不。它上下文短、模型小,标准 attention 够用;这些变体的收益只在长序列显著。

## 练习

1. 长上下文降本有哪三条正交路线?本篇讲的是哪一条?另两条在哪两篇?
2. DCA 的「dual」指什么?它为什么是 training-free 的推理补丁、适合什么场景?
3. S2-Attention 为什么只 shift 一半 head?全 shift 会有什么问题?
4. Gated Attention 和其他四个变体最大的不同是什么?它顺带解决了什么长期困扰?
5. NSA 和 DSA 都是动态稀疏,方向有何相反?它们相比 DCA/S2 前进在哪?

<details>
<summary>参考答案</summary>

1. 三条:改位置编码撑长度(附录 03 RoPE 外推)、压 KV cache(附录 04,MHA→MLA)、改 attention 连接/计算形式(本篇)。本篇讲第三条。
2. 「dual」指双层信息粒度:块内细粒度自注意力(保局部细节)+ 块间粗粒度信息流(传跨段线索),不是两套模型。它 training-free 是因为不重训、只重排计算,就能把已有模型推到更长上下文。适合长文检索/问答/长代码/论文回指这类「细节多为局部、偶尔接回远线索」的场景。
3. 只 shift 一半 head:一半忠于本地窗口保住局部精度,一半错位半个 group 负责把边界信息带到相邻组。全 shift 会破坏所有 head 的局部窗口稳定性,失去「一半保精度」的作用;而且 shift 引入信息泄漏,还需配套调 mask 保因果。
4. 最大不同:它不改连接、不改复杂度,而在 attention 输出后加输入相关的门控(改信息流),其他四个都在动稀疏/连接。它顺带解决了 attention sink(注意力异常压到开头 token)、massive activation 和训练不稳。
5. NSA 重写底层范式(默认计算图就稀疏、硬件对齐),DSA 在现有 attention 上外挂 query-aware 选点器(lightning indexer + top-k)——一个改骨、一个加件,方向相反。相比 DCA/S2 的预设稀疏模式,NSA/DSA 的稀疏由 query 动态决定、且从训练就参与(而非推理补丁或仅微调期)。
</details>
