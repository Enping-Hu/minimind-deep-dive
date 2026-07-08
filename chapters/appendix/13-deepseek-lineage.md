# 延伸：DeepSeek 谱系——MiniMind 借来的那些设计，出自哪一代

翻遍这份笔记会发现 DeepSeek 的名字反复出现：主线 [ch09](../09-minimind2-vs-3/01-overview-five-changes.md) 说 MiniMind-3 的多处改动向 DeepSeek/Qwen 看齐，附录里 [MLA](04-kv-cache-compression.md)、[共享专家](12-moe-evolution.md)、[CISPO](05-grpo-variants.md)、[MTP](07-speculative-decoding.md)、[QK-Norm](02-normalization-evolution.md) 又各自出自 DeepSeek 的某一代。这些技术散落在各篇，容易只见树木不见森林。

这一篇把它们串成一条时间线：DeepSeek 从 V1 到 V3.2，每一代解决什么问题、引入什么设计，这些设计又如何一步步累积成今天的样子。它不重讲每个技术的细节（那些在对应深入篇里），而是做一张**谱系索引**——看清 MiniMind 借来的每块砖，出自这条线的哪一段。

一句话主线：**DeepSeek 这条线的主轴始终是「效率」——在把模型做强的同时，反复追问训练和推理的成本能不能再压下去；MiniMind 借鉴的，正是这条线上几个最成熟的效率设计。**

## V1：搭好起步链路

DeepSeek-V1 本身不是最亮眼的一代，但它把后面整个家族的**起步链路**搭完整了：LLaMA 路线的现代 decoder-only recipe（`Pre-RMSNorm + SwiGLU + RoPE`）、自己的 BBPE tokenizer、中英文 SFT、DPO 偏好优化。

这套 recipe 正好**就是 MiniMind 的底座**:Pre-Norm + RMSNorm(附录 [02 归一化](02-normalization-evolution.md))、SwiGLU、RoPE(附录 [09 位置编码](09-positional-encoding-evolution.md)),MiniMind 和 DeepSeek 站在同一个起点上。V1 的 67B 版还加了 GQA(附录 [04 KV cache 压缩](04-kv-cache-compression.md))降推理成本,这个「从第一代就重视部署效率」的意识,是理解整条线的钥匙。

## V2：MLA + DeepSeekMoE,效率双线并进

V2 是这条线的第一个转折,主题明确:**总参数已经很大,怎么让训练和推理成本还能压住**。236B 总参数、每 token 只激活 21B、支持 128K 上下文。两项结构创新,都是纯效率导向:

- **MLA(Multi-head Latent Attention)**:把 KV cache 压成低秩 latent,长上下文推理的缓存成本大降。这就是附录 [04 KV cache 压缩](04-kv-cache-compression.md) 里 MHA→MQA→GQA→**MLA** 那条线的终点——MiniMind 停在 GQA,MLA 是它没走到的下一步。
- **DeepSeekMoE**:细粒度专家 + 共享专家。这正是附录 [12 MoE 演进](12-moe-evolution.md) 讲的、也是 **MiniMind 的 `n_shared_experts=1` 的来源**。

V2 还配了一整套系统设计(Device-Limited Routing、通信平衡损失、token dropping)把 MoE 的通信成本控住。它的意义是证明「MLA + DeepSeekMoE」这套高性价比底座真能跑通。

## V3:把效率路线做成完整训练体系

V3 沿用 V2 的 MLA + DeepSeekMoE(两条主线没变),规格拉到 671B 总参数、37B 激活、128K 上下文,新增三件事,每件都和别处呼应:

- **auxiliary-loss-free 负载均衡**:过去 MoE 用 aux loss 强推均衡,但这会干扰主任务。V3 把均衡从「主目标的一部分」降成「系统约束」(靠可学习偏置调路由),只保留一个序列级 aux loss 兜底。附录 [12 MoE 演进](12-moe-evolution.md) 提过这点——而 **MiniMind 用的仍是经典 aux loss**,没跟进这个。V3 还把 gate 从 softmax 改 sigmoid(专家多到 256 时 softmax 区分度下降)。
- **MTP(Multi-Token Prediction)**:训练时用顺序的 MTP module 预测后续多个 token,既提升表示、又能在推理时喂给 speculative decoding。这正是附录 [07 投机解码](07-speculative-decoding.md) 里「把 MTP 头复用为 draft」那条路线的源头。
- **FP8 混合精度训练 + 通信重叠**:把训练精度(接上附录 [06 量化](06-quantization.md) 的位宽话题)和跨卡通信一起优化,让 671B 训得起、且全程无不可恢复的 loss spike。

V3 是「承上启下」的节点:往前把 V2 的效率结构训成超大规模稳定底座,往后成了 R1 reasoning 路线的母体。

## R1:把 reasoning 放到训练中心

R1 换了主题——从「效率」转向「推理」。它的关键发现:**R1-Zero 直接在 base model 上做大规模 RL、不先 SFT,推理能力(自验证、反思、长 CoT)竟能自然冒出来**。但纯 RL 也带来重复、可读性差、语言混杂,所以正式的 R1 用「两次 SFT + 两次 RL」把这种能力整理成可读、可控、可部署的流程。

R1 和 MiniMind 的连接点在 ch09:MiniMind-3 的 thinking 路线、以及 GRPO/CISPO(附录 [05 GRPO 变体](05-grpo-variants.md))都受这波 reasoning-first 后训练的影响。R1 还证明了 reasoning pattern 可**蒸馏**到小 dense 模型(1.5B~70B),让「强推理不只属于超大模型」——这对 MiniMind 这种小模型尤其是个鼓励信号。

## V3.2:效率 + 推理 + agent 三者统一

V3.2(公开节点 `V3.2-Exp` 于 2025-09 放出)继续收束,主题是 `Efficient Reasoning & Agentic AI`:

- **DSA(DeepSeek Sparse Attention)**:如果 MLA 回答「KV cache 怎么压」,DSA 回答「长上下文时 attention 计算复杂度还能不能再省」。它是 MLA 之后的下一步长上下文提效。
- **更可扩展的 RL 后训练** + **thinking in tool-use**:把 reasoning 和工具调用绑成统一过程,向 agent 系统逼近(合成了覆盖 1800+ 环境的 agent 训练数据)。这条 Agent RL 方向,正是主线 ch09 提到的 MiniMind-3 新增 `train_agent.py` 所对标的前沿(附录 [01 进阶入口](01-advanced-pointers.md) 也点过 Agent RL)。

## 落回 MiniMind

把这条谱系和 MiniMind 对照,能清楚看到「MiniMind 借了哪几块砖、来自哪一代」:

| DeepSeek 设计 | 出自 | MiniMind 用了吗 | 深入篇 |
|---|---|---|---|
| Pre-RMSNorm + SwiGLU + RoPE | V1(承 LLaMA) | ✅ 底座 | [02](02-normalization-evolution.md) / [09](09-positional-encoding-evolution.md) |
| GQA | V1 (67B) | ✅ v2/v3 都用 | [04](04-kv-cache-compression.md) |
| MLA | V2 | ❌ 停在 GQA | [04](04-kv-cache-compression.md) |
| DeepSeekMoE 共享专家 | V2 | ✅ `n_shared_experts=1` | [12](12-moe-evolution.md) |
| aux-loss-free 均衡 | V3 | ❌ 仍用经典 aux loss | [12](12-moe-evolution.md) |
| MTP | V3 | ❌（但 v3 可接投机解码） | [07](07-speculative-decoding.md) |
| GRPO / CISPO | (承 R1 系 RL) | ✅ v3 默认 CISPO | [05](05-grpo-variants.md) |
| QK-Norm | (Qwen/DeepSeek 系) | ✅ v3 新增 | [02](02-normalization-evolution.md) |
| Agent RL | V3.2 方向 | ⚠️ v3 新增 `train_agent.py` | [01](01-advanced-pointers.md) |

读这张表:MiniMind **不是复刻某一代 DeepSeek**,而是从这条谱系里**挑了几个最成熟、最适合小模型教学的设计**——V1 的底座 recipe、V2 的共享专家、R1 系的 CISPO、Qwen/DeepSeek 系的 QK-Norm。它有意没上 MLA、aux-loss-free、MTP 这些「大模型/大规模场景才划算」的设计。所以理解 MiniMind 的结构,某种程度上就是理解「DeepSeek 谱系里哪些设计已经下沉成了小模型也该有的标配」。

## 常见误区

- **「DeepSeek 是从 V2/V3 才突然变强」**——不。V1 就搭好了稳底座 + tokenizer + SFT + DPO 的完整起步链路,后面每代都沿同一条数据和对齐管线往前长。
- **「DeepSeek 的核心是把模型堆大」**——不。它这条线的主轴是**效率**:MLA 压缓存、DeepSeekMoE 稀疏扩容、FP8 训练、DSA 稀疏 attention,都在压成本而非单纯堆参数。
- **「MiniMind 就是小号 DeepSeek」**——不准确。MiniMind 只借了谱系里几个成熟设计(底座 recipe、共享专家、CISPO、QK-Norm),有意没上 MLA/MTP/aux-loss-free 这些大规模才划算的。
- **「R1 证明了 CoT 越长越好」**——反了。R1-Zero 的长 CoT 又强又乱,R1 要学的恰恰是「该展开时展开、该收束时收束」,不是无节制拉长思维链。

## 练习

1. DeepSeek 这条线的主轴是什么?举两个体现这个主轴的具体设计。
2. V2 的两项核心结构创新是什么?它们各自对应本书哪篇深入篇?MiniMind 用了哪个、没用哪个?
3. V3 的负载均衡方式相比经典 MoE aux loss 有什么变化?MiniMind 跟进了吗?
4. R1-Zero 证明了什么反直觉的事?为什么还需要正式的 R1(两次 SFT + 两次 RL)?
5. 对照落回表,MiniMind 从 DeepSeek 谱系借了哪些设计、有意没借哪些?为什么这个取舍对一个 26M 教学模型是合理的?

<details>
<summary>参考答案</summary>

1. 主轴是**效率**（在把模型做强的同时反复压训练/推理成本）。体现:MLA（压 KV cache）、DeepSeekMoE（稀疏激活扩容量而不增单 token 计算）、FP8 训练 + 通信重叠、DSA（长上下文稀疏 attention）——任举两个即可。
2. MLA（对应附录 04 KV cache 压缩，MHA→MLA 那条线的终点）和 DeepSeekMoE（对应附录 12 MoE 演进，共享专家+细粒度）。MiniMind 用了 DeepSeekMoE 的共享专家（`n_shared_experts=1`），没用 MLA（停在 GQA）。
3. 经典 aux loss 强推均衡但会干扰主任务；V3 改成 auxiliary-loss-free（把均衡从主目标降成系统约束、靠可学习偏置调路由，只留一个序列级 aux 兜底），还把 gate 从 softmax 改 sigmoid。MiniMind 没跟进，仍用经典 aux loss（`aux_loss_alpha=0.01`）。
4. 证明了：直接在 base model 上做大规模 RL、不先 SFT，推理能力（自验证、反思、长 CoT）能自然冒出来。但纯 RL 有重复、可读性差、语言混杂的问题，所以正式 R1 用两次 SFT + 两次 RL 把这种能力整理成可读、可控、可部署的流程。
5. 借了:V1 的底座 recipe（Pre-RMSNorm/SwiGLU/RoPE）、GQA、V2 的共享专家、R1 系的 CISPO、QK-Norm。有意没借:MLA、aux-loss-free 均衡、MTP。合理是因为这些没借的设计都是「大模型/大规模训练场景才划算」的（MLA 在大 hidden/长上下文才显著、MTP/aux-loss-free 针对超大 MoE），对一个 26M 教学小模型收益不大、反而增加复杂度。
</details>
