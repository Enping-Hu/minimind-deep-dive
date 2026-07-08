# 延伸：MoE 的演进——MiniMind 的每个 MoE 选择，来自谱系的哪一段

主线 [02-model/06-moe](../02-model/06-moe.md) 已经把 MiniMind 的 MoE 逐行讲透了：`MoEGate` 怎么给 token 选专家、`aux_loss` 怎么防专家塌缩、routed experts 和 shared experts 的分工、训练路径和推理路径的差别。这一篇不重复那些实现，而是往外看一层：MiniMind 那几个 MoE 配置（top-2、共享专家、序列级 aux loss、`alpha=0.01`）**不是随手定的，而是 MoE 这些年演进里每一步的沉淀**。读完你会发现，`MiniMindConfig` 里那一小段默认值，其实是一张 MoE 谱系的坐标。

一句话主线：**MoE 的所有工作都在回答同一个问题——想让总参数变大、但每个 token 只激活一小部分，那么「怎么选专家、怎么不让专家失衡、怎么让专家真分工」这三件事该怎么做。**

## 为什么要 MoE

先说动机。想让模型更强，最直接的办法是把参数堆大;但 dense 模型每个 token 都要过全部参数,参数一大,单 token 的计算量(FLOPs)跟着涨,训练和推理都吃不消。

MoE 的思路是**稀疏激活**:把一层 FFN 换成 `N` 个并列的专家 FFN,每个 token 只经过其中 `k` 个(`k≪N`)。于是总参数(容量)可以做得很大,而单 token 激活的参数(计算量)只由 `k` 决定。参数量从此变成一个**几乎独立于单 token FLOPs 的缩放轴**——这是 Switch Transformer 点破的关键视角:能在不显著增加单 token 计算的前提下,把总容量越推越大。

代价是三个新问题,整条演进线都在解决它们:**怎么选专家(router)、怎么不让专家失衡(负载均衡)、怎么让专家真分工(专家设计)**。

## GShard:把 MoE 的部件一次摆齐

GShard 是学 MoE 最该先啃的早期工作,因为它几乎把后面反复出现的部件都提前摆了出来:

- **gate / router**:一个 `隐藏维 M → 专家数 E` 的小线性层,给每个 token 算出对每个专家的匹配分数,再挑 top-k。这就是主线 `MoEGate` 里那个 `self.weight`(`n_routed_experts × gating_dim`)+ softmax + topk 的由来。
- **top-2 routing**:每个 token 去得分最高的两个专家。比 top-1 多留一点冗余表达,比全激活省。**MiniMind 的 `num_experts_per_tok=2` 走的正是这条**。
- **capacity(容量上限)**:一个专家一轮最多接多少 token,防止热门专家被挤爆。约等于 `(token数/专家数) × top_k`,不是玄学常数。
- **token dropping + zero padding**:超过容量的 token 这一步不进专家、直接走残差;buffer 没占满的位置用零补齐(硬件喜欢固定形状)。
- **auxiliary loss**:不提升主任务,只让路由别总挤到少数专家。

把这五件记住,后面每个 MoE 变体都能沿这个模板拆。MiniMind 的 MoE 基本就是这个模板的一个精简实例(它没实现 capacity/token dropping,因为专家数少、教学为主)。

## Switch Transformer:把 top-k 砍到 1

Switch 是个转折点:它把路由激进简化到 **top-1**——每个 token 只去一个专家。好处是通信路径更短、实现更简单、合并成本更低,更容易扩到万亿参数(最大版 1.6T、2048 专家)。

代价是丢了 top-2 的冗余表达空间,所以它必须更认真解决负载均衡:

- **capacity factor**:在平均容量 `token数/专家数` 上乘一个冗余系数,大则 overflow 少但通信贵,小则省但易丢 token。
- **负载均衡损失**:组合两个量——每个专家实际接收 token 的**占比**(硬分配、不可导)和 router 给每个专家的**平均概率**(可导),要求「模型想怎么分」和「实际怎么分」都尽量均匀。**这正是主线讲的 `aux_loss` 的形式**(`Pi × fi` 求和)。
- **aux loss 权重**:Switch 论文扫描后推荐 `1e-2`。**MiniMind 的 `aux_loss_alpha=0.01` 就是这个经验值**。

Switch 还证明了 `MoE→dense` 蒸馏可行(压掉 99% 参数仍比直接训 dense 好),说明 MoE 也能当强教师。

## Mixtral:top-2 稀疏 MoE 的成熟落地

Mixtral 8x7B 是 top-2 路线的代表:总参数 46.7B,但每个 token 只激活约 12.9B,单 token 成本更接近一个 13B 模型。它的专家就是标准 SwiGLU FFN(和主干一致),每层 attention 之后过 router、top-2 加权求和:

$$y = \sum_{i \in \text{Top-2}} g_i(x)\, E_i(x)$$

这个式子和主线 `MOEFeedForward` 的 `topk_weight` 加权合并专家输出**完全对应**。Mixtral 还澄清了一个常见误解:专家的分工往往不是干净的「这个懂语法、那个懂语义」,而更像对某些 **token 类型/格式片段**形成稳定偏好(如代码缩进、`self`、标点会稳定路由到相同专家)。所以 MoE 是 **token-level** 分工,同一句话里不同 token 可能走不同专家。

## DeepSeekMoE:共享专家 + 细粒度

MiniMind 有一个配置在 GShard/Switch/Mixtral 里都找不到对应:`n_shared_experts=1`。这来自 **DeepSeekMoE** 的两个设计:

- **共享专家(shared expert)**:除了 routed experts(按 token 路由的),再设几个**所有 token 都过**的共享专家。直觉是:有些通用能力(基础语言建模)每个 token 都需要,与其让每个 routed expert 都重复学一遍,不如抽出来放进共享专家,让 routed experts 专注各自的「专」。**主线讲的 routed vs shared 分工,正是这个设计**。
- **细粒度专家(fine-grained)**:把专家切得更小更多(比如把 1 个大专家拆成 4 个小的、top-k 相应放大),让路由组合更灵活、专家分工更细。

DeepSeekMoE 的负载均衡也更精细(设备级、专家级多重 aux loss),但核心就这两点:**共享专家兜底通用能力,细粒度专家做精细分工**。V3 里进一步用无辅助损失的负载均衡(靠可学习偏置调节路由),但 MiniMind 用的仍是经典 aux loss。

## 落回 MiniMind

现在回看 `MiniMindConfig` 那段 MoE 默认值,每一个都能定位到谱系上:

| 配置 | 值 | 来自谱系哪一段 |
|---|---|---|
| `num_experts_per_tok` | 2 | GShard/Mixtral 的 **top-2**(非 Switch 的 top-1) |
| `n_routed_experts` | 4 | 按 token 路由的专家数(教学规模,远小于 Mixtral 的 8、Switch 的 2048) |
| `n_shared_experts` | 1 | **DeepSeekMoE 的共享专家** |
| `scoring_func` | softmax | GShard 起的经典 gate 打分 |
| `aux_loss_alpha` | 0.01 | **Switch 论文推荐的 1e-2** |
| `seq_aux` | True | 序列级负载均衡(DeepSeek 风格,比 token 级更稳) |
| `norm_topk_prob` | True | top-k 概率归一化(top-2 时把两权重归一) |

所以 MiniMind 的 MoE 在谱系上的坐标很清楚:**它是「GShard 的 top-2 模板 + Switch 的 aux loss 经验 + DeepSeekMoE 的共享专家」的一个精简教学实现**,只是省掉了大规模场景才需要的 capacity 控制、token dropping、EP 并行。想在它上面进阶,方向也顺着谱系:

- 加 capacity + token dropping,复现 GShard/Switch 的溢出处理;
- 把 `num_experts_per_tok` 改 1,体验 Switch 的 top-1 和随之而来的均衡压力;
- 加细粒度专家(拆小 `n_routed_experts` 同时放大 top-k),体验 DeepSeekMoE 的路由灵活性。

## 一张表看懂演进

| 工作 | 路由 | 关键贡献 | 负载均衡 |
|---|---|---|---|
| GShard | top-2 | 一次摆齐 MoE 全部部件(gate/capacity/drop/aux) | aux loss(只看 1st expert) |
| Switch | **top-1** | 把 MoE 简化到能扩万亿参数;参数=独立缩放轴 | aux loss(占比×概率)+ capacity factor |
| Mixtral | top-2 | top-2 稀疏 MoE 成熟落地;token-level 分工 | 沿用经典 aux loss |
| DeepSeekMoE | top-k + 共享 | **共享专家 + 细粒度专家** | 设备/专家级多重 aux |
| **MiniMind** | top-2 + 1 共享 | 上述的精简教学实现 | seq 级 aux(alpha=0.01) |

读这张表:GShard 立模板 → Switch 砍到 top-1 换取极致扩展 → Mixtral 用 top-2 做成好用的开源模型 → DeepSeekMoE 用共享+细粒度让专家分工更合理。MiniMind 站在这条线的下游,取了各家最实用的组合。

## 常见误区

- **「MoE 减少了模型参数」**——反了。MoE 是**增大总参数**、只减少单 token **激活**的参数;总容量变大、单 token 计算不变。
- **「专家各自懂一个学科」**——不准确。实际分工更像对 token 类型/格式的稳定偏好(token-level),不是干净的学科划分。
- **「top-1 比 top-2 落后」**——不是落后,是取舍。top-1(Switch)通信/实现更省、更易扩万亿,代价是丢冗余表达、更依赖负载均衡;top-2(GShard/Mixtral/MiniMind)表达更稳。
- **「aux loss 是为了提升主任务」**——不。它只管负载均衡(别让专家塌缩到少数几个),不直接提升主任务,所以权重很小(0.01)。
- **「共享专家是多余的重复」**——不。共享专家专门兜底所有 token 都需要的通用能力,让 routed experts 能专注各自的「专」,反而提升分工效率。

## 练习

1. MoE 为什么能「总参数大、单 token 计算不变」?它把哪两个量解耦了?
2. GShard 摆出的 MoE 五大部件是什么?MiniMind 的 MoE 实现了其中哪些、省了哪些?
3. Switch 把 top-2 砍到 top-1 换来什么、又付出什么代价?为什么它更依赖负载均衡损失?
4. MiniMind 的 `aux_loss_alpha=0.01` 和 `n_shared_experts=1` 分别来自谱系的哪一段?
5. 共享专家(shared expert)解决什么问题?为什么它不是「多余的重复」?

<details>
<summary>参考答案</summary>

1. 因为把一层 FFN 换成 N 个专家、每个 token 只过 k 个(k≪N):总参数由 N 决定(可以很大),单 token 激活参数由 k 决定(保持小)。它解耦了「总容量(参数量)」和「单 token 计算量(FLOPs)」这两个原本绑在一起的量。
2. 五大部件:gate/router、top-k routing、capacity(容量上限)、token dropping + zero padding、auxiliary loss。MiniMind 实现了 gate(MoEGate)、top-k(top-2)、aux loss;省掉了 capacity 控制和 token dropping(专家数少、教学为主,不需要溢出处理)。
3. 换来:通信路径更短、实现更简单、合并成本更低、更易扩到万亿参数。代价:丢了 top-2 的冗余表达空间。更依赖负载均衡因为 top-1 没有第二专家兜底,一旦热门专家过载、overflow 增多,扩出来的参数就白费,所以必须靠 aux loss + capacity factor 把分配压均匀。
4. `aux_loss_alpha=0.01` 来自 Switch Transformer(论文扫描后推荐的 1e-2 经验值);`n_shared_experts=1` 来自 DeepSeekMoE(共享专家设计)。
5. 共享专家让所有 token 都经过、专门承载「每个 token 都需要的通用能力(如基础语言建模)」。不是多余重复:如果没有它,每个 routed expert 都得各自重复学一遍这些通用能力;抽出来共享后,routed experts 能专注各自的精细分工,整体分工效率更高。
</details>
