# 延伸：GRPO 的变体家族

[03-grpo](../07-ppo-grpo/03-grpo.md) 讲了 MiniMind 用的 GRPO，[第 9 章](../09-minimind2-vs-3/04-grpo-cispo.md) 讲了 v3 默认换上的 CISPO。GRPO 提出后，社区给出了一批变体，各自针对它的某个具体问题。这一节把这些变体放进一条对比脉络，让你读到 Dr.GRPO、DAPO、GSPO 这些名字时，知道它们分别改了什么。

本节是**延伸**：MiniMind 只实现了 GRPO 和 CISPO，其余变体不在它的代码里，所以这里讲机制与定位，不贴源码。

## GRPO 留下的几个问题

回顾 [03-grpo](../07-ppo-grpo/03-grpo.md)，GRPO 的目标里有两处归一化和一处粒度选择，后来的变体多半在改这几个地方：

- **长度归一带来的偏差**：loss 对每条回答除以它的长度。长短回答被拉到同一尺度，副作用是写错的长回答里每个 token 受罚更轻，训练中容易让错误回答越写越长。
- **标准差归一在极端组失效**：advantage 除以组内 reward 的标准差。当一组回答全对或全错时，标准差趋近 0，advantage 失去意义、数值还可能爆炸。
- **对称裁剪压制探索**：沿用 PPO 的对称裁剪区间 `1±ε`，会把低概率 token 的向上更新一并压住，多样性下降偏快。
- **token 级重要性比在长序列上累积方差**：逐 token 的概率比，噪声随回答变长而累积，再被裁剪放大；在大模型、长 CoT、尤其 MoE 路由场景下容易训练发散。

下面四个变体的出发点，正对应这四点。

## 四条变化轴

这一族变体的差异基本落在四条正交的轴上：

| 轴 | 在改什么 |
|---|---|
| A. advantage 怎么归一 | 保留还是去掉长度归一、标准差归一 |
| B. 在哪一层、裁什么 | 裁 token 概率比（对称 / 非对称），还是裁重要性权重本身 |
| C. token 级还是序列级 | 重要性比按 token 算，还是按整条序列算 |
| D. 采样 / 分组怎么筛 | 全收，还是丢弃无区分度的组 |

## 四个主要变体

**Dr.GRPO**（*Understanding R1-Zero-Like Training*，Sea AI Lab，arXiv 2503.20783）改的是轴 A。它把目标里的两个归一化项一起删掉：既不除长度，也不除标准差，回到更接近原始 PPO / REINFORCE 的无偏形式。这样写错的长回答不再因为「长」而少受罚，长度偏差和难度偏差一并缓解。核心判断是：GRPO 那两个除法本身就是偏差来源。

**DAPO**（Decoupled Clip and Dynamic Sampling Policy Optimization，字节跳动 Seed + 清华，arXiv 2503.14476）是一组工程化改动的组合，最常被单独引用的是两点。其一是 **Clip-Higher**，把裁剪区间改成非对称的 `ε_high > ε_low`，给低概率 token 的向上更新留出空间（轴 B，对应上面的问题三）。其二是**动态采样**，把一组里全对或全错、标准差为 0 的 prompt 直接丢弃，只留有区分度的组（轴 D，对应问题二）。此外它主张 token 级损失，并对超长回答做奖励整形。

**GSPO**（Group Sequence Policy Optimization，Qwen 团队，arXiv 2507.18071）改的是轴 C。它把重要性比从 token 级换成序列级：按整条回答算一个似然比，并做长度归一（逐 token 比的几何平均），裁剪与优化也放到序列层。长序列上的方差不再逐 token 累积，附带的好处是 MoE 训练不必再依赖 Routing Replay。Qwen3 用的就是它。

**CISPO**（Clipped Importance Sampling Policy Optimization，MiniMax-M1，arXiv 2506.13585）是轴 B 的另一种思路，[第 9 章](../09-minimind2-vs-3/04-grpo-cispo.md) 已讲过：它不裁 token 的更新，改裁重要性采样权重本身，于是没有 token 被整个丢弃，反思类的低概率、高权重 token（However、Wait 这类）也能持续贡献梯度。MiniMind-3 把它设为 GRPO 的默认 loss。

## 一张表对照

| 方法 | 轴 | 关键改动 | 出处 |
|---|---|---|---|
| GRPO | 基准 | 组内相对 advantage（除标准差）+ 长度归一 | DeepSeek |
| Dr.GRPO | A | 去掉长度与标准差两处归一化 | arXiv 2503.20783 |
| DAPO | B + D | 非对称 Clip-Higher + 丢弃 std=0 的组 | arXiv 2503.14476 |
| GSPO | C | 重要性比改序列级（长度几何平均） | arXiv 2507.18071 |
| CISPO | B | 裁重要性权重而非裁 token，保留全部梯度 | arXiv 2506.13585 |

读这张表的方式和 [归一化演进](02-normalization-evolution.md) 一样：每个变体不是「取代」GRPO，而是各自回答它留下的一个问题。你已经学过的 CISPO，在这张表里的位置就是轴 B 的一种——裁权重，不裁 token。

## 命名上的一个提醒

RL 后训练这两年出得快，缩写重名不少，引用时要认准 arXiv 号：

- **SAPO** 至少对应两篇不同的工作：Qwen 的 Soft Adaptive Policy Optimization（arXiv 2511.20347，用温度可控的软门控替代硬裁剪）和 Gensyn 的 Swarm Sampling Policy Optimization（arXiv 2509.08721，讲的是去中心化 rollout 收集，与目标函数无关）。
- **GTPO** 同样有两篇同月预印本（arXiv 2508.03772 与 2508.04349），全称和机制都不同。

这类名字在站稳之前，本节不纳入正式对照，以免把还在流动的命名当成定论。

## 练习

1. GRPO 的长度归一会带来什么偏差？哪个变体如何处理它？
2. 一组回答全对或全错时，GRPO 的标准差归一会出什么问题？DAPO 的动态采样怎样回避？
3. GSPO 把重要性比从 token 级改成序列级，主要解决哪种场景下的问题？
4. CISPO 和 DAPO 都和「裁剪」有关，两者裁的对象有什么不同？

<details>
<summary>参考答案</summary>

1. 长度归一让长回答里每个 token 受罚更轻，错误的长回答因此越写越长（长度偏差）；Dr.GRPO 直接去掉长度归一（连同标准差归一一起删），回到无偏形式。
2. 全对或全错时组内 reward 标准差趋近 0，advantage 失去意义甚至数值爆炸；DAPO 的动态采样把这类 std=0 的组整组丢弃，只用有区分度的组更新。
3. 解决大模型、长 CoT、尤其 MoE 路由场景下，token 级重要性比方差逐 token 累积、训练易发散的问题；序列级比加长度归一后更稳，并让 MoE 训练免去 Routing Replay。
4. CISPO 裁的是重要性采样权重，不丢弃任何 token 的梯度；DAPO（Clip-Higher）裁的仍是 token 概率比，只是把上界放宽成非对称。两者都在轴 B 上，但裁的对象不同。
</details>
