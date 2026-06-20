# 延伸：归一化的演进——从 BatchNorm 到 QK-Norm

[01-block-and-rmsnorm](01-block-and-rmsnorm.md) 讲了 MiniMind 用的 RMSNorm，[第 9 章](../09-minimind2-vs-3/02-architecture-diffs.md) 会讲 v3 新增的 QK-Norm。这一节把这**两个真实落点**放进一条更大的脉络：归一化这些年怎么从 BatchNorm 一路演化到 RMSNorm、QK-Norm，每一步在解决前一步的什么问题。

本节是**延伸 survey**，不绑定某段 MiniMind 源码——但它的终点正好落回 MiniMind 的两个选择（v2 的 RMSNorm、v3 的 QK-Norm）。读它能回答一个面试常问、读源码也绕不开的问题：**为什么是 RMSNorm，而不是 LayerNorm 或别的？**

一句话主线：**归一化的演进，就是在反复回答「往哪加、加几处、能不能再砍一步」。**

## 为什么需要归一化

层数一深，`hidden_states` 的数值尺度会在不同 token、不同层之间漂移，后面的 Attention/FFN 越来越难稳定处理，梯度也容易爆炸或消失。归一化在每层把尺度收回一个可控范围，是深层网络能训得动的前提（[01](01-block-and-rmsnorm.md) 已从 block 视角讲过）。分歧只在于：**对谁、在哪一步、怎么算。**

## BatchNorm：为什么 NLP 基本不用它

BatchNorm 沿 **batch 维**对同一个特征做归一化（统计这一批样本里该特征的均值方差）。在 CV 里很成功，但 NLP 几乎不用：

- 序列**变长**，同一位置不同样本未必可比；
- batch 常**偏小**，batch 统计噪声大、不稳定；
- 推理时 batch 可能是 1，得依赖训练期攒的滑动统计，训练/推理行为不一致。

结论：依赖 batch 统计这件事在 NLP 里水土不服。于是转向「每个样本自己归一化」。

## LayerNorm：转向逐样本归一化

LayerNorm 不看 batch，只在**单个 token 的特征向量**（最后一维）上做：减均值、除标准差，让向量大致变成「均值 0、方差 1」。它和 batch 无关，变长序列、小 batch、batch=1 推理都一致——所以成了原始 Transformer 的标配。

## Pre-Norm vs Post-Norm：位置之争

同样用 LayerNorm，**加在哪一步**也经历过演化：

- **Post-Norm**（原始 Transformer）：`Norm(x + Sublayer(x))`，归一化在残差**相加之后**。表达力强，但深层时残差路径被 norm 反复挤压，训练难、常要 warmup 和小心调度。
- **Pre-Norm**：`x + Sublayer(Norm(x))`，归一化只作用于喂进子层的那份，残差是**未归一化的直通路**。深层更容易稳定训练，成为现代 LLM 主流。

MiniMind 用的就是 Pre-Norm（[01](01-block-and-rmsnorm.md) 的两次残差）。这一步的演进与「算哪种统计量」正交——它改的是**位置**。

## RMSNorm：把 LayerNorm 再砍一刀

RMSNorm 在 LayerNorm 基础上**去掉减均值**，只用 root mean square 控制尺度（机制见 [01](01-block-and-rmsnorm.md)）。少一步中心化，计算更省，而在 Transformer 里效果足够好，还更完整地保留向量方向。LLaMA、Qwen、**MiniMind（v2 和 v3）**都改用它。

到这里归一化一直在 block **主干**上做文章。下一步它走进了注意力**内部**。

## QK-Norm：归一化进入注意力内部

大模型或长时间训练时，注意力打分 `QK^T` 的数值容易越来越大，softmax 进入饱和、注意力熵坍塌，训练发散。QK-Norm 的做法是：在算 `QK^T` **之前**，给 Q、K 各加一层归一化（MiniMind-3 用 RMSNorm，作用在 `head_dim` 上），把 Q/K 的尺度稳住，注意力分数就不易爆。

这是归一化从「block 主干」延伸到「注意力子模块内部」的一步。**MiniMind-3 / Qwen3 加的正是这个**——详见 [第 9 章的架构差异](../09-minimind2-vs-3/02-architecture-diffs.md)。于是 MiniMind 自己就横跨了演进谱系的两端：v2 = Pre-Norm + RMSNorm，v3 再叠加 QK-Norm。

## 更深、更前沿（点到为止）

谱系还在延伸，但已超出本书主线，知道方向即可：

- **DeepNorm**：一种 Post-Norm 变体，通过缩放残差和初始化（α/β）把网络训到上千层。
- **Sandwich-Norm、NormFormer 等**：Sandwich-Norm 在子层前后都加 norm，NormFormer 在更多位置（如 FFN 内）补 norm，进一步稳住超大模型训练。

它们都没跳出那条主线——**调整「往哪加、加几处」**。

## 一张表看懂演进

| 方法 | 归一化对象 | 相对前一步的关键改动 | 代表 |
|---|---|---|---|
| BatchNorm | 跨 batch 的同一特征 | —（NLP 因依赖 batch 统计而弃用） | CV |
| LayerNorm | 单 token 特征向量（减均值+除 std） | 改成**逐样本**、不依赖 batch | 原始 Transformer |
| Pre-Norm | （位置）子层**之前** | norm 移出残差路径、留直通路 | 现代 LLM 主流、**MiniMind** |
| RMSNorm | 单 token 特征向量（只除 RMS） | **去掉减均值**，更省 | LLaMA/Qwen/**MiniMind v2·v3** |
| QK-Norm | 注意力的 **Q、K** | norm **进入注意力内部** | **MiniMind-3**、Qwen3 |
| DeepNorm 等 | 残差缩放 / 多处加 norm | 训超深层（点到为止） | 前沿 |

读这张表的方式：每一行不是「更高级所以取代上一行」，而是**回答上一行留下的问题**——BatchNorm 依赖 batch→LayerNorm 改逐样本；Post-Norm 深层难训→Pre-Norm 改位置；LayerNorm 还能省→RMSNorm 砍掉中心化；主干稳了但注意力还会爆→QK-Norm 进注意力内部。MiniMind 的两个选择（RMSNorm、QK-Norm）正好是这条线上的两个点。

## 练习

1. NLP 里为什么基本不用 BatchNorm？LayerNorm 改了哪一点来回避这个问题？
2. Pre-Norm 和 Post-Norm 的区别是什么？MiniMind 用哪种、好处是什么？
3. RMSNorm 相对 LayerNorm 砍掉了哪一步？为什么还够用？
4. QK-Norm 解决什么问题？它把归一化加在了哪里、和前面几种 norm 的位置有何不同？MiniMind 的哪个版本用了它？

<details>
<summary>参考答案</summary>

1. BatchNorm 依赖 batch 维统计，而 NLP 序列变长、batch 偏小、推理可能 batch=1，统计不稳且训练/推理不一致；LayerNorm 改成在单个 token 的特征向量上归一化，与 batch 无关。
2. Post-Norm 在残差相加之后归一化、深层难训；Pre-Norm 只归一化喂进子层的那份、残差是未归一化直通路。MiniMind 用 Pre-Norm，深层训练更稳定。
3. 去掉了「减均值」（中心化），只用 RMS 控制尺度；在 Transformer 里效果足够好且更省，还更完整保留方向信息。
4. 解决大模型/长训练时 `QK^T` 数值过大、softmax 饱和、注意力熵坍塌导致的训练不稳；它把归一化加在注意力**内部**的 Q、K 上（算 `QK^T` 之前），不同于前面几种作用在 block 主干上的 norm；MiniMind-3 用了它（q_norm/k_norm）。
</details>
