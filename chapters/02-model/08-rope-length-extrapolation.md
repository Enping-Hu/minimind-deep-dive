# 延伸：RoPE 长度外推——从 MiniMind 那个默认关闭的开关说起

[03-rope](03-rope.md) 讲了 MiniMind 的 RoPE 怎么按位置旋转 Q/K，末尾「长上下文与 YaRN」一节点到为止：源码 `precompute_freqs_cis` 里有个 `rope_scaling`，默认关。这一节把那个开关打开——它背后是一整条**长度外推**的技术脉络：PI、NTK-aware、NTK-by-parts、Dynamic NTK、YaRN，每一步都在解决前一步的什么问题。

本节是**延伸 survey**，但它不空谈：MiniMind v2 的 `precompute_freqs_cis` 里**真的实现了 YaRN**（`inference_rope_scaling=True` 就启用），所以谱系的终点能逐行落回源码。读它能回答一个上长上下文时绕不开的问题：**训练只见过 2K，推理想上 32K，RoPE 为什么会崩、又该怎么救？**

一句话主线：**长度外推，就是在反复回答「别让模型硬闯没见过的位置，怎么把长位置映射回它熟悉的表示」。**

## 为什么直接外推会崩

RoPE 给不同维度对绑定不同频率：靠前的维度对频率高、转得快，靠后的低、转得慢（[03-rope](03-rope.md) 讲过 `θ_i = rope_base^(-2i/dim)`）。训练时只见过 `0~L` 这段位置，模型学到的是这段内的旋转规律。

推理时把位置直接扩到远大于 `L`，高频维度会在很短距离内快速绕圈，Q 和 K 的点积出现训练时没见过的剧烈波动，attention 分数异常、长上下文性能突然掉下去。所以长度外推的难点不是「模型看不懂更长的文本」，而是**位置系统在更长区间里失真了**。

一个「数字编码」的类比很贴切：模型训练时只见过 `0~999`，直接塞一个更大的数，要么新增一位、要么让原来每一位承担没见过的取值，都带来分布漂移；反过来把大范围压回旧范围，模型还认得，但每个位置之间的分辨率变差。**后面所有方法，本质都在这两件事之间权衡：让模型见全新位置值，还是继续看旧范围的值但接受表示被压缩。**

## PI：把长区间整体压回去

Positional Interpolation（位置内插）是最直接的补救：不往外推，而是把更长区间**整体线性压回**预训练见过的范围。模型原来支持 `2048`、想支持 `4096`，就把 `[0,4096]` 映射回 `[0,2048]`。位置 `4096` 不再对应陌生编码，而是变成熟悉的 `2048` 附近，位置 `1` 变成 `0.5`。

它很稳：扩到 `8192` 时不微调的 PI 已能把困惑度控制在可用范围，而直接外推可能直接炸；再少量微调，性能很快回升。

但 PI 有个粗暴之处：**对所有维度一视同仁地缩放**。而 RoPE 不同维度频率不同：低频维度周期长、压一下问题不大；高频维度周期短，本来一个周期就只覆盖一小段位置，统一压缩后变得特别拥挤，局部位置关系反而更容易糊掉。PI 是用「全体一起变密」换长度。

## NTK-aware：高低频不能一刀切

NTK-aware 的出发点就是别让所有频率一起受同样的压缩。核心一句话是**高频外推、低频内插**：减少对高频区域的缩放、增加对低频区域的缩放，把原来集中在高频的失真分散到不同频率上。相比 PI 更像「插值和外推混着来」。

缺点是某些维度仍会轻微越过训练边界，所以微调结果有时不如 PI；而且因为存在「越界」值，理论尺度因子并不能准确描述真实扩展尺度，实践中常要把缩放因子设得比名义倍数更大。

## NTK-by-parts：按波长分段

NTK-by-parts 进一步把「不同维度波长不同」说清楚。**波长**是某维度完成一次完整旋转所需的 token 长度：

- 波长**远短于**上下文的维度：主要编码近距离、相对位置的细粒度变化，应尽量少插值，保住局部分辨率；
- 波长**接近或超过**上下文的维度：更像编码粗粒度甚至接近绝对位置的信息，可以多做插值；
- 中间区域：用平滑过渡衔接。

它把「按频率缩放」推进成「按功能分段缩放」，避免统一缩放把该保留的局部关系一起抹平。

**这一步正是 MiniMind 源码实现的东西**。`precompute_freqs_cis` 里那段（`model_minimind.py:158–164`）：

```python
inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
ramp = torch.clamp((torch.arange(dim // 2) - low) / max(high - low, 0.001), 0, 1)
freqs = freqs * (1 - ramp + ramp / factor)
```

对着上面读：`inv_dim(beta_fast/beta_slow)` 算出 `low/high` 两条边界，正是「波长短到该保留」和「波长长到该内插」的分界；`ramp` 在 `low~high` 之间做 `0→1` 的线性过渡，就是那段「平滑衔接」；`freqs * (1 - ramp + ramp/factor)`——`ramp=0`（高频端）频率不动（外推），`ramp=1`（低频端）频率除以 `factor`（内插），中间线性混合。**这就是 NTK-by-parts 的分段缩放。**

## Dynamic NTK：缩放随长度动态调

Dynamic NTK 解决另一个现实问题：推理时序列长度不是一上来就固定为最大值，而是自回归**从短到长递增**（每步加一）。若缩放因子从头到尾不变，短序列阶段会白白承受长序列配置的精度损失，跨过某阈值后又可能突然退化。

它让缩放因子随当前序列长度动态变化：序列还短时位置映射尽量接近原模型，逐渐变长再平滑把缩放推上去。（MiniMind 的实现是静态的 `factor`，没做这一步动态——但知道方向即可。）

## YaRN：工程化整合 + 温度修正

YaRN（Yet another RoPE extensioN method）是这些思路的工程化整合，两个关键点：

1. 继续沿用「对 RoPE 做缩放并结合分段处理」（即上面的 NTK-by-parts 骨架），更细致地处理不同频段映射；
2. 在 attention logit 侧引入**温度修正**——让长度扩展后不同位置上的打分分布更平稳，不只管「位置怎么转」，也管「转完后 attention 分数会不会过尖锐或紊乱」。

这个温度修正就是源码里的 `attention_factor`（`attn_factor`）：

```python
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
```

它直接乘在 cos/sin 表上，等价于给 attention 分数一个统一的温度缩放。YaRN 还不给推理增加额外主干计算（RoPE 本来就要算），所以在不少 LLaMA 长上下文扩展实践里很受欢迎。

## 落回 MiniMind

看懂谱系，再看 `MiniMindConfig` 那个开关就通透了（`model_minimind.py:56–65`）：

```python
self.rope_scaling = {
    "beta_fast": 32, "beta_slow": 1,
    "factor": 16, "original_max_position_embeddings": 2048,
    "attention_factor": 1.0, "type": "yarn"
} if self.inference_rope_scaling else None
```

- **默认 `inference_rope_scaling=False`**，所以 `rope_scaling=None`，`precompute_freqs_cis` 走原始频率、不缩放——MiniMind 训练/正常推理都是纯 RoPE。
- 打开它，`factor=16 × orig_max=2048 = 32768`，即把外推目标设到 32K。此时那段 NTK-by-parts + YaRN 温度修正才生效。
- 为什么默认关？因为 YaRN 是**给已训好的模型做推理期长度扩展**用的：MiniMind 训练本身没上 32K 长序列，日常也不需要，开着反而改变短上下文行为。它是「想上长上下文时才拨」的开关，不是常开件。

于是 MiniMind 在这条谱系上的位置很清楚：**代码里备好了 YaRN（NTK-by-parts + 温度修正）这一档，默认不启用**。想复现长上下文扩展，把 `inference_rope_scaling` 打开、必要时补一轮长序列微调即可。

## 一张表看懂演进

| 方法 | 相对前一步的关键改动 | 代价 | MiniMind |
|---|---|---|---|
| 直接外推 | —（基线：直接用更大位置） | 高频维度剧烈波动、性能骤降 | 默认行为（不开 scaling） |
| PI | 长区间**整体线性压回**旧范围 | 所有维度一刀切，高频关系变糊 | — |
| NTK-aware | **高频外推、低频内插**，分散失真 | 部分维度仍越界，尺度因子不准 | — |
| NTK-by-parts | 按**波长分段**缩放 + 平滑过渡 | 需定 `beta_fast/slow` 边界 | ✅ `precompute_freqs_cis` 实现 |
| Dynamic NTK | 缩放因子随**当前长度**动态调 | 实现更复杂 | 未做（静态 factor） |
| YaRN | 分段缩放 + attention **温度修正** | 需调温度 | ✅ `attn_factor` + `type:"yarn"` |

读这张表的方式：每一行不是「更高级取代上一行」，而是**回答上一行留下的问题**——直接外推会崩→PI 压回旧范围；PI 一刀切→NTK-aware 分高低频；高低频还不够细→NTK-by-parts 按波长分段；长度会变→Dynamic NTK 动态调；分布还会尖锐→YaRN 补温度。MiniMind 的 `rope_scaling` 正好停在 YaRN 这一档。

## 常见误区

- **「长度外推要改 attention 主体」**——不。改的只是 RoPE 的频率表（`precompute_freqs_cis`）和一个温度系数，attention 计算本身一行不动。
- **「PI/NTK/YaRN 是三种互斥方案」**——不。它们是同一条线上逐步改进：YaRN 内含 NTK-by-parts 的分段骨架，NTK-by-parts 是 NTK-aware 的细化。
- **「MiniMind 没有长上下文能力」**——不准确。代码里 YaRN 齐全，只是 `inference_rope_scaling` 默认关；它缺的是长序列训练/微调，不是外推机制。

## 练习

1. 训练只见过 2K，直接把位置扩到 32K，RoPE 为什么会崩？崩在高频还是低频维度？
2. PI 和 NTK-aware 的核心区别是什么？为什么说 PI「一刀切」有问题？
3. 「波长」指什么？NTK-by-parts 为什么按波长决定某个维度多插值还是少插值？
4. 对照 `precompute_freqs_cis` 的源码，`low/high`、`ramp`、`freqs * (1 - ramp + ramp/factor)` 分别对应 NTK-by-parts 的哪一步？
5. MiniMind 的 `inference_rope_scaling` 为什么默认关？打开它需要配套做什么？

<details>
<summary>参考答案</summary>

1. 因为模型只学过 `0~L` 段的旋转规律，扩到远大于 `L` 时高频维度在很短距离内快速绕圈，Q/K 点积出现训练没见过的剧烈波动；崩在**高频**维度（转得快、最先进入陌生区间）。
2. PI 对所有维度**统一线性压缩**位置；NTK-aware **高频少缩、低频多缩**，把失真分散到不同频率。PI 一刀切的问题：高频维度本来周期就短、覆盖位置少，统一压缩后局部位置关系变拥挤、变糊。
3. 波长 = 某维度完成一次完整旋转所需的 token 长度。波长远短于上下文的维度编码近距离细节，应少插值保分辨率；波长接近/超过上下文的维度编码粗粒度信息，可多插值——所以按波长分段。
4. `inv_dim(beta_fast/slow)` 给出 `low/high` 两条波长边界（该保留 vs 该内插的分界）；`ramp` 是 `low~high` 间 `0→1` 的线性过渡（平滑衔接）；`freqs * (1 - ramp + ramp/factor)` 在高频端（ramp=0）不缩放、低频端（ramp=1）除以 factor、中间混合——即分段缩放。
5. 默认关是因为 YaRN 用于**推理期给已训模型扩长度**，而 MiniMind 训练没上长序列、日常也不需要，开着会改变短上下文行为。打开需把 `inference_rope_scaling=True`，并通常配一轮长序列微调让模型适应扩展后的位置分布。
</details>
