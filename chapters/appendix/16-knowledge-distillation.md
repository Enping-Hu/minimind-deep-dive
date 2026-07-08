# 延伸：知识蒸馏——让小模型继承大模型的能力

附录 [01 进阶入口](01-advanced-pointers.md) 给了 MiniMind `train_distillation.py` 的蒸馏最小原理(软标签 + 温度 + `α·CE + (1-α)·KL`)。这一篇把它扩开:MiniMind 的蒸馏逐行讲清,再放进完整的蒸馏谱系(白盒/黑盒/R1 蒸馏),并和 [量化](06-quantization.md) 一起放进「模型压缩」的框架里。

它能回答一个实用问题:**手里有个强大的大模型,想要一个又小又快、但尽量不掉能力的模型,除了量化压缩,还能怎么办?**

一句话主线:**量化和剪枝是在同一个模型上「压」,蒸馏是另起一个小模型去「学」大模型,所以它既能压缩、又能迁移能力。**

## 从 MiniMind 的蒸馏源码讲起

MiniMind 的 `train_distillation.py` 是标准的**白盒 response-based 蒸馏**,核心就两个 loss 加权。

先看蒸馏损失(`distillation_loss`):

```python
teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()   # 老师的软标签
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
return (temperature ** 2) * kl
```

关键在**软标签**:student 不只学「正确答案是哪个 token」(硬标签),还学 teacher 给整个词表分配的**概率分布**。teacher 认为「猫」概率 0.6、「狗」0.3、「鸟」0.1,这个分布里藏着「猫和狗比鸟更接近」的暗知识,比一个硬标签「猫」信息量大得多。

两个细节:

- **温度 `T`**:`logits / T` 让分布更平滑,放大那些次高概率的相对差异(把 teacher 的暗知识「摊开」让 student 看清)。`T` 越大越平滑。
- **`T²` 补偿**:除以 `T` 会让梯度缩小 `T²` 倍,所以损失乘回 `T²` 保持梯度尺度。

再看总损失(`train_epoch`):

```python
ce_loss = F.cross_entropy(...)          # 1) 对 ground-truth 硬标签的交叉熵
distill_loss = distillation_loss(...)   # 2) 对 teacher 软标签的 KL
loss = alpha * ce_loss + (1 - alpha) * distill_loss   # 3) 加权组合
```

student 同时学两个信号:**真实标签(CE)兜底正确性,teacher 软标签(KL)传递暗知识**。`alpha` 是两者的权重旋钮。teacher 全程冻结(`teacher_model.requires_grad_(False)`)。

还有一个容易忽略的**工程细节**:MiniMind 允许 teacher 和 student 词表大小不同,代码里 `teacher_logits = teacher_logits[..., :vocab_size_student]` 把 teacher 的 logits **截断对齐**到 student 词表——白盒蒸馏要求两者词表能对上,这是最简单的对齐方式。

## 蒸馏的三个要素

MiniMind 的实现是蒸馏最基础的一档。把视野放大,任何蒸馏都由三部分组成:

- **知识**(传什么):输出分布 / 中间特征 / 层间关系 / 生成的数据。
- **蒸馏算法**(怎么传):KL 对齐 / 特征匹配 / 用生成数据微调。
- **师生架构**(谁教谁):teacher 通常比 student 大。

按**对 teacher 的访问程度**,蒸馏分两大类,这是最重要的分野。

## 白盒蒸馏:拿得到 teacher 的输出分布

白盒蒸馏(标准 KD)能访问 teacher 的参数或输出分布——teacher 通常是开源模型。**MiniMind 的蒸馏就是白盒**:它直接拿 teacher 的 logits 算 KL。

按「传哪种知识」,白盒又分三层,信息一层比一层细:

- **Response-based**:学 teacher 的最终输出分布(软标签)。最直接,MiniMind 用的就是这个。
- **Feature-based**:连 teacher 的中间层特征也学。难点在师生结构不同——选 teacher 哪一层、对应 student 哪一层、尺寸不一致怎么匹配,至今没有统一方案。
- **Relation-based**:学「关系」而非单点——如层间特征的 Gram 矩阵、样本间的表示分布。

按「师生怎么更新」还分:**Offline**(先训好 teacher 再蒸,最主流,如 BERT→TinyBERT、DeepSeek-R1→R1-Distill)、**Online**(师生同时更新,针对没有现成强 teacher 的情况)、**Self**(师生同结构)。

## 黑盒蒸馏:只能看到 teacher 的文本输出

黑盒蒸馏拿不到 teacher 内部(teacher 通常是闭源的 GPT/Claude),只能让它**生成一个数据集**,再用这个数据集微调 student。对 LLM 来说,要转移的往往是大模型的**涌现能力**,所以又叫 Emergent Abilities-based KD,主要蒸三种:

| 能力 | 蒸什么 | 代表做法 |
|---|---|---|
| ICL(上下文学习) | 少样本学习能力 | 上下文学习目标 + 语言建模目标结合 |
| **CoT(思维链)** | 中间推理步骤 | MT-CoT / Fine-tune-CoT / SOCRATIC CoT 等 |
| IF(指令跟随) | 靠任务描述执行新任务 | 让 teacher 生成困难指令增强 student |

CoT 蒸馏是当下最热的一支(把大模型的推理能力搬到小模型),几条思路:`Fine-tune-CoT` 让 teacher 对一题采样多条推理解扩数据;`SOCRATIC CoT` 训「问题分解器 + 子问题解决器」;`SCOTT` 用对比解码把理由和答案绑定,让 student 学真正相关的推理依据。

**DeepSeek-R1-Distill** 是最有名的例子(附录 [13 DeepSeek 谱系](13-deepseek-lineage.md) 提过):把 R1 的 reasoning pattern 蒸到 Qwen/Llama 的 1.5B~70B dense 模型,证明「强推理能力可以迁移到小模型」,不必人人训超大模型。它介于白盒黑盒之间——用 R1 生成推理数据(黑盒式),但 teacher 开源(可白盒)。

## 落回 MiniMind：蒸馏在压缩版图里的位置

把蒸馏和 [量化](06-quantization.md) 放一起,「模型压缩」有三条正交主线:

| 方法 | 怎么「省」 | MiniMind |
|---|---|---|
| **量化** | 把每个权重表示得更省(FP→INT) | 无(端侧导出时才用,附录 06) |
| **剪枝** | 直接删掉冗余权重/结构 | 无 |
| **蒸馏** | 另训一个小模型继承大模型能力 | ✅ `train_distillation.py` |

三者正交、常组合(如先蒸出小模型再量化部署)。MiniMind 只实现了蒸馏,而且是最基础的白盒 response-based 档——teacher logits 截断对齐 student 词表、软标签 KL + 硬标签 CE 加权。想在它上面进阶:

- 换 **R1 蒸馏**:用一个推理强的 teacher(如 R1-Distill)产 CoT 数据,蒸出会推理的小 MiniMind,这就是白盒 KL 软标签蒸馏 R1 那条路;
- 组合 **蒸馏 + 量化**:蒸出小模型后再量化到 INT4 端侧部署;
- 复用为**投机解码 draft**:蒸出的小模型和大模型同源、又便宜,天然满足附录 [07 投机解码](07-speculative-decoding.md) 里 draft「猜得准、猜得快」的要求。

蒸馏和 DPO/RL 的 reference 有一点相通又不同:都「冻结一个模型当参照」,但**蒸馏是让 student 逼近 teacher 分布(学得像),DPO/RL 是约束 policy 别漂移(别学坏)**——一个是拉近,一个是拴住。

## 常见误区

- **「蒸馏和量化是一回事」**——不。量化在同一个模型上把权重表示压小,蒸馏是另起一个小模型去学大模型,两者正交、可组合。
- **「软标签只是带噪的硬标签」**——不。软标签的价值恰在非最大项:teacher 给「猫 0.6/狗 0.3/鸟 0.1」这个分布传递了「猫狗比鸟接近」的暗知识,硬标签「猫」没有。
- **「温度只是随手加的超参」**——不。`/T` 平滑分布、放大次高概率的相对差异让 student 看清暗知识,`×T²` 补偿梯度尺度,两者配套。
- **「白盒黑盒差在模型大小」**——不。差在**对 teacher 的访问程度**:白盒拿得到输出分布/参数(开源 teacher),黑盒只能拿到文本输出(闭源 teacher)、要靠 teacher 生成数据。
- **「MiniMind 的蒸馏能蒸不同词表的 teacher」**——有限制。它靠 `teacher_logits[..., :vocab_size_student]` 截断对齐,要求词表能这样对上;真正跨 tokenizer 的蒸馏要复杂得多。

## 练习

1. 蒸馏和量化、剪枝同属压缩,三者「省」的方式有何本质不同?
2. 对照 MiniMind 的 `distillation_loss`,软标签为什么比硬标签信息量大?温度 `T` 和 `T²` 各起什么作用?
3. MiniMind 的总 loss 是 `α·CE + (1-α)·KL`,这两项各自负责什么?teacher 为什么要冻结?
4. 白盒蒸馏和黑盒蒸馏的核心区别是什么?DeepSeek-R1-Distill 属于哪一类、为什么?
5. MiniMind 蒸馏怎么处理 teacher 和 student 词表大小不同?这种做法有什么前提?

<details>
<summary>参考答案</summary>

1. 量化把每个权重表示得更省(如 FP16→INT4),剪枝直接删掉冗余权重/结构,蒸馏另训一个小模型来继承大模型能力。前两者在同一模型上「压」,蒸馏是另起小模型去「学」,所以蒸馏既能压缩又能迁移能力。
2. 软标签是 teacher 给整个词表的概率分布,非最大项里藏着暗知识(如「猫 0.6/狗 0.3/鸟 0.1」传递了猫狗比鸟接近),硬标签只有一个正确 token。温度 `T`:`logits/T` 平滑分布、放大次高概率的相对差异让 student 看清这些暗知识;`T²`:补偿 `/T` 带来的梯度缩小,保持梯度尺度。
3. CE 对 ground-truth 硬标签,兜底正确性;KL 对 teacher 软标签,传递暗知识。`α` 是两者权重。teacher 冻结(`requires_grad_(False)`)是因为它只当「参照/老师」,不参与更新,只提供软标签目标。
4. 核心区别是对 teacher 的访问程度:白盒能拿到 teacher 输出分布/参数(开源 teacher,直接算 KL),黑盒只能拿到文本输出(闭源 teacher,靠它生成数据集再微调 student)。DeepSeek-R1-Distill 介于两者之间——用 R1 生成推理数据(黑盒式),但 teacher 开源可白盒对齐。
5. 用 `teacher_logits[..., :vocab_size_student]` 把 teacher 的 logits 截断到 student 词表大小来对齐。前提是两者词表能这样对上(比如 student 词表是 teacher 的前缀子集);真正跨 tokenizer/词表不一致的蒸馏要复杂得多,不能简单截断。
</details>
