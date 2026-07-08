# 延伸：开源模型代际史——MiniMind 的骨架从哪来

读主线时你可能有个疑问没被正面回答:MiniMind 为什么长这样?为什么是 decoder-only + Pre-RMSNorm + SwiGLU + RoPE + GQA 这套组合,而不是别的?主线 [00-overview](../00-overview/01-what-is-minimind.md) 只说它是「标准 Transformer」,但「标准」二字背后,是十年模型演进一步步收敛出来的结果。

这一篇把这条线捋清:从 GPT 确立自回归范式,到 LLaMA 把现代 recipe 定型(**MiniMind 的骨架原型**),再到 Qwen、GLM 两大中文开源家族的吸收与演进。它和 [附录 13 DeepSeek 谱系](13-deepseek-lineage.md) 一起,构成 deep-dive 的「模型家族」卷。重叠的技术点(RoPE/MoE/MLA…)给链接略讲,重点是**各家的独有增量**和一个大问题:**为什么现代开源模型的骨架,几乎都收敛到了 MiniMind 这一套?**

一句话主线:**模型架构这些年在收敛:GPT 定下 decoder-only 自回归,LLaMA 把 RMSNorm/SwiGLU/RoPE 定为「三件套」,后来者(Qwen/GLM/DeepSeek/MiniMind)几乎全盘吸收,差异只在 MoE、长上下文、对齐这些「三件套之外」的地方。**

## GPT:确立 decoder-only 自回归

今天几乎所有聊天大模型的祖先逻辑,是 GPT 这条线定下的。

- **GPT-1**:证明「生成式预训练 + 下游微调」可行,只用左到右语言建模(next-token prediction),也能学到可迁移表示。结构是纯 decoder-only(只保留 masked self-attention,没有 encoder、没有 cross-attention),这一步就站定了「纯自回归生成」这条路。
- **GPT-2**:放大规模,提出 `pre-train + zero-shot`,语料够大够杂时很多有监督任务可以重写成自然语言条件生成、不必再微调。它还把 LayerNorm 从 post-norm 调成 **pre-norm**(深层更稳),这个选择一直延续到今天(附录 [02 归一化演进](02-normalization-evolution.md) 讲过 Pre-Norm 为什么赢)。
- **GPT-3**:175B 参数,把 `few-shot / in-context learning` 从零星现象做成可系统展示的能力,prompt 里给几个示例、不更新参数就能临时学会一类任务。它让「上下文也是编程手段」变成常识,后来的 instruction tuning、CoT、system prompt 都能追溯到这里。

GPT 留给后来者的,是两个几乎不再动摇的地基:**decoder-only 自回归**(MiniMind 也是)和 **Pre-Norm**。但 GPT-3 之后闭源了,真正把「现代开源 recipe」定型并交给社区的,是 LLaMA。

## LLaMA:把「三件套」定型(MiniMind 的骨架原型)

如果只读一段,读这段。**MiniMind 的骨架基本就是 LLaMA1 的 recipe**。

**LLaMA1** 在 GPT 的 decoder-only + Pre-Norm 基础上,把三个组件固定下来,合称现代 LLM 的「三件套」:

- **Pre-RMSNorm**:比 LayerNorm 少一步中心化,深层训练更稳(附录 [02](02-normalization-evolution.md))。
- **SwiGLU**:门控 FFN,表达力更强(主线 [02-model/05-swiglu](../02-model/05-swiglu.md))。
- **RoPE**:旋转位置编码,替代绝对位置嵌入(主线 [02-model/03-rope](../02-model/03-rope.md) + 附录 [09 位置编码谱系](09-positional-encoding-evolution.md))。

再配上一套稳定的优化 recipe(AdamW、余弦调度、weight decay 0.1、grad clip 1.0、warmup)。**对照 MiniMind 你会发现它几乎一模一样**——这不是巧合,而是 MiniMind 有意站在这套被反复验证的 recipe 上。LLaMA1 还有个生态贡献:给出 7B/13B/33B/65B 一整套「同 recipe 可扩展」的尺度,让社区能在自己算力下复现。

后面几代加的,都是「三件套之外」的东西:

- **Llama2**:引入 **GQA**(仅 34B/70B,降 KV cache,附录 [04](04-kv-cache-compression.md))、上下文 2K→4K;首次系统公开后训练方法论(下一段细讲)。
- **CodeLlama**:基于 Llama2 继续预训练,核心是 **infilling / fill-in-the-middle**(挖掉中段、保留前后缀让模型补全)——这是**训练目标层面**的独有设计,和普通 next-token 不同。
- **Llama3**:词表 32K→128K(token efficiency,利好代码/多语言);15T+ tokens;上下文最高 128K。主干**不追求离群创新**,继续沿用 GQA+RoPE+RMSNorm+SwiGLU——这本身就说明三件套已经稳定成标准。

**Llama2 的后训练方法论**特别值得写,因为它讲的是「HF 怎么组织」而非「用哪个 RL 算法」:

- **RM 语料要来自自己模型的分布**,而非 GPT-4 等外部分布(否则偏好落在分布外,RM 学不准);
- **SFT 不该训太久**(上万条快速塑形即可,标注员质量有天花板);
- **RLHF 是迭代闭环**(共 5 轮,像宏观的梯度下降),而且**前 4 轮更像 reject sampling/筛选,只有最后一轮才真上 PPO**——这提醒我们:RLHF 的效果大头常在「数据怎么迭代组织」,不全在 RL 算法本身。

**Llama3 的 DPO 三个工程细节**也很实用(和附录 [08 DPO 变体](08-dpo-variants.md) 呼应):① DPO loss **mask 掉格式化 token**(否则同一 token 概率既增既减,导致尾部重复或突然终止);② 给 chosen 加 **NLL 正则**(系数≈0.2,防 chosen 对数概率被一起压低——正是附录 08 讲的 DPOP 那类隐患);③ **model averaging**(平均不同 RM/SFT/DPO 阶段的权重)。

## Qwen:中文开源的体系化

Qwen 是 LLaMA-like 路线在中文开源上做得最体系的一家。结构起点(Qwen1)就是三件套 + 几个家族标志:

- **QKV 层保留 bias、其余线性层去 bias**(家族标志);
- **Untied Embedding**(输入/输出 embedding 不共享,性能优先);
- tokenizer 从 tiktoken 扩展、数字拆分、词表≈152K。

代际演进的主线是「从能做出来 → 成熟可部署 → 做厚 → 加推理」:

- **Qwen1.5**:三件套固定化,发布 AWQ/GPTQ 量化版(附录 [06 量化](06-quantization.md)),推出 **MoE-A2.7B**(4 共享 + 60 路由专家、激活 4 个——细粒度 + 共享专家,附录 [12 MoE 演进](12-moe-evolution.md))。
- **Qwen2/2.5**:长上下文用 **YaRN + DCA**(附录 [03 RoPE 长度外推](03-rope-length-extrapolation.md)),**ABF** 把 RoPE base 从 10000 提到 1000000(甚至 10000000)。
- **Qwen3**:dense + MoE 双路线;引入 **thinking / non-thinking 双模式 + thinking budget**——把「测试时算多久」显式交给调用侧(`/think`、`/no_think`)。

Qwen 的**独有增量**(deep-dive 没覆盖、值得记):

- **执行反馈 / 可验证奖励**:Qwen2 起大规模用「写验证器检查结果」做对齐信号——代码编译测试、指令跟随写验证函数、检查 JSON schema/字段。这套「客观可执行」的奖励,比训一个主观 RM 更硬、更省(呼应附录 [14 RL 全景](14-rl-alignment-landscape.md) 里 RLVR 的思路)。
- **thinking budget 统一接口**:把推理深度做成一个可调旋钮,而非两个模型。

## GLM:走过弯路又收敛回来

GLM 最有教学价值,因为它**试过一条不同的路,最后又收敛回 decoder-only**——这恰好反证了「为什么现代模型都是 causal decoder」。

- **GLM-1**:没走 GPT 的纯 decoder,而用 **prefix-decoder**(靠 attention mask 让前缀像双向 encoder、缺口像单向 decoder),训练目标是 **Autoregressive Blank Infilling**(挖空 + 自回归填空),还配了**二维位置编码**。这是和 GPT(纯自回归)、BERT(纯双向)都不同的**第三条路**。
- **GLM-2**:**回归 decoder-only causal**,换上 RoPE + RMSNorm + SwiGLU(三件套),MHA→MQA 降 KV cache。**回归的关键动机是多轮对话训练**——decoder-only 可以把一整段多轮对话写成一条序列、只在 assistant 片段算 loss,比 prefix-decoder 拆成多条样本简单得多。**这是理解「为什么现代模型都收敛到 causal decoder」的最佳案例**:不是 prefix-decoder 不行,而是 causal decoder 在对话数据的组织上天然更顺。
- **GLM-4**:三件套 + GQA(省下的容量给 FFN,隐藏维扩到 10/3)+ 统一 15 万词表。
- **GLM-4.5**:MoE 355B/激活 32B,定位 **ARC(agentic/reasoning/coding)**。几个设计和 DeepSeek 谱系可对照:**deep over wide**(比 DeepSeek-V3 减宽增深)、**QK-Norm**(附录 [02](02-normalization-evolution.md))、**loss-free balance routing + sigmoid 门控**(附录 [12](12-moe-evolution.md) 讲过 V3 的这招)、**额外一层 MoE 当 MTP 层支持投机解码**(附录 [07](07-speculative-decoding.md))。

## 落回 MiniMind

把四条线放一起,MiniMind 的骨架就不再是「凭空的标准 Transformer」,而是**一段收敛史的终点**:

| 组件 | 谁定型 | MiniMind |
|---|---|---|
| decoder-only 自回归 | GPT-1 | ✅ |
| Pre-Norm | GPT-2 | ✅(Pre-RMSNorm) |
| RMSNorm + SwiGLU + RoPE 三件套 | LLaMA1 | ✅ 全用 |
| GQA | Llama2 / Qwen / GLM4 | ✅ v2/v3 都用 |
| MoE(细粒度+共享专家) | Qwen/DeepSeek | ✅ v2/v3 可选 |
| QK-Norm | Qwen3/GLM4.5/DeepSeek | ✅ v3 新增 |

读这张表:MiniMind 用的每一个组件,都是这条代际史上被反复验证、最后被主流普遍采纳的选择。它不是「随便搭的教学模型」,而是**把 2024 年主流开源模型的骨架共识,浓缩到 26M 规模上**。所以读懂 MiniMind 的结构,某种意义上就读懂了「现代开源 LLM 长什么样、为什么长这样」。

GLM 那段「走 prefix-decoder 又回归 causal」的弯路尤其值得记:它证明了 MiniMind(以及几乎所有现代模型)选 decoder-only,不是路径依赖,而是被对话数据组织、训练简洁性反复筛选出来的更优解。

## 常见误区

- **「MiniMind 的架构是随便定的标准 Transformer」**——不。它是 GPT→LLaMA→Qwen/GLM 收敛史的终点,每个组件都有代际出处。
- **「decoder-only 是唯一可能的选择」**——不。GLM-1 试过 prefix-decoder + blank infilling 的第三条路,是 GLM-2 出于「多轮对话数据更好组织」才回归 causal decoder 的。
- **「LLaMA 发明了 RMSNorm/SwiGLU/RoPE」**——不准确。这些组件此前已存在,LLaMA 的贡献是把它们**组合固定成一套可复现的现代 recipe**并开源,让社区都跟上。
- **「新模型都在改架构主干」**——多数不是。Llama3、Qwen、GLM4 的主干都沿用三件套,真正的差异在 MoE、长上下文外推、对齐方法这些「主干之外」的地方。

## 练习

1. GPT 三代各确立了什么?哪两个地基被后来者(含 MiniMind)几乎全盘继承?
2. LLaMA1 的「三件套」是什么?为什么说 MiniMind 的骨架是 LLaMA1 recipe 的原型?
3. Llama2 的后训练方法论里,「RM 语料要来自自己模型分布」「PPO 只在最后一轮」各说明了什么?
4. GLM 从 prefix-decoder 回归 decoder-only 的关键动机是什么?这对「为什么现代模型都是 causal decoder」有什么启示?
5. 对照落回表,MiniMind 的每个组件分别由哪一代模型定型?这说明 MiniMind 的架构是怎么来的?

<details>
<summary>参考答案</summary>

1. GPT-1 证明生成式预训练可迁移(确立 decoder-only 自回归);GPT-2 提出 pre-train + zero-shot(并把 LayerNorm 改成 Pre-Norm);GPT-3 把 few-shot/in-context learning 做成可系统展示的能力。被全盘继承的两个地基:decoder-only 自回归 + Pre-Norm。
2. 三件套 = Pre-RMSNorm + SwiGLU + RoPE。MiniMind 用的正是这三件套 + 同款优化 recipe(AdamW/余弦调度/wd 0.1/grad clip 1.0/warmup),几乎和 LLaMA1 一致,所以说是 LLaMA1 recipe 的原型。
3. 「RM 语料来自自己模型分布」说明偏好数据要落在当前策略的分布内,否则 RM 在分布外学不准、PPO 追的是错的偏好;「PPO 只在最后一轮(前 4 轮是 reject sampling/筛选)」说明 RLHF 的效果大头常在「数据怎么迭代组织」,而非 RL 算法本身。
4. 关键动机是多轮对话训练:decoder-only 可以把整段多轮对话写成一条序列、只在 assistant 片段算 loss,比 prefix-decoder 拆多条样本简单。启示:现代模型选 causal decoder 不是路径依赖,而是被对话数据组织和训练简洁性筛选出来的更优解。
5. decoder-only←GPT-1、Pre-Norm←GPT-2、RMSNorm/SwiGLU/RoPE 三件套←LLaMA1、GQA←Llama2/Qwen/GLM4、MoE(细粒度+共享)←Qwen/DeepSeek、QK-Norm←Qwen3/GLM4.5/DeepSeek。说明 MiniMind 的架构是主流开源模型代际共识的浓缩,不是随意搭建。
</details>
