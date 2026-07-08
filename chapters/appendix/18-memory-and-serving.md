# 延伸：显存账本与推理系统——为什么纸面 50GB 实际跑到 75GB

MiniMind 的训练脚本里藏着几个你可能没细想的开关:`--dtype bfloat16`、`--accumulation_steps 8`、`DistributedDataParallel`(`train_pretrain.py`)。它们都是**显存优化**手段,但主线只把它们当训练流程的一部分带过,没讲背后那本「显存账」。这一篇把账本摊开:训练和推理的显存各由哪几块组成、哪一块最容易先爆、业界用什么招各个击破。

这是 deep-dive 里偏「系统工程」的一篇。它和前面几篇是互补的:附录 [06 量化](06-quantization.md) 压权重位宽、[04 KV cache 压缩](04-kv-cache-compression.md) 压缓存、[10 FlashAttention](10-flash-attention.md) 降 attention 的显存往返——它们都是这本账里的某一笔。这篇把它们放进一张完整账单,并补上一个 deep-dive 还没讲的推理系统核心:**PagedAttention**。

一句话主线:**显存不是一个数字,而是一张账单;所有显存优化都不是凭空变出显存,而是一种交换:时间换空间、空间转移、或精度换容量。**

## 为什么纸面估算总是不准

很多人第一次算显存,估个模型参数大小,然后发现和 `nvidia-smi` 差很多。这不是公式错,而是显存里从来不只装那几个能点名的大对象。分三层看:

- **可估算值**:模型参数、优化器状态、梯度、激活值、输入输出张量;
- **未命名数据**:临时 buffer、kernel workspace、内存碎片;
- **框架开销**:autograd、allocator 等。

所以「纸面 50GB、实际 75GB」一点不稀奇:大头通常仍是模型状态和激活,但序列一长、算子一复杂、allocator 行为不理想,未命名开销和碎片就被放大。**显存公式是「下限和主导项分析」,用来判断问题大头在哪,不能替代真实 profiling。** 排障两步:先用公式预算、预判大头,再用 profiler 对账。

## 训练显存:静态 + 动态四块

训练显存最适合分静态(形状固定)和动态(随 batch/序列波动)两类,拆成四块:

- **模型参数**:参数量 × 每参数字节数。fp32 按 4B,fp16/bf16 按 2B。
- **优化器状态**(常被低估的大头):Adam 除参数本体还要一阶矩、二阶矩,混合精度还常保留 fp32 master weight。**优化器状态常是好几份参数量级**——这正是 `8-bit optimizer`、`ZeRO`、offload 主要省的地方(不是省注意力矩阵)。
- **激活值**:一次 forward 要暂存多少中间结果供 backward 用。它随序列长度 `s`、micro-batch `b`、隐藏维 `h`、层数 `L` 一起变。参数像「常住人口」,激活像「当前这批样本的临时客流」:**一开长上下文就 OOM,常常不是参数变了,而是激活这笔临时账一下变大**。
- **梯度**:和参数形状一致,量级约「再来一份参数张量」。

算笔账感受一下:一个 `1B` 模型,参数和梯度 bf16 各约 2GB;Adam 带 fp32 master weight 再放约 4GB 主权重 + 两份各约 4GB 矩——**激活还没算,就已经 16GB 左右**。放大到 7B,单卡 24GB/48GB 很快不够。

这四块正好对应四类优化(下面「谁先爆用什么」细讲)。**MiniMind 在这本账上的选择**:用 `bfloat16`(参数/激活/梯度都减半)、`accumulation_steps=8`(梯度累积——用小 micro-batch 攒够大 batch,压激活峰值)、DDP(数据并行)。它没上 ZeRO/offload,因为 26M 模型单卡绰绰有余,这些重武器是给放不下的大模型准备的。

## 推理显存:权重 + KV cache + workspace

推理比训练简单——没有 optimizer、不留反向图。三块:

- **模型权重**(最稳定);
- **KV cache**(随上下文长度、输出长度、并发数、head 结构增长);
- **临时 workspace + 框架开销**。

训练时的「激活」在这里大幅缩小(decode 每步只走一个 token,不留反向图)。但多出一个主角:**KV cache**。把它想成「为避免重算旧 token 的 K/V 而付的长期租金」——单 token 不大,但随长度和并发**线性增长**。短 prompt、低并发时权重是大头;一旦长会话、高并发,KV cache 从「小尾巴」变成主角。

这就是为什么长对话/高并发系统在意 `GQA/MQA/MLA/PagedAttention`:它们都在动 KV cache 这块成本。前三个附录 [04](04-kv-cache-compression.md) 讲过(减份数、压表示),第四个是纯内存管理的招,下面细讲。

## PagedAttention:像操作系统管内存一样管 KV cache

PagedAttention(vLLM 的核心)和 FlashAttention 不同层:**Flash 优化「算 attention 时怎么少搬数据」(kernel 层),Paged 优化「KV cache 怎么放、怎么长、怎么回收」(内存管理层)**,两者可叠加。

它解决的痛点:自回归生成时每条请求的 KV cache 不断变长、且长度事先不知道。传统框架为每条请求预留一块**连续大区域**(按最大长度),导致大量过度预留和碎片——在线请求长度长尾(有人几百 token 结束、有人生成上万),保守预留会让内存浪费高达 60%–80%。

核心思路:**打碎「逻辑连续必须物理连续」的假设**,借操作系统虚拟内存的直觉。把每条序列的 KV cache 按固定长度切成「页」(块),逻辑块通过**块表**映射到物理块,物理块**按需分配、复用、回收**。请求长了就追加新块,不必一次性预留整条未来轨迹。vLLM 报告内存浪费能降到 4% 以下。

两个漂亮的衍生收益:

- **共享前缀**:并行采样/beam search 时多条序列共享同一段 prompt 前缀,KV cache 完全一样,就让多个逻辑块**指向同一组物理块** + 引用计数,不复制多份(vLLM 报告省约 55% 内存、最高 2.2x 吞吐)。
- **Copy-on-Write**:多条序列共享前缀块,一旦某条要写入共享块(引用计数 > 1),才为它分配新块、改指向——和 OS 处理共享页同一思路。「共享直到必须分开,再分开」。

块大小是权衡:太大尾块浪费多,太小块表管理开销重。PagedAttention 把 KV cache 从「预先整体占位的大对象」变成「按生命周期逐块申请释放的小对象」,这直接决定一张卡能挂多少并发、batch 能做多厚——**同样 7B 模型不同推理引擎吞吐差很多,大头常在这里,而非算子**。

## 谁先爆,用什么:一张决策表

显存优化的关键不是「找最强技巧」,而是「先解决当前最大那笔账」。把手段和「谁先爆」对应起来:

| 谁先爆 | 优先用什么 | 本质 |
|---|---|---|
| 参数本体太大 | bf16/fp16、量化、张量并行、流水线并行、ZeRO-3、offload | 精度换容量 / 空间转移 |
| 优化器状态太重 | 8-bit optimizer、ZeRO-1/2 | 压冗余 / 分摊 |
| 激活值太大 | activation checkpointing、降序列/micro-batch、FlashAttention | 时间换空间 |
| 推理 KV cache 顶不住 | GQA/MQA/MLA、KV 量化、Prompt Cache、PagedAttention | 减份数 / 压表示 / 管得紧 |

几个关键术语落进这张表:

- **activation checkpointing(重计算)**:少存中间激活,反向时再算一遍——典型的时间换空间。
- **ZeRO**:把参数副本/梯度/优化器状态的冗余在多卡间切开,ZeRO-1/2/3 逐级切更多。
- **offload**:把挤不下的东西(优化器状态/参数)搬到 CPU/NVMe,吃 I/O 带宽换容量。

所有优化都有代价、还会互相干扰:checkpointing 省激活但增重算时间,ZeRO-3 省参数副本但通信可能成新瓶颈,offload 救容量但 PCIe 拖速度。**「把所有省显存开关全打开」往往不是最优解**——工程顺序是先动最便宜、最影响大头的(bf16、Flash、checkpointing),参数实在放不下再上重并行和 offload。

## 落回 MiniMind

MiniMind 是这本账的「轻量版样本」:26M 模型,单卡就能训、能推,所以它只用了最基础的几招——bf16 减半、梯度累积压激活峰值、DDP 数据并行。它**没上** ZeRO、offload、张量/流水线并行、PagedAttention,因为这些都是「单卡放不下、单请求扛不住」时才需要的重武器,对一个教学小模型是杀鸡用牛刀。

但正因为 MiniMind 小,它反而是理解这本账的好起点:你能在一张卡上清楚看到参数、梯度、优化器状态、激活各占多少(主线 [10 章实验](../10-experiments/02-server-training-records.md) 有真实训练显存记录),再把每一块放大想象到 7B、70B,就明白为什么大模型训练/推理要动用那一整套 ZeRO、offload、PagedAttention。**读懂 MiniMind 的显存,再读这张账单,就读懂了大模型「训得起、跑得动」背后的系统工程。**

## 常见误区

- **「显存 = 模型参数大小」**——远不止。训练还有优化器状态(常是几份参数量级)、梯度、激活;推理还有 KV cache、workspace;加上碎片和框架开销,实际常比纸面高 50%。
- **「显存优化是凭空变出显存」**——不。它永远是交换:重计算用时间换、并行/offload 用空间转移、量化用精度换,每招都有代价。
- **「PagedAttention 和 FlashAttention 二选一」**——不。Flash 在 kernel 层优化 attention 计算,Paged 在内存层管理 KV cache,不同层、可叠加。
- **「把所有省显存开关全开最省」**——不。它们会互相干扰(checkpointing 增重算、ZeRO-3 增通信、offload 拖 I/O),该按「谁先爆」精准下药。
- **「MiniMind 该上 ZeRO/PagedAttention」**——不。它 26M、单卡够用,这些是大模型放不下时才需要的,对小模型是过度工程。

## 练习

1. 训练显存和推理显存各由哪几块组成?为什么推理没有「优化器状态」这块?
2. 为什么「纸面估 50GB、实际 75GB」很常见?显存公式的正确用途是什么?
3. 为什么说优化器状态常是「好几份参数量级」?哪些手段专门省它?
4. PagedAttention 借了操作系统的什么思路?它和 FlashAttention 分别在哪一层优化?
5. 面对 OOM,为什么不该「把所有省显存开关全打开」?正确的下药顺序是什么?MiniMind 用了其中哪几招、为什么不用其余的?

<details>
<summary>参考答案</summary>

1. 训练:模型参数 + 优化器状态 + 梯度 + 激活值(静态两块 + 动态两块);推理:模型权重 + KV cache + 临时 workspace。推理没有优化器状态,是因为不做参数更新(无 Adam 的一阶/二阶矩、无 master weight),也不留反向图。
2. 因为显存不只装能点名的大对象,还有未命名数据(临时 buffer、kernel workspace、碎片)和框架开销(autograd、allocator);序列长、算子复杂、allocator 行为不理想时这些被放大。公式是「下限和主导项分析」,用来判断大头在哪,不能替代真实 profiling。
3. 因为 Adam 除参数本体还要维护一阶矩、二阶矩,混合精度还常保留 fp32 master weight,加起来常是参数的好几倍。专门省它的:8-bit optimizer(把矩从 4B 压到 1B)、ZeRO-1/2(把优化器状态/梯度在多卡间切开)、offload(搬到 CPU/NVMe)。
4. 借了虚拟内存的「逻辑连续不必物理连续」:KV cache 切成页、逻辑块经块表映射到物理块、按需分配回收。FlashAttention 在 kernel 层优化 attention 怎么算(少搬数据),PagedAttention 在内存管理层优化 KV cache 怎么放/长/回收,两者可叠加。
5. 因为各招会互相干扰(checkpointing 增重算时间、ZeRO-3 增通信、offload 拖 I/O 带宽),全开往往不是最优。正确顺序:先用最便宜、最影响大头的(bf16、FlashAttention、activation checkpointing、调 micro-batch + 梯度累积),参数和优化器仍太胖再上 ZeRO-2/3、offload。MiniMind 用了 bf16、梯度累积、DDP;没用 ZeRO/offload/张量并行/PagedAttention,因为 26M 模型单卡够用,这些是大模型放不下时才需要的重武器。
</details>
