# 延伸：PEFT 全家族——从 MiniMind 的 LoRA 往外看

主线 [05-sft](../05-sft/01-assistant-only-supervision.md) 提过 MiniMind 的 `train_lora.py`：和 Full SFT 用同一个 `SFTDataset`、标签逻辑一致，区别只在**更新哪些参数**——Full SFT 更新全部，LoRA 冻结主干、只训注入的低秩旁路。附录 [01 进阶入口](01-advanced-pointers.md) 给了 LoRA 的最小原理。这一篇把它扩开：LoRA 逐行对上 MiniMind 源码，再把它放进整个 **PEFT（参数高效微调）** 家族——Prompt Tuning、Prefix、Adapter、LoRA 变体，各自在省什么、改什么。

PEFT 和 SFT/DPO 是**正交**的两个维度：后者管「用什么目标对齐」，PEFT 管「怎样以更低成本完成训练」。无论什么目标，全量更新几十亿参数都很贵，PEFT 的思路是冻住绝大多数权重、只训一小部分。

一句话主线：**PEFT 的所有路线都在回答同一个问题:冻住主干后，那一小撮可训练参数该放在哪、以什么形式存在。**

## 从 MiniMind 的 LoRA 讲起

LoRA 的核心判断:很多下游任务并不需要把整个权重矩阵都改一遍,真正要学的增量 `ΔW` 往往落在一个**低秩子空间**里。于是不直接更新原始权重 `W`,而把增量写成两个小矩阵的乘积 `ΔW = BA`,冻结 `W`、只训 `B`、`A`。

MiniMind 的实现(`model/model_lora.py`)极简、正好逐行对上这套原理:

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.A = nn.Linear(in_features, rank, bias=False)   # 降维 d→r
        self.B = nn.Linear(rank, out_features, bias=False)  # 升维 r→d
        self.A.weight.data.normal_(mean=0.0, std=0.02)      # A 随机初始化
        self.B.weight.data.zero_()                          # B 零初始化
    def forward(self, x):
        return self.B(self.A(x))                            # BA·x
```

三个细节都有讲究:

- **`B` 零初始化、`A` 随机初始化**:训练一开始旁路输出 `BA·x = 0`,不扰动主干的预训练权重;但不能两个都零,否则梯度传不动。「起步不扰动主干、又保留可学习性」,这个初始化就是那句话的实现。
- **秩 `rank`(默认 8)决定旁路容量**:原权重若是 `d×d`,LoRA 新增参数只有 `d×r + r×d = 2dr`。以 MiniMind 的 `hidden_size=512` 为例,一个 `512×512` 的层全量微调要更新约 26 万参数,LoRA 取 `r=8` 只新增 `512×8×2 = 8192`,降了一个多数量级。
- **旁路与主干并行相加**(`apply_lora` 里 `forward_with_lora` = `layer1(x) + layer2(x)`):即 `h = Wx + BAx`,两路同时算再相加。这正是 LoRA 区别于 Adapter 的关键:Adapter 是**串行**插进主干、推理时必须多跑一段,LoRA 是**并行**旁路、任务固定时甚至能把 `BA` 合并回 `W`(推理零额外开销)。

`apply_lora` 只给**方阵线性层**(`module.weight.shape[0] == shape[1]`)挂旁路,`train_lora.py` 也只把 lora 参数交给 optimizer、主干 `requires_grad=False`。存的时候只存 adapter(`save_lora`),所以能像插件一样切换(`lora_identity` / `lora_medical`)。这就是「一个底座 + 多套轻量 adapter」的部署方式。

## PEFT 的两条大路线

MiniMind 只用了 LoRA,但 PEFT 是一整个家族,分两条大路线——区别在「可训练参数以什么形式存在」:

- **加提示(soft prompt 线)**:不改权重,在输入或每层 attention 前塞一小段可训练向量,把模型「推」到适合某任务的状态。
- **加旁路/模块(低秩与结构线)**:在权重旁边挂一个小的可训练结构(低秩旁路或 bottleneck 模块),直接修正表示。LoRA 属这条。

下面各走一遍。

## soft prompt 线:提示从「输入层」一路走到「每层内部」

这条线的演化,是提示注入得越来越深:

- **Prompt Tuning**:最轻。在输入 embedding 前拼一小段可训练的连续向量(soft prompt),只更新这段、主干全冻。它把「该如何引导模型」也参数化了。短板:只在输入层动手,深层表示靠主干自己传播,底座不够强时表达力不足。
- **P-Tuning**:发现「光有一排随机软向量」不够,先用一个小 `Prompt Encoder`(Bi-LSTM/MLP)把这些虚拟 token 组织出内部结构再送进去。仍只动输入层,但让 soft prompt 更像一个有组织的小子网络。
- **Prefix Tuning**:关键一跳——不只在输入层,而在**每一层 attention** 都挂一组可训练前缀(改写成每层的 `K/V` 前缀)。提示从「embedding 层的对象」变成「attention 层的对象」,每个 block 都能读到任务条件,控制力强得多。
- **P-Tuning V2**:把 Prefix 这条路在更广任务上做完整——每层注入 deep prompt、回归标准分类头、提示长度按任务调。可训练参数从约 `0.01%` 提到 `0.1%~3%`,仍远小于全量。

一张 2×2 收束这条线,两根轴是**注入多深**和**有没有编码器加工**:

| | 无 / 弱编码器 | 带编码器加工 |
|---|---|---|
| **仅输入层** | Prompt Tuning | P-Tuning |
| **每层注入** | P-Tuning V2 | Prefix Tuning |

越往右下,提示越深入、表达力越强,但参数和实现复杂度也越高。

## 低秩与结构线:Adapter 与 LoRA

这条线不塞提示,而在权重旁边加可训练结构。

**Adapter** 在每层 Transformer 里插一个 bottleneck 小模块:先把隐藏状态降到小维度、做非线性变换、再升回原维度,通过残差合并(`h' = h + W_up·φ(W_down·h)`)。只训 `W_down`/`W_up`,主干冻结。它比 soft prompt 更直接地介入网络内部变换,适配更稳。代价:它是**串行**插进主干的,训练和推理都多一段计算。

**LoRA**(MiniMind 用的)针对 Adapter 的推理代价做改进:不新增串行模块,而给原线性层挂一个**并行低秩旁路** `ΔW = BA`。因为是并行相加,任务固定时能把 `BA` 合并回 `W`,推理零额外开销;要热切换多个 adapter 时不合并也行。所以 LoRA 兼顾了 Adapter 的有效性和更好的推理友好性,这也是它成为今天最主流 PEFT 方法的原因。

LoRA 的变体都围绕「省显存 / 训练效率 / 秩分配 / 表达能力」做局部增强:

| 变体 | 改什么 | 一句话 |
|---|---|---|
| `QLoRA` | 省显存 | 把冻结的基座以 4-bit 存(NF4 + double quant + paged optimizer),LoRA 分支仍高精度;`65B` 显存从 >780GB 压到 <48GB |
| `LoRA+` | 训练效率 | 不改结构,给 `A`、`B` 设不同学习率(`B` 约 `A` 的 6 倍) |
| `AdaLoRA` | 秩分配 | 按重要性动态分配各层的秩预算,裁掉不重要的奇异值 |
| `DoRA` | 表达能力 | 把权重的「大小」和「方向」拆开,只对方向上 LoRA |
| `VeRA` / `LoRA-FA` | 省参数 | 共享随机 `A/B` 只训小向量 / 冻结 `A` 只训 `B` |

一个常见误解顺手纠正:`QLoRA` 不是「所有东西都 4-bit」,而是把**冻结的基座**低比特存放,真正参与学习的 LoRA 参数仍保留高精度——它省的是大头存储,没牺牲可学习增量。

## 落回 MiniMind

MiniMind 在 PEFT 家族里的坐标很清楚:**它选了 LoRA,且是最基础的那一档**(纯 `BA` 旁路、`rank=8`、`A` 随机 `B` 零初始化,没上 QLoRA/DoRA 那些变体)。

放进两条大路线看,它的取舍合理:soft prompt 线适合「一个大底座服务多任务」、但对小模型表达力常不够(而 MiniMind 本就是小模型);LoRA 则直接、推理友好、实现极简,对一个 26M 的教学模型是最合适的 PEFT 入口。想在 MiniMind 上进阶,最小改动:

- 换 QLoRA——把主干量化存储(接上 [量化](06-quantization.md) 那篇),进一步省显存;
- 调 `rank` 或换 DoRA——看表达能力和参数量怎么权衡;
- 给更多矩阵挂 LoRA——MiniMind 现在只给方阵线性层挂,可以试着覆盖 `W_q`/`W_v` 之外的位置。

这些都不用动数据管线(`train_lora.py` 和 Full SFT 共用 `SFTDataset`),只改 adapter 的形式。

## 常见误区

- **「PEFT 是一种对齐目标,和 SFT/DPO 并列」**——不。PEFT 是「怎么省成本训」的维度,和「用什么目标训」正交;LoRA 既能做 SFT,也能做 DPO。
- **「LoRA 把大矩阵变小了」**——没有。原矩阵 `W` 一点没动,LoRA 只在旁边**新增**一个低秩旁路 `BA`,学的是增量。
- **「LoRA 和 Adapter 一样都会增加推理开销」**——不。Adapter 串行插入、推理必多跑;LoRA 并行旁路,任务固定时可合并回主干、推理零额外开销。
- **「QLoRA 把所有参数都 4-bit 训练」**——不。只把冻结基座低比特存,LoRA 分支仍高精度,省的是存储不是可学习增量。
- **「soft prompt 一定比 LoRA 差」**——看场景。soft prompt 在超大底座 + 多任务共享时很省,只是对小模型和需要深层改造的任务常不够,MiniMind 这种小模型用 LoRA 更合适。

## 练习

1. 为什么说 PEFT 和 SFT/DPO 是正交的两个维度?LoRA 能用在 DPO 上吗?
2. 对照 MiniMind 的 `LoRA` 类,为什么 `B` 零初始化、`A` 随机初始化?两个都零会怎样?
3. LoRA 和 Adapter 都在权重旁加小结构,推理开销为什么差别很大?
4. soft prompt 线从 Prompt Tuning 到 Prefix Tuning,最关键的一跳是什么?为什么它表达力更强?
5. QLoRA 到底把什么 4-bit 了、什么没动?为什么这样还能省大头显存而不太伤效果?

<details>
<summary>参考答案</summary>

1. 因为 SFT/DPO 决定「用什么目标/数据对齐」,PEFT 决定「怎么以更低成本完成这次训练(冻主干、只训一小部分)」,两件事互不冲突。LoRA 能用在 DPO 上——比如 QLoRA 冻结 4bit 主干、只训 LoRA 分支跑 DPO,就是常见组合。
2. `B` 零初始化让训练起步时旁路输出 `BA·x = 0`、不扰动主干预训练权重;`A` 随机初始化保证有非零梯度、能学起来。两个都零则旁路恒为 0、梯度也传不动,学不到东西。
3. Adapter 是串行插进主干的小模块,推理时数据必须多流过它一段;LoRA 是并行旁路 `h = Wx + BAx`,任务固定时可把 `BA` 合并回 `W` 得到新权重,推理零额外开销(要热切换多 adapter 时不合并也行)。
4. 最关键的一跳:从「只在输入 embedding 层加提示」到「在每一层 attention 都注入前缀(改写成每层 K/V)」。因为很多任务相关模式在中间层才形成,提示只停在输入层能否影响深层要看主干愿不愿传播;每层注入让提示直接进入各层注意力,减少了深层传播中的衰减。
5. QLoRA 把**冻结的基座权重**用 4-bit(NF4)存放,LoRA 可训练分支和关键状态仍保持高精度。基座是参数大头,低比特存它省下的显存最多;而真正参与学习的 LoRA 增量没被压,所以效果损失有限——省了大头存储、没牺牲最关键的可学习部分。
</details>
