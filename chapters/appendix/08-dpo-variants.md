# 延伸：DPO 变体家族——各自在补 DPO 的哪个洞

主线 [06-dpo/02](../06-dpo/02-dpo-loss-and-math.md) 把 DPO 讲透了：它把「训 reward model + 跑 PPO」压成一个只用偏好对的离线对比损失，还专门有一节讲 `dpo_loss` 是**弱指标**——优化器常靠压低 rejected（而非抬高 chosen）来降 loss，极端时 chosen 概率自己也在降。那一节留下一个自然的问题：既然原始 DPO 有这些短板，后来的人怎么补？

这一篇就是答案。DPO 提出后衍生出一大批变体（SimPO、IPO、KTO、ORPO、DPOP…），名字多得像字母汤。但它们不是各自独立的新算法，而是**围绕 DPO 的几个固定短板分头改进**。抓住「每个变体在补哪个洞」，这串缩写就串成了一张图。

一句话主线：**DPO 变体大多在回答四个老问题:reference 太贵能不能去掉、loss 力道能不能调、chosen 绝对质量怎么守住、非成对数据能不能用。**

## 先列清 DPO 留下的洞

主线已经讲过其一（弱指标/差值隐患），这里把变体们针对的短板一次列全，后面每类变体对号入座：

- **洞一:还要带 reference model。** DPO 比 PPO 轻，但仍要一个冻结的 ref 算 `log π_ref`，多一份显存和前向。
- **洞二:更新力道不好调。** 全靠 `beta` 一个旋钮，太大容易 reward overfitting、把 chosen/rejected 分得失真。
- **洞三:只保相对排序，不保绝对质量。** 就是主线那个隐患:差值变大了，但 chosen 的绝对生成概率可能一起跌，模型「更保守了但也更闷了」。
- **洞四:必须要成对偏好数据。** 每个 prompt 都要配一个 chosen 和一个 rejected，标注贵；很多场景只有「这条好/这条坏」的单条反馈。

## 洞一:去掉 reference —— SimPO / CPO / ORPO

reference model 是 DPO 的锚，但也是负担。这类变体想省掉它。

- **SimPO** 直接不要 ref，用**序列平均 log 概率**当隐式奖励（`log π_θ(y|x) / |y|`），再配一个目标 margin 把 chosen/rejected 拉开。少一个模型、少一份前向，工程上更轻。
- **CPO** 同一条路，也走「尽量减少 reference 依赖」。
- **ORPO** 更彻底：把 SFT 和偏好优化揉进**一个**目标，用 odds ratio 同时表达「chosen 要更像、rejected 要更不像」，连「先 SFT 再 DPO」的两阶段都省了。

代价要清楚：去掉 ref 就去掉了那个「别漂太远」的锚。ref 在 DPO 里不只是算力负担，它还是防止 policy 整体乱抬乱压的基线坐标系（主线讲过这个作用）。去掉后得靠 margin、正则或 SFT 项补上约束，不是白省。

## 洞二:调 loss 力道 —— IPO / β-DPO

这类不动 DPO 主体，只改「拉开差距」的方式。

- **IPO** 关心的不是框架能不能用，而是原始 DPO 更新有时**太激进、太依赖 beta、太容易过拟合**。它换一种目标形式，让 chosen/rejected 的分离既能发生、又别分得太失真,更像给 DPO 的力道做校准。
- **β-DPO** 同线，围绕 `beta` 与 margin 调更新力道、抑制 reward overfitting。

主线说 `beta` 是「更新有多激进」的旋钮、点到为止。这一类就是把那个旋钮本身当成研究对象:固定的 beta 对所有样本一视同仁未必合适,难度不同的偏好对该有不同力道。

## 洞三:守住绝对质量 —— DPOP

这类直接回应主线那个弱指标隐患:排序对了还不够,chosen 的绝对生成概率也得守住。

**DPOP** 的做法是拿 SFT 模型当底线锚点。在 DPO loss 上加一个正则项,盯着 policy 对 chosen 的概率有没有跌破 SFT:

- policy 对 chosen 的概率**低于** SFT → 说明这个好答案还没学稳,减弱更新,别再把 chosen 往下压;
- policy 对 chosen 的概率**已高于** SFT → 好答案学稳了,火力转向压低 rejected。

一句话:DPO 只优化 chosen 相对 rejected 的排序,DPOP 额外加一个「chosen 别跌破 SFT 太多」的保护项。它防的正是主线描述的那个症状:模型排序学好了,却越来越不愿意把话讲开、输出变短变闷。这条线和 TDPO(把显式 KL 加回来、约束到 token 级、用 mass-covering 的 forward KL 保住输出多样性)是同一动机的两种打法,都在给「只顾排序」的 DPO 补一个「别把分布训坏」的约束。

## 洞四:换数据接口 —— KTO / RRHF

这类改的是「DPO 要吃什么数据」。

- **KTO** 不要成对偏好,只要**「这条好 / 这条坏」的单条标签**。它还借前景理论,把「人对坏结果更敏感」(损失厌恶)显式写进 loss——好样本和坏样本的增减不按同一个函数算。现实里单条好坏反馈比成对偏好便宜得多,KTO 承认这种粗粒度反馈也有价值。
- **RRHF** 吃**多候选排序**:手里若是多条回答的相对好坏顺序(而非干净的一对),直接用 ranking loss 逼近,比硬凑 pairwise 更自然。

## 还有一条正交的路:online 化

以上都在改 loss 或数据形式,但没跳出**离线**——数据给定,模型从固定 pair 里学分界。

**online DPO + rejection sampling** 补的是这件事:让当前模型先对一个 prompt 采样多条回答,再用 judge / reward model / 规则挑出更好的,重新组成偏好对继续训。相当于给 DPO 长出一点探索能力——不像 PPO 那样进完整 RL 环,但比纯离线多了「自己产数据、再反哺自己」的闭环。再往前一步的 Self-Reward 那类「模型给自己打分造偏好对」的方法,本质就是把这个闭环推到极致。

## 一张表看懂家族

| 洞 | 变体 | 怎么补 |
|---|---|---|
| 还要 reference | SimPO / CPO | 去掉 ref，用序列平均 log 概率 + margin |
| 还要 reference | ORPO | SFT 与偏好优化合并成一个目标（odds ratio） |
| loss 力道难调 | IPO / β-DPO | 换目标形式 / 调 beta，抑制过拟合 |
| 只保排序、丢绝对质量 | DPOP | 加 SFT 锚点保护项，chosen 别跌破 SFT |
| 只保排序、分布训窄 | TDPO | 加 token 级 forward KL，保输出多样性 |
| 必须成对数据 | KTO | 只需好/坏单标签 + 损失厌恶 |
| 必须成对数据 | RRHF | 吃多候选排序，ranking loss |
| 只能离线 | online DPO | 边采样边挑、重组偏好对，补探索 |

读这张表的方式和 [GRPO 变体家族](05-grpo-variants.md) 一样:别背缩写,记「问题 → 方法」。嫌两阶段重看 ORPO,嫌 ref 贵看 SimPO/CPO,嫌 beta 难调看 IPO,怕把 chosen 训垮看 DPOP,只有好坏标签看 KTO,有排序信号看 RRHF,想要探索看 online DPO。

## 落回 MiniMind

MiniMind 的 `train_dpo.py` 用的是**原始 DPO**:带 reference、成对偏好、离线、`beta` 默认 0.1(主线 [06-dpo/02](../06-dpo/02-dpo-loss-and-math.md) 逐行讲过)。放进这张家族图,它正好在原点:所有变体都是从这个原点出发补某个洞。

这也解释了主线那条「dpo_loss 是弱指标」纪律为什么重要:MiniMind 没上 DPOP/TDPO 这类保护项,所以原始 DPO 那个「压低 rejected、chosen 也可能跌」的隐患在它身上是真实存在的,只能靠盯 reward margin / accuracy + 下游 eval 来把关(主线已强调)。想在 MiniMind 上试变体,最小改动是 SimPO(去掉 ref 分支、改隐式奖励)或给 loss 加 DPOP 的 SFT 保护项——都不用动数据管线。

## 常见误区

- **「DPO 变体是一堆互相竞争的新算法」**——不。它们大多针对 DPO 的某个固定短板(去 ref / 调力道 / 保质量 / 换数据),是分工不是竞争。
- **「去掉 reference 纯是省显存、没代价」**——有代价。ref 还是防止 policy 整体乱漂的基线,去掉后得靠 margin / SFT 项补约束。
- **「KTO 是更强的 DPO」**——定位不同。KTO 主要放宽数据接口(单标签)+ 引入损失厌恶,不是在成对偏好上更强,而是能用 DPO 用不了的数据。
- **「上了变体就不用看 reward margin 了」**——仍要看。变体改的是优化方式,主线那套 DPO 健康度监控(margin/accuracy/KL/eval)依然适用。

## 练习

1. DPO 留下的四个「洞」分别是什么?试各举一个针对它的变体。
2. SimPO 怎么去掉 reference model?去掉 ref 的代价是什么?
3. DPOP 针对主线讲的哪个 DPO 隐患?它用什么当锚点、怎么保护 chosen?
4. KTO 和原始 DPO 在**数据接口**上最大的区别是什么?它为什么要引入「损失厌恶」?
5. online DPO 相比原始 DPO 补的是哪个维度?它和 PPO 的区别在哪?

<details>
<summary>参考答案</summary>

1. 四个洞:①还要带 reference model(SimPO/CPO/ORPO);②loss 力道难调(IPO/β-DPO);③只保相对排序、丢绝对质量(DPOP/TDPO);④必须成对数据(KTO/RRHF)。
2. SimPO 不用 ref,改用序列平均 log 概率 `log π_θ(y|x)/|y|` 当隐式奖励,配目标 margin 拉开 chosen/rejected。代价:去掉了 ref 这个「别漂太远」的基线坐标系,得靠 margin 补约束,否则 policy 容易整体乱抬乱压。
3. 针对「只保相对排序、chosen 绝对概率可能一起跌」(主线弱指标隐患)。DPOP 拿 SFT 模型当锚点:policy 对 chosen 概率低于 SFT 就减弱更新(别把 chosen 压没),高于 SFT 才把火力转向压低 rejected。
4. 原始 DPO 要成对的 chosen+rejected,KTO 只要「这条好/这条坏」的单条标签,数据接口更灵活、更便宜。引入损失厌恶是因为人对坏结果比对等量好结果更敏感,好/坏样本的增减不该按同一个对称函数算。
5. 补的是「离线 → 在线探索」这个维度:online DPO 让当前模型自己采样多条回答、挑出更好的重组偏好对继续训。和 PPO 的区别:它没有完整 RL 环(无 critic、无逐 token advantage 回传),只是多了一层「自产数据反哺自己」的轻量闭环。
</details>
