# 延伸：RL 对齐算法全景——PPO 之外，还有一大片

主线 [ch07](../07-ppo-grpo/01-rl-overview.md) 讲了 MiniMind 实现的三种 RL(PPO/GRPO/SPO)、[附录 05](05-grpo-variants.md) 讲了 GRPO 之后的变体(Dr.GRPO/DAPO/GSPO/CISPO)、[附录 08](08-dpo-variants.md) 讲了 DPO 家族。但 RL 对齐的版图比这更大:MiniMind 借现成 RM、简化了经典 RLHF 的 Reward Model 那步;PPO/GRPO 之外还有一整条更轻的 REINFORCE 路线(ReMax/RLOO/REINFORCE++);以及 RLAIF、TTRL、VAPO 这些换奖励来源、换优化基座的方向。

这一篇把这片版图铺开:不重讲 GRPO 变体和 MiniMind 实现,而是补齐它们周围那些同样重要、MiniMind 没走的路。读完你会有一张「RL 对齐算法地图」,知道每个方法在这张图上站哪。

一句话主线:**RL 对齐的所有方法都在动同一套零件:奖励从哪来、baseline 怎么定、优势怎么估、要不要 value model、要不要 KL;换哪个零件,就得到哪个方法。**

## 经典 RLHF:三步范式

先补主线跳过的那一段。标准 RLHF(InstructGPT 的 `SFT → RM → PPO`)分三步,缺一不可:

1. **SFT**:把 base model 拉到「会按指令回答」。没这步,后面 RL 像让不会走路的人学跑,极不稳。
2. **训练 Reward Model**:给同一 prompt 采样多条回答,人工标注「哪条更好」(pairwise,比绝对打分稳)。RM 通常在 SFT 模型上接一个 `[hidden, 1]` 线性头,输入 `prompt+response`、输出标量分,**只取最后一个有效 token 位置的值**当整句奖励。训练损失是 `-logsigmoid(chosen_reward − rejected_reward)`,对应 **Bradley-Terry** 偏好模型——不要求奖励有绝对意义,只要求「同 prompt 下 chosen 分高于 rejected」。
3. **PPO 策略优化**:以 SFT 为初始 policy、冻结版为 reference,让 policy 生成、RM 打分、PPO 更新,同时用 KL 把 policy 拴在 reference 附近。

记成一句话:**先教会回答(SFT)、再教会偏好(RM)、最后教会取舍(PPO)**。这三步同时存在四个模型:policy(actor)、critic(value)、reward、reference,这正是 RLHF 工程成本高的根源,也是后面所有「减负」方法的靶子。

**MiniMind 走了简化版**:它不自己从零训练 Reward Model,而是**加载一个现成的外部 RM**(`internlm2-1_8b-reward`,冻结,`train_grpo.py` 用 `AutoModel.from_pretrained`)给回答打分,再叠加一部分**规则 format 奖励**(答案格式、标签是否齐全)。这省掉了「采集偏好数据 + 从头训 RM」这一大步,但保留了「RM 打分 + 在线生成 + KL 约束」的 RL 内核。

## RLAIF:把打分的人换成 AI

RM 依赖人工偏好标注,贵、慢、规模受限。RLAIF(AI 反馈的 RL)把偏好判断部分交给强模型:让 LLM 当 judge,对候选回答做排序/打分/critique,形成 AI 偏好数据,再喂给 PPO 或 DPO。

它不是「彻底不要人」,而是把人从重复标注里抽出来、去定标准和抽检。`Constitutional AI` 是代表:先给一套原则(帮助性/无害性/诚实),让模型基于原则做 self-critique 和修订。RLAIF 真正可扩展,是因为它把评判标准**文本化、流程化、批量化**了。

风险也清楚:judge 有偏会被放大;judge 和 policy 同源时容易「自己欣赏自己」;还有 rubric overfitting(学会迎合判卷标准而非真解决问题)。所以稳妥做法是混合——少量人工金标校准 judge,再用 AI 放大规模,高风险样本人工抽检。MiniMind 的 `RLAIFDataset` 命名正呼应这条线。

## PPO 之外的轻量路线:REINFORCE 家族

PPO 要 critic、要 GAE、要 clip,重。有一整条更轻的路线回到最原始的 REINFORCE(`用回报 × logprob 更新`),再各自改 baseline 来降方差。它们和 GRPO 是「兄弟」,都想干掉 critic,只是 baseline 构造不同:

| 方法 | baseline 怎么定 | 一句话 |
|---|---|---|
| **REINFORCE** | 无(或固定值) | 最原始:回报直接乘 logprob,方差极大 |
| **ReMax** | 同 prompt 的 **greedy 回答**得分 | `advantage = r(sample) − r(greedy)`,一个稳定锚点 |
| **RLOO** | 同组**其他回答**的均值(leave-one-out) | `A_i = r_i − mean(其余)`,组内中心化、不含自己 |
| **GRPO** | 同组均值 **÷ 标准差** | 组内标准化,主线 ch07 讲过 |
| **REINFORCE++** | baseline + 归一化 + KL + 稳定器 | 把经典策略梯度认真工程化 |

这几个的关系很清楚:**ReMax 是「一个 prompt、一条 greedy 基准」,RLOO 是「一个 prompt、多条采样互为参照」,GRPO 在 RLOO 基础上再除标准差、并结合 PPO 风格更新**。它们共同证明:critic 不是必需的,一个便宜的组内/贪心 baseline 就能把 RL 训起来。

关键提醒:这几个都是**序列级 credit assignment**——句末一个奖励,广播到整条回答所有 token。它们解决的是「回报方差大」,不是「到底哪一步 token 真正关键」这个根难题(那需要 process reward 或 value model)。

## 逆流而上:VAPO 把 value model 训回来

GRPO/DAPO 都在**逃离** value model,VAPO 反过来主张:**如果能把 value model 训准,它的上限更高**——因为它能给每个 token 估计「从这里往后还值多少」,提供比组内相对奖励更细的 credit assignment。

这在 long-CoT 尤其重要:一道长推理题,一个局部小错就让整条崩,只有句末奖励时模型不知道「哪一步开始偏」。VAPO 针对 long-CoT 的三个难点(长序列 value 偏差、序列长度异构、verifier 奖励稀疏)各配一套补丁(value 预训练、解耦 GAE、长度自适应 GAE、正样本 LM loss 等),在 AIME 上超过了 DAPO。它的意义:**value-based 路线没被淘汰,只要训得稳,仍可能比纯组内相对奖励更强。**

## 极端情形:TTRL 没有标注也能 RL

TTRL(Test-Time RL)反转常规顺序:**没有标准答案时,让模型自己造监督信号**。对一个无标注问题采样 N 条(如 64 条、温度 1.0),用**多数投票**得到伪标签,和伪标签一致的回答给 reward=1、否则 0,再用 GRPO/PPO 更新。

它为什么不荒唐:RL 本就对奖励噪声有容忍度,只要「多数投票比随机更常给对方向」就有用。但它有明确失败条件——模型太弱时,采样共识可能是「集体自信地犯错」,自举会变成「自己把自己带偏」。所以 TTRL 是**强模型的自举工具**,不是弱模型的救命药。

## 一张地图看全景

按「动了哪个零件」把这片版图组织起来:

| 维度 | 方法 | 相对经典 RLHF 的改动 |
|---|---|---|
| **完整范式** | RLHF (SFT→RM→PPO) | 基线:四模型协同 |
| 换打分者 | RLAIF | 人工偏好 → AI judge |
| 去 critic(换 baseline) | ReMax / RLOO / GRPO | value model → 贪心/组内 baseline |
| 稳定经典策略梯度 | REINFORCE++ | 裸 REINFORCE + 一套稳定器 |
| 改 GRPO 细节 | Dr.GRPO/DAPO/GSPO/CISPO | 见 [附录 05](05-grpo-variants.md) |
| 逆流:训回 value | VAPO | 重新拥抱 value-based,专治 long-CoT |
| 换奖励来源 | TTRL | 有标注 → 多数投票伪标签 |
| 绕开 RL | DPO 家族 | 见 [附录 08](08-dpo-variants.md) |

读这张图:中心是经典 RLHF 的四模型范式,四周所有方法都在拆/换其中某个零件——RLAIF 换奖励标注者,REINFORCE 系去 critic,VAPO 反而强化 critic,TTRL 换奖励来源,DPO 干脆绕开在线 RL。**它们不是互相取代,而是针对不同约束(标注贵/算力紧/序列长/无标注)各自的解。**

## 落回 MiniMind

MiniMind 在这张地图上的位置:**它实现了 PPO/GRPO/SPO 三种在线 RL(主线 ch07),用一个现成的外部 Reward Model + 规则奖励打分(不自己训 RM),保留 KL + reference 的内核**。

放进全景看,它有意选了「轻」的那一支:

- 不从零训 Reward Model(省下偏好数据采集 + RM 训练),直接加载现成的 `internlm2-reward` + 叠加规则 format 奖励;
- GRPO/SPO 都去 critic(和 ReMax/RLOO 同思路),SPO 用自适应 baseline tracker;
- 没上 VAPO 的重 value 路线、也没上 TTRL 的无标注自举,这些是特定场景(long-CoT、无标注)才划算的。

所以理解 MiniMind 的 RL,某种程度上是理解「在算力和标注都受限时,RL 对齐该怎么做减法」:不自己训 RM(借现成的)、去 critic、加规则奖励、留住 KL。这张全景图的价值,是让你看清它**减掉的每一样**在完整版图里原本是什么、为什么在小模型教学场景可以减。

## 常见误区

- **「RLHF 就是 PPO」**——不。RLHF 是 `SFT→RM→PPO` 三步范式,PPO 只是最后一步的策略优化算法,可以换成 GRPO、RLOO 等。
- **「去 critic 的方法(GRPO/RLOO)更先进」**——不是先进,是取舍。去 critic 省资源,但丢了细粒度 credit assignment;VAPO 反向证明训好 value 上限更高。
- **「RLAIF 不需要人了」**——不。它把人从重复标注抽出来去定标准/抽检,judge 仍需人工校准,否则偏差会被放大。
- **「Reward Model 给每个 token 打分」**——不。RM 只取最后一个有效 token 位置输出一个**句级**标量,对应人类标注的「整条回答更好还是更差」。
- **「TTRL 靠多数投票一定学对」**——不。它赌的是「多数投票比随机更常给对方向」,模型太弱时会集体犯错、越学越偏。

## 练习

1. 经典 RLHF 的三步是什么?各解决什么问题?MiniMind 跳过了哪一步、用什么替代?
2. Reward Model 为什么只取最后一个有效 token 的输出?它的训练损失对应什么偏好模型?
3. ReMax、RLOO、GRPO 都想去掉 critic,它们的 baseline 分别怎么定?
4. VAPO 和 GRPO/DAPO 在「要不要 value model」上立场相反,VAPO 的理由是什么?它针对什么场景?
5. TTRL 在没有标注时怎么造训练信号?它的失败条件是什么、为什么说它是「强模型的自举工具」?

<details>
<summary>参考答案</summary>

1. 三步:SFT(教会回答)→ 训练 Reward Model(教会偏好,从 pairwise 标注学句级奖励)→ PPO(教会取舍,在 KL 约束下把 policy 往高奖励推)。MiniMind 简化了第二步——不自己从零训 RM,而是加载现成的外部 RM(`internlm2-reward`)+ 规则 format 奖励打分。
2. 因为人类标注的是「整条回答更好还是更差」这个句级偏好,不是逐 token 真值;取最后一个有效 token(而非张量最后一列,因为右侧有 pad)的输出当整句代表。训练损失 `-logsigmoid(chosen−rejected)` 对应 Bradley-Terry 偏好模型,只要求同 prompt 下 chosen 分高于 rejected。
3. ReMax 用同 prompt 的 greedy 回答得分当 baseline(`r(sample)−r(greedy)`);RLOO 用同组其他回答的 leave-one-out 均值(`r_i − mean(其余)`);GRPO 用同组均值再除以标准差(组内标准化)。
4. VAPO 认为若能把 value model 训准,它能给每个 token 估计未来回报、提供更细的 credit assignment,上限比纯组内相对奖励更高。它针对 long-CoT——长推理里一个局部错就让整条崩,只有句末奖励时定位不到「哪步偏了」,细粒度 value 正好补这个。
5. 对无标注问题采样多条(如 64 条),用多数投票得伪标签,与伪标签一致给 reward=1、否则 0,再做 RL。失败条件:模型太弱时采样共识可能是集体犯错,自举变成自己把自己带偏。说它是强模型工具,因为只有模型已有足够先验、采样里包含足够多正确路径时,多数投票才能形成有效伪标签。
</details>
