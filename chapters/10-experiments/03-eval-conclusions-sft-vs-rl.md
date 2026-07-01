# 评测结论：SFT vs RL

[02 节](02-server-training-records.md) 的训练曲线只是过程证据。这一节是真正的能力评测——对各阶段权重做 [固定 prompt eval](01-fixed-prompt-design.md)，看输出行为到底变了什么。所有观察来自 8 层 768 同参（`--max_new_tokens 512`、temp 0.85、top_p 0.95、8 条内置 prompt）的真实日志。

参与对比的四个权重 `full_sft / dpo / ppo_actor / grpo` **均为 8 层 768，解码参数完全相同**——这是公平对比的前提，任何差异都来自训练阶段而非配置。

## pretrain vs full_sft：SFT 改的是形式

同参对比 pretrain 和 full_sft（2026-06-17 日志），行为级观察：

1. **开头**：pretrain 多以「？/。」续写（把 prompt 当半句话续）；full_sft 直接正面应答。
2. **收尾**：pretrain 会自造下一轮假对话（如聊完美食接「你能帮我查一下明天的天气吗？」）；full_sft 干净 EOS 收束。
3. **形式**：full_sft 走 chat template，输出 markdown 结构、代码块换行；pretrain 结构更乱。
4. **速度**：两者稳态都 ~112–116 tokens/s，无实质差异。

这正好印证 [05-sft](../05-sft/01-assistant-only-supervision.md)：SFT 只监督 assistant 区域、连 EOS 也监督，所以学会了「正面回答 + 适时收尾」。但**边界**很关键：**SFT 改善的是形式 / 指令遵循，不是事实正确性**——两者「天空为什么蓝」「斐波那契」都有错。

## full_sft vs DPO / PPO / GRPO：RL 让输出更长，但没更对

以 full_sft 为基准，三个 RL/对齐权重的行为级观察（2026-06-19 日志）：

1. **形式**：DPO ≈ full_sft；PPO 更结构化（`###` 分节、给多方案）；GRPO 最繁复（`##` / `###` 多层标题），最长，偶尔被 512 截断。
2. **长度**：RL（尤其 PPO/GRPO）明显比 full_sft 长。
3. **事实/代码正确性**：未改善。fibonacci 全部错或有 bug（full_sft `return a+b`；DPO 解释自相矛盾；PPO 幻觉「输出应为 4426180958」；GRPO f-string `{n}` 未定义）。
4. **新增自信错误**：PPO/GRPO 讲机器学习都写「无监督学习 (Simple Vector Machine)」——SVM 实为监督学习，这是 full_sft 里没有的错。
5. **速度**：四者都 ~112–117 tokens/s，无差异。

## 为什么 RL 没让它变对：reward hacking

结论：**RL 把输出推向更长、更结构化（迎合 reward model 偏好，这部分正是 reward hacking），但事实 / 代码正确性没提升，甚至新增了 full_sft 里没有的自信错误。**

把它和前面的机制接起来：[07-ppo-grpo/01](../07-ppo-grpo/01-rl-overview.md) 讲过 RL 只追 reward 会钻 reward model 的空子——reward model 偏好长而结构化的回答，policy 就往这个方向漂，于是输出更长、分节更多，但事实/代码没人给正确性打分，自然不会变对。叠加 0.1B 量级小模型的奖励稀疏限制，RL 不会让它在事实/代码上变对。**这是机制内在，不是训练参数调错。**

## Dense vs MoE：激活参数少，不等于推理快

前面都是 Dense（63.91M 全参激活）。把同一套五阶段在 MoE 上重跑一遍（`--use_moe 1`，198M 总参、top-1 激活 64M），做同参 A/B（2026-07-01 日志），得到一个反直觉的结果。

先看速度。MoE 的激活参数（64M）和 Dense（64M）几乎相等，理论 FLOPs 相近，但实测 decode 速度差一截：

| 阶段 | Dense（tokens/s） | MoE（tokens/s） |
|------|------------------|-----------------|
| full_sft | ~112 | ~74 |
| DPO | ~116 | ~76 |
| PPO | ~117 | ~75 |
| GRPO | ~113 | ~79 |

MoE 稳态慢约 35%。原因是 top-1 路由的 gather/scatter 和专家分派带来不连续访存，在 0.1B 这种小模型上，这部分开销盖过了稀疏激活省下的算力。[02-model/06-moe](../02-model/06-moe.md) 讲过 MoE 用「激活一部分专家」换算力，但那省的是 FLOPs，不是墙钟时间——路由本身有代价，小模型上代价还更显眼。**「激活参数少」和「推理快」是两件事。**

再看正确性。MoE 各阶段和 Dense 一样，事实硬伤照旧：光合作用全部错挂、天空为什么蓝大多讲反（MoE full_sft 甚至自相矛盾，先说蓝再说「呈现红色」）。RL/DPO 依然只改表达繁复度，不改知识。个别题目上 MoE 有零星变好（MoE DPO 把瑞利散射讲对了、MoE PPO/GRPO 的斐波那契基本正确），但同一份输出里光合作用仍全错，所以只是单题波动，不是能力整体抬升。

结论：在 0.1B 这个规模，Dense 和 MoE 的正确性没有实质差异（都被知识容量卡住），MoE 反而更慢。「MoE 更强」在训练 loss / reward 层面看不出（[02 节](02-server-training-records.md)），在能力 eval 层面也没兑现。这不是训练出了问题——MoE 的收益要在更大规模、更多专家时才显现，0.1B 加 4 专家只够验证「路由能跑通」，不够拉开能力差距。这是个诚实的中性结果，正好说明**不能用 aux_loss 非零、或总参数量大，就断言 MoE 更好**。

## 结论的边界（必须挂在前面）

这套观察先天有限，写进任何材料时都要带上：

1. **单次随机采样**（`do_sample=True`），非确定结论。
2. **训练侧 reward 可能上升，但与「能力提升」不等价**——[02 节](02-server-training-records.md) 是训练曲线，本节才是能力/行为评测，谈效果应以本节为准。
3. **速度是单卡单条推理**，未测 batch 吞吐——MoE 在大 batch 下访存摊薄后，和 Dense 的差距可能缩小，本节的「慢 35%」只对单条解码成立。

这条纪律——**训练曲线（reward 升没升）和能力评测（答得对不对）严格分开**——是这一章、也是看待所有 RLHF 结果的核心。

## 练习

1. SFT 相比 pretrain，在「开头/收尾/形式」上改善了什么？它改善事实正确性了吗？
2. 8 层同参 eval 下，RL（PPO/GRPO）相比 full_sft 输出有什么变化？事实/代码正确性呢？
3. 为什么说「RL 输出更长更结构化但没更对」是机制内在、不是参数调错？
4. 为什么训练曲线 reward 上升不能当作能力提升的证据？
5. MoE 的激活参数和 Dense 几乎相等，为什么实测反而慢约 35%？这对「MoE 省算力」的理解有什么修正？

<details>
<summary>参考答案</summary>

1. pretrain 把 prompt 当半句续写、会自造假对话；SFT 直接正面应答、干净 EOS 收束、走 markdown 格式。但没改善事实正确性——「天空为什么蓝」「斐波那契」两者都错。
2. RL（尤其 PPO/GRPO）输出明显更长、更结构化（多层标题、分节、多方案），GRPO 偶被 512 截断；事实/代码正确性未改善，PPO/GRPO 还新增「SVM=无监督学习」这类自信错。
3. reward model 偏好长而结构化的回答，policy 追 reward 就往这方向漂（reward hacking），而事实/代码没有正确性奖励信号；叠加 0.1B 小模型奖励稀疏，机制上就不会变对。
4. reward 上升只说明 policy 更迎合 reward model 的偏好，而 reward model 的偏好（长、结构化）不等于正确；要谈能力须做固定 prompt 行为评测。
5. 稀疏激活省的是 FLOPs（算力），不是墙钟时间。top-1 路由的 gather/scatter 和专家分派带来不连续访存，这部分开销在 0.1B 小模型上盖过省下的算力，于是更慢。修正：「激活参数少」不等于「推理快」，路由本身有代价，MoE 的速度收益要在更大规模、配合 batch 推理时才体现。
</details>
