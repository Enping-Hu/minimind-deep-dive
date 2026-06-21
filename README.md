# minimind-deep-dive

源码精读式的 MiniMind 学习笔记。面向想从每一行代码弄清楚一个最小可训练 LLM 怎么搭、怎么训练、怎么对齐的人。

这套笔记不复制 MiniMind 的源码，而是带你对照源码读：每一节都标出对应的源码文件与符号位置（函数/类名），均以 MiniMind2 主线为准。建议把它和源码并排打开。

- 原项目（源码）：[MiniMind](https://github.com/jingyaogong/minimind)
- 笔记中的符号引用（函数/类名）对应 MiniMind2 主线（仓库内称 `minimind-master`）；涉及版本差异处会标注 MiniMind-3，集中在第 9 章。

## 组织方式

按 **结构 → 训练 → 机制 → 版本 → 实验** 推进：先把模型拆开看清楚，再看它怎么从一条数据走到一次参数更新，然后深入贯穿各训练阶段的数学链，最后对照 MiniMind-3 的演进，并给出真实的服务器训练与评测证据。

几个别处少见、值得先知道的点：

- 一条贯穿的训练数学链：`logits → token log-prob → 序列 loss → backward → optimizer.step`，把 Pretrain / SFT / DPO / PPO / GRPO 的更新骨架统一起来（第 8 章）。
- MiniMind2 与 MiniMind-3 / Qwen3-style 的逐条源码对照：QK-Norm、移除 shared expert、PPO 重写、GRPO 默认 CISPO（第 9 章）。
- 真实服务器训练曲线 + 固定 prompt 评测结论，含 RL 的 reward-hacking 现象（训练 reward 与实际能力的背离）（第 10 章）。

## 学习路径

| Part | 章 | 内容 | 状态 |
|---|---|---|---|
| 结构 | [00-overview](chapters/00-overview/) | MiniMind 是什么、四层源码地图、环境与快速开始 | ✅ v1 |
| 结构 | [01-foundations](chapters/01-foundations/) | Tokenizer、Embedding、数据格式（从源码读起） | ✅ v1 |
| 结构 | [02-model](chapters/02-model/) | Block / RMSNorm / Attention / RoPE / GQA / SwiGLU / MoE（+归一化演进延伸） | ✅ v1 |
| 训练 | [03-pretrain](chapters/03-pretrain/) | 数据与标签、前向到 loss、Pretrain 主循环 | ✅ v1 |
| 训练 | [04-inference](chapters/04-inference/) | KV cache 与 generate（也是 RL 在线采样的前置）、推理服务、权重格式 | ✅ v1 |
| 训练 | [05-sft](chapters/05-sft/) | SFT：为什么只监督 assistant 回复 | ✅ v1 |
| 训练 | [06-dpo](chapters/06-dpo/) | DPO：偏好优化与 −logsigmoid 目标 | ✅ v1 |
| 训练 | [07-ppo-grpo](chapters/07-ppo-grpo/) | RL 总览、PPO、GRPO、SPO、训练信号总表（+GRPO 变体家族延伸） | ✅ v1 |
| 机制 | [08-training-mechanics](chapters/08-training-mechanics/) | 从 logits 到参数更新的完整训练机制 | ✅ v1 |
| 版本 | [09-minimind2-vs-3](chapters/09-minimind2-vs-3/) | MiniMind2 → MiniMind-3 / Qwen3-style 逐条对照 | ✅ v1 |
| 实验 | [10-experiments](chapters/10-experiments/) | 固定 prompt 实验设计、服务器训练记录、SFT vs RL 评测结论 | ✅ v1（实验持续补充） |
| 进阶 | [appendix](chapters/appendix/) | 进阶方向（Flash Attention / LoRA / 蒸馏 / Agent RL）点到为止 | ✅ v1 |

每章是一个目录，下面按 `NN-子主题.md` 编号；章内用 `##` 分小节。每章末尾有思考题，参考答案折叠在题目下方。

## 版本与收录原则

这是 **v1**：忠于 MiniMind 主线源码做一遍完整、准确的精读，符号位置与默认值都对照 MiniMind2（默认 `hidden_size=512`，即 MiniMind2-Small 约 26M）核对。有源码 / 实操支撑的写深，没有的诚实标注边界、点到为止（见 appendix）。

**收录原则**：一节进入正文主线，须同时满足 ① 锚在 MiniMind 真实代码 / 选择上，② 回答源码读不出的「为什么 / 怎么来」。只为对标其他资料、而 MiniMind 未实现的内容不进主线——例如 Flash Attention 的 V1/V2 自实现深入（MiniMind 仅调用 PyTorch 的 flash 路径）、面试题库（每章末思考题已承担自测），至多在「进阶」挂入口。

**进阶层**：超出 MiniMind 代码的广度内容统一归为「进阶」——目前是 appendix，加两节延伸 survey（[归一化演进](chapters/02-model/07-normalization-evolution.md)、[GRPO 变体家族](chapters/07-ppo-grpo/06-grpo-variants.md)），它们都锚回 MiniMind 的真实选择，按上述原则收录。

## 路线图

本仓库随自学进度持续更新。主线（源码精读）已完成，以下按「收录原则」规划，标注进度：

- **可运行对照实验**（计划中）：新建 `experiments/` 目录，放 CPU 可跑的小实验——如 RoPE 多频证「为什么要 d/2 组频率」、Pre/Post-Norm 证「为什么要归一化」，把第 10 章的证据从训练曲线扩展到读者可复现。
- **第 10 章实验补充**（更新中）：MiniMind-3 的服务器训练 / 评测仍在进行，后续补更多训练曲线与评测证据（含 GRPO 8 层 run id 回填）。
- **RoPE 配图**（待补）：ch02/03 的旋转示意图。
- **其余视需要**：Tokenizer BPE 训练细节、进一步 Part 重组。

不进主线（仅在进阶 / appendix 挂入口，见收录原则）：Flash Attention 的 V1/V2 深入、面试题库。

## 配图

正文流程图（SVG）来自源码精读时绘制的结构图，存于 [`images/`](images/)；训练曲线截图来自 SwanLab，存于 [`images/swanlab/`](images/swanlab/)。

## 前置知识

不需要精通，但先了解这些概念会读得更顺：交叉熵、反向传播与链式法则、self-attention、causal mask、RoPE、RMSNorm、SFT / 偏好数据 / PPO clip。

## 来源与致谢

- 源码：MiniMind / MiniMind-3，作者 jingyaogong。
- 数据格式与样本：引自 MiniMind-3 README；训练数据集见 [minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset)（ModelScope / HuggingFace）。
- 本仓库是个人学习笔记的重构整理，不隶属于原项目。

## 许可

本仓库教学内容采用 [CC BY 4.0](LICENSE)（署名）许可；其中引用的 MiniMind 源码片段版权归原项目 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 所有，遵循其 Apache-2.0 协议。
