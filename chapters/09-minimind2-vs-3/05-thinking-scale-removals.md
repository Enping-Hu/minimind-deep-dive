# D/E/F：thinking 路线、规模数据、移除与新增

本节收尾三类更新：thinking 路线（D）、规模/数据/默认值（E）、移除与新增清单（F）。

## D. thinking 路线

v2 有独立的 `train_reason.py` 训练「思考」能力，tokenizer 只有简单的 `enable_thinking`。v3 把思考收进 chat_template，移除独立 reason 脚本。

- **chat_template 内建 think**（`model/tokenizer_config.json`）：v3 模板会解析 `reasoning_content`、注入 `<think>\n...\n</think>` 段，用模板变量 `open_thinking` 控制生成提示里是否放 `<think>`。对照 grep：v3 模板出现 `reasoning_content`（6 处）、`open_thinking`（2 处）、`<think>`（5 处）；v2 只有 `enable_thinking`（2 处）、`<think>`（1 处）。
- **eval_llm.py**：v3 加 `--open_thinking`，且对所有非 pretrain 权重统一走 chat_template；v2 的 `enable_thinking` 只对 `reason` 权重生效（[04-inference/02](../04-inference/02-eval-and-service.md) 看过 v2 的 `if args.weight == 'reason'`）。
- **移除 `train_reason.py`**：思考能力从「独立 reason 训练脚本」改为「chat_template + open_thinking + 数据里混 reasoning 样本」承接。
- **数据侧**：`RLAIFDataset` 增加 `thinking_ratio` 参数。

影响 [SFT](../05-sft/01-assistant-only-supervision.md) 的标签逻辑——assistant 区域里可能含 `<think>` 段，但 assistant-only 监督的框架不变。

## E. 规模 / 数据 / 默认值对照

| | v2 | v3 |
|---|---|---|
| Dense hidden_size | 512 | 768 |
| pretrain epochs | 1 | 2 |
| pretrain 数据 | `pretrain_hq.jsonl` | `pretrain_t2t(_mini).jsonl` |
| SFT lr / seq_len / 数据 | 1e-6 / 340 / `sft_mini_512` | 1e-5 / 768 / `sft_t2t(_mini)` |
| DPO beta | 0.1 | 0.15 |
| PPO lr（actor/critic） | 8e-8 / 8e-8 | 3e-7 / 5e-7 |
| PPO clip_epsilon | 0.1 | 0.2 |
| GRPO num_generations / beta | 8 / 0.02 | 6 / 0.1 |
| RL 数据 | `rlaif-mini.jsonl` | `rlaif.jsonl` |
| eval 推理精度 | fp32 | `model.half()` fp16 |

再强调一次（[01-overview](01-overview-five-changes.md) 的口径）：`hidden_size 512→768` 是**默认值差异，不是 v3 才有的能力**。两版都能配任意 `hidden_size`，本书 1–8 章行号以 v2 默认 512 为准。其余都是默认超参的调整，不是机制变化——比如 PPO lr 调大、clip_epsilon 放宽，对应的是 [03 节](03-ppo-rewrite.md) 重写后的 PPO2 需要不同的超参区间。

## F. 移除 / 新增清单（`trainer/`）

直接对照两版 `trainer/` 目录：

- **移除**：`train_spo.py`（SPO，[第 7 章 SPO](../07-ppo-grpo/04-spo.md) 为 v2-only）、`train_reason.py`（并入 thinking 路线）。
- **新增**：`train_agent.py` + `rollout_engine.py`（Agent RL，多轮 rollout + 延迟 reward）；`dataset/lm_dataset.py` 增加 `AgentRLDataset`。
- **两版都有**：pretrain / full_sft / dpo / ppo / grpo / lora / distillation / tokenizer。

SPO 在 v3 被移除，呼应 [第 7 章](../07-ppo-grpo/04-spo.md) 提到的「SPO 偏工程尝试、理论标准性不如经典 PPO」。Agent RL 是 v3 的新方向（多轮工具调用 / 延迟奖励），属于进阶内容，本书 [附录](../appendix/02-advanced-pointers.md) 点到为止。

## 至此第 9 章收束

v2→v3 的五类更新都落到了源码：A 模型结构（[02](02-architecture-diffs.md)）、B PPO 重写（[03](03-ppo-rewrite.md)）、C GRPO CISPO（[04](04-grpo-cispo.md)）、D/E/F 本节。配合 [01 节](01-overview-five-changes.md) 的防误记清单，对照看「同一个东西在两版怎么变」就完整了。

## 练习

1. v3 的 thinking 能力是怎么承接 v2 `train_reason.py` 的？
2. `hidden_size 512→768` 和 `PPO clip_epsilon 0.1→0.2` 这两类差异，性质上有什么不同？
3. v3 的 `trainer/` 相比 v2 移除了什么、新增了什么？

<details>
<summary>参考答案</summary>

1. v2 用独立 `train_reason.py`；v3 移除它，改为 chat_template 内建 think（`reasoning_content` / `<think>` / `open_thinking`）+ 数据里混 reasoning 样本（`RLAIFDataset` 的 `thinking_ratio`）承接。
2. `hidden_size 512→768` 是默认值差异（两版都能配任意值）；`clip_epsilon 0.1→0.2` 是配合 PPO 重写（PPO2）后的默认超参调整。两者都不是「只有 v3 能做」的能力差异。
3. 移除 `train_spo.py`、`train_reason.py`；新增 `train_agent.py` + `rollout_engine.py`（Agent RL）和 `AgentRLDataset`；pretrain/full_sft/dpo/ppo/grpo/lora/distillation/tokenizer 两版都有。
</details>
