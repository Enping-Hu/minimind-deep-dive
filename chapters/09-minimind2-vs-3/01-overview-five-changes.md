# MiniMind2 → MiniMind-3：五类更新总览

前面 1–8 章讲的全是 MiniMind2（仓库 `minimind-master`，默认 `hidden_size=512`）。MiniMind-3（`minimind-3`）的 README 自述「主线结构对齐 Qwen3 / Qwen3-MoE」。这一章逐条对照两版源码，讲清 v2 到 v3 改了什么。本节先给总览和一份**防误记清单**。

事实来源：本地直接 diff `minimind-master`（v2）与 `minimind-3`（v3）的 `model/model_minimind.py`、`trainer/`、`tokenizer_config.json`，行号可复核。

## 五类更新

| 类别 | 一句话 | 对应本书章节 |
|---|---|---|
| A 模型结构 | +QK-Norm、−shared expert、head_dim 解耦 | [02-architecture-diffs](02-architecture-diffs.md) |
| B PPO 重写 | 五模型→四模型，简化优势→token-level GAE | [03-ppo-rewrite](03-ppo-rewrite.md) |
| C GRPO | 默认改用 CISPO 变体 | [04-grpo-cispo](04-grpo-cispo.md) |
| D thinking 路线 | chat_template 内建 think，移除独立 reason | [05-thinking-scale-removals](05-thinking-scale-removals.md) |
| E 规模/数据/默认值 | hidden 512→768，数据换 t2t/rlaif | [05-thinking-scale-removals](05-thinking-scale-removals.md) |
| F 移除/新增 | 移除 SPO；新增 Agent RL | [05-thinking-scale-removals](05-thinking-scale-removals.md) |

每条都落到源码，对应到前面讲过的 MiniMind2 章节，方便对照看「同一个东西在两版怎么变」。

## 防误记：这三项不是差异

最容易写错的是把「两版本来就一样的东西」当成 v3 的新特性。下面三项**两版完全一致**：

- **`rope_theta` 都是 1e6**。v2、v3 的 `MiniMindConfig` 都默认 `rope_theta=1e6`。RoPE 的基底没变。
- **`tie_word_embeddings` 两版都绑定** embedding 与 lm_head 权重。差别只在实现方式：v2 在 `MiniMindForCausalLM.__init__` 里硬编码 `self.model.embed_tokens.weight = self.lm_head.weight`，v3 走 config flag（`if self.config.tie_word_embeddings:`）——但结果都是绑定（[02-model/embedding](../01-foundations/02-embedding.md) 讲过 weight tying）。
- **MoE 的 router top-k 机制一致**：都是 `gate → softmax → topk → norm_topk_prob` 选 top-k 专家、训练时算 aux_loss。v3 改的只是「有没有 shared expert 那条支路」，不是 router 本身（见 [02-architecture-diffs](02-architecture-diffs.md)）。

写 v2/v3 对照时，先排除这三项，避免「v3 才有 RoPE / v3 才绑定权重」这类错误。

## 一个口径澄清：512 vs 768

E 类里 `hidden_size 512→768` 是**默认值差异，不是 v3 才有的能力**。两版都能配任意 `hidden_size`——v2 默认 512（MiniMind2-Small），v3 默认 768。本书 1–8 章的行号和参数量都以 v2 默认 512 为准（如 `head_dim=512/8=64`）；到了 v3，默认换成 768，`head_dim` 还能独立配置（A3）。别把「v3 默认更大」读成「只有 v3 能做大模型」。

## 受影响的 v2 章节速查

| v2 内容 | v3 差异 | 本章定位 |
|---|---|---|
| [Attention](../02-model/02-attention.md) | +QK-Norm、head_dim 解耦 | 02 节 |
| [MoE](../02-model/06-moe.md) | −shared expert | 02 节 |
| [PPO](../07-ppo-grpo/02-ppo.md) | 5→4 模型、GAE、clipped value、k3 KL | 03 节 |
| [GRPO](../07-ppo-grpo/03-grpo.md) | 默认 CISPO | 04 节 |
| [SPO](../07-ppo-grpo/04-spo.md) | v3 移除 | 05 节 |
| [SFT](../05-sft/01-assistant-only-supervision.md) | chat_template 内建 think | 05 节 |

## 练习

1. v2→v3 的五类更新分别是什么？
2. 哪三项「看似变了其实没变」？各自的细节是什么？
3. `hidden_size 512→768` 是能力差异还是默认值差异？为什么这个区分重要？

<details>
<summary>参考答案</summary>

1. A 模型结构（QK-Norm/−shared expert/head_dim 解耦）、B PPO 重写（5→4 模型 + token-level GAE）、C GRPO 默认 CISPO、D thinking 路线（chat_template 内建 think）、E 规模数据默认值、F 移除 SPO+新增 Agent RL。
2. `rope_theta`（两版都 1e6）、`tie_word_embeddings`（两版都绑定，v2 硬编码 / v3 config flag）、MoE router top-k 机制（一致，v3 只去掉 shared expert 支路）。
3. 是默认值差异——两版都能配任意 hidden_size，v2 默认 512、v3 默认 768。区分重要是因为不能把「v3 默认更大」误读成「只有 v3 能做大模型」，本书行号以 v2 的 512 为准。
</details>
