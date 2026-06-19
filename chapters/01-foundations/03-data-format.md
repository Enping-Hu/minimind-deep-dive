# 数据格式：三类训练数据，以及它们怎么变成 input_ids / labels

MiniMind 所有训练数据统一用 jsonl（每行一个 JSON）。每个训练阶段在 `dataset/lm_dataset.py` 里对应一个 `Dataset` 类，职责相同：读一行 jsonl，吐出模型要的张量 `input_ids` 和监督目标 `labels`（或 mask）。本节看三类数据长什么样、以及对应的 `__getitem__` 怎么把它们变成张量。哪些 token 进 loss 的细节留到各训练章节，这里先建立「格式 → 张量」的整体印象。

数据样本格式取自 MiniMind README；文件可从 [minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)（ModelScope / HuggingFace）下载。

## Pretrain：纯文本

预训练数据（`pretrain_hq.jsonl`）每行只有一个 `text` 字段：

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

`PretrainDataset.__getitem__`（L41）把它变成张量：

```python
tokens = self.tokenizer(str(sample['text']), add_special_tokens=False,
                        max_length=self.max_length - 2, truncation=True).input_ids
tokens = [bos_token_id] + tokens + [eos_token_id]          # 手动加起止符
input_ids = tokens + [pad_token_id] * (self.max_length - len(tokens))  # 补齐到定长
input_ids = torch.tensor(input_ids, dtype=torch.long)
labels = input_ids.clone()
labels[input_ids == pad_token_id] = -100                   # pad 不算 loss
return input_ids, labels
```

几个细节：

- `add_special_tokens=False` + 手动 `[bos] + tokens + [eos]`：起止符由代码显式控制（呼应 [01-tokenizer](01-tokenizer.md) 里 `add_bos_token=False` 的配置）。
- `labels` 直接是 `input_ids` 的拷贝，只把 pad 位置设成 `-100`。也就是说，**除了 pad，每个 token 都是监督目标**。
- 这里没有做「预测下一个 token」的错位（shift）——那一步在模型前向里完成（`shift_logits` / `shift_labels`，见 [03-pretrain/02-forward-to-loss](../03-pretrain/02-forward-to-loss.md)）。`-100` 是 `F.cross_entropy` 的 `ignore_index`，被忽略的位置不产生梯度。

## SFT：多轮对话

SFT 数据（`sft_mini_512.jsonl`）每行是一个 `conversations` 数组，记录多轮对话，也可能带 Tool Call：

```jsonl
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
```

`SFTDataset.__getitem__`（L92）先用 `chat_template` 把对话拼成字符串，再 tokenize，最后构造标签：

```python
conversations = pre_processing_chat(sample['conversations'])  # 20% 概率随机加 system
prompt = self.create_chat_prompt(conversations)              # apply_chat_template
prompt = post_processing_chat(prompt)                        # 处理空 think 段
input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
input_ids += [pad_token_id] * (self.max_length - len(input_ids))
labels = self.generate_labels(input_ids)
```

和 Pretrain 的关键区别在 `labels`：`generate_labels` 只把 **assistant 回复**的 token 设为监督目标，user/system 部分全是 `-100`。这就是「只监督 assistant 回复」，第 5 章细讲。`pre_processing_chat` 会以 20% 概率随机插入一个 system prompt，让模型见过有无 system 两种情况。

## DPO：偏好对

DPO 数据（`dpo.jsonl`）每行有 `chosen` 和 `rejected` 两个对话，结构和 SFT 的 `conversations` 一样，只是同一个问题配了「更好」和「更差」两个回答：

```json
{
  "chosen":   [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "good answer"}],
  "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "bad answer"}]
}
```

`DPODataset.__getitem__`（L121）对 chosen / rejected 各跑一遍 chat_template + tokenize，再用 `generate_loss_mask` 标出 assistant token，返回错位好的 `x`（输入）、`y`（目标）和 `mask`。怎么用这对偏好算 loss，见第 6 章。

## RL 数据

PPO / GRPO 等在线 RL 用 `RLAIFDataset`，数据格式和 SFT 一致，但用法不同：它只取到最后一个 assistant **之前**的内容当 prompt（`add_generation_prompt=True`，结尾是 `<|im_start|>assistant\n`），把答案留空，交给模型在线生成（rollout）后再打分：

```python
return {'prompt': prompt, 'answer': answer}
```

也就是说，SFT/DPO 喂的是「完整对话」，RL 喂的是「半截 prompt + 待生成」。第 7 章展开。

## 小结

| 阶段 | 文件 | 关键字段 | Dataset 类 | 监督范围 |
|---|---|---|---|---|
| Pretrain | `pretrain_hq.jsonl` | `text` | `PretrainDataset` | 除 pad 外全部 token |
| SFT | `sft_mini_512.jsonl` | `conversations` | `SFTDataset` | 仅 assistant 回复 |
| DPO | `dpo.jsonl` | `chosen` / `rejected` | `DPODataset` | assistant 回复（chosen 与 rejected 对比） |
| RL | 同 SFT 格式 | `conversations` | `RLAIFDataset` | prompt 留空，在线生成后打分 |

## 练习

1. Pretrain 的 `labels` 是怎么从 `input_ids` 得到的？哪些位置不算 loss？预测下一个 token 的错位在哪一步做？
2. SFT 和 Pretrain 的数据处理，最关键的区别是什么？
3. RL 数据和 SFT 数据格式相同，但喂给模型的内容有什么不同？

<details>
<summary>参考答案</summary>

1. `labels = input_ids.clone()`，再把 pad 位置设为 `-100`；pad 不算 loss，其余 token 都是监督目标。预测下一个 token 的 shift 在 `MiniMindForCausalLM.forward` 里做（`shift_logits`/`shift_labels`），不在 dataset。
2. SFT 用 `generate_labels` 只把 assistant 回复的 token 设为监督目标，user/system 全设 `-100`；Pretrain 则监督除 pad 外的全部 token。
3. SFT 喂完整对话（含 assistant 回复）；RL 只喂到 assistant 之前的 prompt（`add_generation_prompt=True`），回复留给模型在线生成后再打分。
</details>
