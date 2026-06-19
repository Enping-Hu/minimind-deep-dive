# Tokenizer：文本和 token id 之间的翻译

模型不直接处理文字，它处理整数。tokenizer 负责两个方向的翻译：把一段文本切成 token、映射成 token id 序列（encode），再把 id 序列还原成文本（decode）。词表（vocabulary）就是 token 和 id 的对照表。

MiniMind 自带训练好的 tokenizer，放在 `model/tokenizer.json` 和 `model/tokenizer_config.json`。`trainer/train_tokenizer.py` 是它的训练示例——源码开头就提醒：不建议重训，因为换了词表，已有权重、数据格式、推理接口全部不兼容。但读一遍训练代码，是理解 tokenizer 的最好方式。

## 6400 的词表，为什么这么小

主流模型的词表都在几万到十几万：

| Tokenizer | 词表大小 |
|---|---|
| Qwen2 | 151,643 |
| Llama 3 | 128,000 |
| Mistral | 32,000 |
| MiniMind | 6,400 |

词表大小直接决定两个层的参数量：embedding 层 `vocab_size × hidden_size`，输出层（lm_head）同样大小。对 MiniMind 这种 `hidden_size=512` 的小模型，6400 的词表对应 `6400 × 512 ≈ 3.3M` 参数；如果换成 Qwen2 的 15 万词表，光这一层就 7000 万以上，会把一个 0.1B 模型的参数预算吃掉一大半。所以小模型选小词表是合理取舍：牺牲一点编解码效率（中文分得更碎），换回参数预算。代价是 PPL 这类按 token 统计的指标在不同 tokenizer 间不可直接比，跨 tokenizer 比较时 BPB（Bits Per Byte）更可靠。

## train_tokenizer.py：训一个 BPE 分词器

核心是 `train_tokenizer()`，用 HuggingFace `tokenizers` 库训练一个 BPE 模型：

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(
    vocab_size=6400,
    special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)
tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.decoder = decoders.ByteLevel()
```

几个要点：

- **BPE（Byte-Pair Encoding）**：从单字节开始，反复把最高频的相邻 token 对合并成一个新 token，直到词表达到 `vocab_size`。高频词组会合并成单个 token，低频内容退化到字节级别，所以任何输入都能编码、不会出现 unk 失败。
- **ByteLevel**：在字节层面切分，配合 `initial_alphabet` 覆盖全部 256 个字节，保证 UTF-8 的中文等多字节字符也能无损还原。
- **训练语料**：`get_texts()` 从 `pretrain_hq.jsonl` 逐行读 `text` 字段（脚本里只取前 10000 行做实验）。tokenizer 的训练只需要纯文本，和模型训练是两回事。

## 三个特殊 token

训练后断言它们的 id 固定：

```python
assert tokenizer.token_to_id("<|endoftext|>") == 0
assert tokenizer.token_to_id("<|im_start|>") == 1
assert tokenizer.token_to_id("<|im_end|>")   == 2
```

写进 `tokenizer_config.json` 后，它们的角色是：

| token | id | 角色 |
|---|---|---|
| `<|endoftext|>` | 0 | `pad_token` / `unk_token` |
| `<|im_start|>` | 1 | `bos_token`（一段消息的开始） |
| `<|im_end|>` | 2 | `eos_token`（一段消息的结束） |

这套 `<|im_start|>` / `<|im_end|>` 是 ChatML 风格的对话标记。注意 `add_bos_token` 和 `add_eos_token` 都是 `False`——是否加起止符由数据构造代码显式控制（见 [03-data-format](03-data-format.md)），不由 tokenizer 自动加。

## chat_template：对话拼成一个字符串

多轮对话在喂给模型前，要先按固定模板拼成一个字符串。模板写在 `tokenizer_config.json` 的 `chat_template` 里，通过 `tokenizer.apply_chat_template(messages)` 调用。它把每条消息包成 `<|im_start|>{role}\n{content}<|im_end|>\n`。`train_tokenizer.py` 的 `eval_tokenizer()` 给了演示，三条消息拼出来是：

```text
<|im_start|>system
你是一个优秀的聊天机器人，总是给我正确的回应！<|im_end|>
<|im_start|>user
你来自哪里？<|im_end|>
<|im_start|>assistant
我来自地球<|im_end|>
```

模板还支持 tool call、以及生成时追加 `<|im_start|>assistant\n` 作为续写起点（`add_generation_prompt=True`）。这些细节在 SFT 和 RL 章节用到时再展开；这里只需记住：**对话先经 chat_template 变成带 `<|im_start|>`/`<|im_end|>` 标记的纯文本，再交给 tokenizer 编码成 id。**

## 练习

1. MiniMind 为什么用 6400 这么小的词表？对小模型它省在哪里，代价是什么？
2. `<|im_start|>` 和 `<|im_end|>` 的 id 分别是多少，各自扮演 tokenizer 配置里的什么角色？
3. BPE 为什么不会遇到「这个字不在词表里」的 unk 问题？

<details>
<summary>参考答案</summary>

1. 小词表让 embedding 层和 lm_head（各 `vocab_size × hidden_size`）的参数大幅缩小，对 0.1B 量级模型省下可观参数预算；代价是编解码效率较低（中文切得更碎），且跨 tokenizer 比 PPL 不公平，需改用 BPB。
2. `<|im_start|>` id=1，是 `bos_token`（消息开始）；`<|im_end|>` id=2，是 `eos_token`（消息结束）。
3. BPE 基于 ByteLevel，最坏情况退化到单字节，而 `initial_alphabet` 覆盖全部 256 个字节，任何 UTF-8 文本都能用字节序列表示，所以不会出现无法编码的 token。
</details>
