# 数据与标签：PretrainDataset 怎么造 input_ids 和 labels

[01-foundations/03-data-format](../01-foundations/03-data-format.md) 看过 pretrain 数据是 `{"text": ...}`。这一节聚焦预训练这一阶段：`PretrainDataset` 怎么把一段文本变成 `(input_ids, labels)`，以及它和模型 loss 之间靠 `-100` 形成的闭环。

源码：`dataset/lm_dataset.py`，`PretrainDataset`。默认 `max_length=512`。

## __getitem__ 六步

```python
def __getitem__(self, index):
    sample = self.samples[index]
    tokens = self.tokenizer(str(sample['text']), add_special_tokens=False,
                            max_length=self.max_length - 2, truncation=True).input_ids
    tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
    input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == self.tokenizer.pad_token_id] = -100
    return input_ids, labels
```

1. 读出 `sample['text']`。
2. tokenizer 编码，`add_special_tokens=False`——不让 tokenizer 自动加特殊符，由代码显式控制；`max_length-2` 给手动加的 BOS/EOS 留两个位置；过长截断。
3. 手动拼 `[BOS] + tokens + [EOS]`。BOS 标记序列开始、EOS 标记结束，让模型学会从起点生成、也学会何时停。
4. 用 `pad_token_id` 补齐到 `max_length`，使 batch 内形状对齐。
5. 转成 `LongTensor`。
6. `labels = input_ids.clone()`，再把 pad 位置设为 `-100`。

## 为什么 labels 几乎就是 input_ids 的拷贝

预训练的任务是 next-token prediction——用前文预测下一个 token，正确答案本就藏在同一条序列里，不需要额外标注。所以 `labels` 直接复制 `input_ids` 即可。

关键是：**这里没有做「错位」**。你可能预期 `labels` 要相对 `input_ids` 右移一位，但平移是在模型前向里做的（`shift_logits` / `shift_labels`，见 [02-forward-to-loss](02-forward-to-loss.md)）。dataset 只提供对齐的 `input_ids` 和 `labels`，把错位留给模型，逻辑更集中。

## -100：dataset 和 loss 的闭环

```python
labels[input_ids == self.tokenizer.pad_token_id] = -100
```

pad 是为了凑长度补的无意义 token，不该参与训练。把这些位置的 label 设成 `-100`，正好对接模型里的 `F.cross_entropy(..., ignore_index=-100)`：

- dataset 负责把「不该监督的位置」标成 `-100`；
- model 的 loss 负责忽略这些位置。

这是数据侧和损失侧的配合。预训练阶段只屏蔽 pad——因为目标是让模型学会预测每个有效正文 token，正文里没有「只监督某一部分」的需求。这一点和 SFT 不同：SFT 只监督 assistant 回复（见 [05-sft](../05-sft/01-assistant-only-supervision.md)），用的是同一套 `-100` 机制，只是屏蔽范围更大。

<details>
<summary>源码细节：clone 为什么必要、布尔索引赋值</summary>

两个一字之差就会出 bug 的点（贴真实片段+函数名锚点，无行号，以片段为准）。

```python
labels = input_ids.clone()
labels[input_ids == self.tokenizer.pad_token_id] = -100
```

**1. `clone()` 不能省。** 如果写成 `labels = input_ids`（不 clone），两个变量指向**同一张底层张量**，下一行 `labels[...] = -100` 会同时把 `input_ids` 的 pad 位置也改成 -100——输入序列被污染，模型拿到的 `input_ids` 里 pad 变成了 -100 这个非法 token id。`clone()` 复制出独立张量，对 `labels` 的修改不影响 `input_ids`。

**2. `input_ids == pad_token_id` 是布尔掩码索引。** `input_ids == pad_token_id` 生成一个和 `input_ids` 等形的布尔张量（pad 位 True、其余 False），`labels[布尔张量] = -100` 只对 True 的位置赋值。这是 PyTorch 的 advanced indexing，等价于「逐元素：是 pad 就设 -100」，但向量化、无 Python 循环。注意它筛的依据是 `input_ids`（原值）而非 `labels`——此刻 `labels` 还是 `input_ids` 的拷贝，两者 pad 位置相同，用哪个都行，但用 `input_ids` 语义更清楚（「按输入里哪些是 pad」）。

</details>

## 练习

1. 为什么 tokenizer 编码时设 `add_special_tokens=False`、`max_length` 还要 `-2`？
2. 预训练的 `labels` 为什么可以直接 `clone` 自 `input_ids`？「预测下一个 token」的错位在哪里完成？
3. `-100` 在 dataset 里写入，在哪里被使用？两者怎么配合？
4. 为什么预训练只把 pad 标成 `-100`，而不像 SFT 屏蔽更多位置？
5.（源码细节）`labels = input_ids.clone()` 的 `clone()` 能省吗？省了会怎样？

<details>
<summary>参考答案</summary>

1. 让 dataset 显式控制特殊符，避免 tokenizer 自动加 BOS/EOS 造成重复；`-2` 是给手动拼接的一个 BOS 和一个 EOS 预留位置。
2. 因为监督信号来自同一条序列的 next-token prediction，正确答案就在序列里；真正的时序错位由模型的 `shift_logits`/`shift_labels` 完成，不在 dataset 做。
3. dataset 把 pad 位置 label 设为 `-100`；模型的 `F.cross_entropy(ignore_index=-100)` 忽略这些位置。dataset 标记、loss 忽略，形成闭环。
4. 预训练要学每个有效正文 token 的预测关系，没有「只监督一部分」的需求；只有 pad 这种无意义位置需屏蔽。SFT 则要把 user/system 也屏蔽，只留 assistant。
5. 不能省。`labels = input_ids`（不 clone）会让两者指向同一张量，`labels[...] = -100` 会同时污染 `input_ids` 的 pad 位置；`clone()` 复制独立张量，改 labels 不影响 input_ids。
</details>
