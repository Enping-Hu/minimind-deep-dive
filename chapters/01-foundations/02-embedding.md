# Embedding：token id 变成向量，以及和输出层共享权重

tokenizer 把文本变成 token id 序列，但模型内部计算的是连续向量。embedding 层负责这一步：把每个整数 id 映射成一个 `hidden_size` 维的向量。这是模型的第一层，也是 `lm_head`（最后一层）的镜像——MiniMind 让这两层共享同一份权重。

源码都在 `model/model_minimind.py`。

## embed_tokens：一张可学习的查找表

`MiniMindModel.__init__`（L570）里：

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
```

`nn.Embedding` 本质是一个形状 `[vocab_size, hidden_size]` 的可学习矩阵，第 `i` 行就是 id 为 `i` 的 token 的向量。前向时按 id 查表（L595）：

```python
hidden_states = self.dropout(self.embed_tokens(input_ids))
```

形状变化：输入 `input_ids` 是 `[batch, seq_len]` 的整数张量，过 embedding 后变成 `[batch, seq_len, hidden_size]`，也就是每个位置一个 512 维向量。这些向量随训练更新，逐渐学到「语义相近的 token 向量也相近」。默认配置下这张表是 `6400 × 512`。

## lm_head：从向量回到词表

模型最后要预测下一个 token，得把 `hidden_size` 维向量映射回 `vocab_size` 维的分数（logits）。`MiniMindForCausalLM.__init__`（L637）：

```python
self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
```

它把 `[batch, seq_len, hidden_size]` 映射成 `[batch, seq_len, vocab_size]`，每个位置给出词表上每个 token 的分数。logits 怎么变成概率、怎么算 loss，见 [03-pretrain/02-forward-to-loss](../03-pretrain/02-forward-to-loss.md)。

## 权重共享（tie embeddings）

注意 embedding 是 `[vocab_size, hidden_size]`，lm_head 的权重是 `[vocab_size, hidden_size]`（`nn.Linear` 权重形状为 `[out, in]`）——两者形状完全一样。MiniMind 直接让它们共用一份（L639）：

```python
# 权重绑定：嵌入层和 LM 头共享权重
self.model.embed_tokens.weight = self.lm_head.weight
```

两个理由：

- **省参数**。两层各 `6400 × 512 ≈ 3.3M`，共享后只存一份。对 MiniMind2-Small（约 26M）这种小模型，省下的这 3M 占比可观。
- **输入输出空间一致**。「id → 向量」和「向量 → id 分数」用同一组向量，输入端和输出端对同一个 token 的表示是绑定的，通常还能略微提升效果。这是小模型里常见的做法。

> 版本提示：MiniMind2 这里是硬编码绑定；MiniMind-3 改成由 config 的 `tie_word_embeddings` 开关控制，行为一致（见第 9 章）。这是**两版都做**的事，不是 v3 才有。

## 练习

1. `input_ids` 形状 `[batch, seq_len]`，过 `embed_tokens` 后形状是什么？
2. embedding 层和 lm_head 的权重形状为什么能共享？共享带来哪两个好处？

<details>
<summary>参考答案</summary>

1. 变成 `[batch, seq_len, hidden_size]`，每个 token 位置对应一个 `hidden_size` 维向量。
2. embedding 是 `[vocab_size, hidden_size]`，`nn.Linear(hidden_size, vocab_size)` 的权重是 `[vocab_size, hidden_size]`，形状相同所以可共用一份。好处：省下一份 `vocab_size × hidden_size` 参数；输入端和输出端对同一 token 用同一组向量，表示一致。
</details>
