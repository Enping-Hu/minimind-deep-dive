# 服务器训练记录

这一节是 MiniMind-3（768 维、8 层 dense）在服务器上的真实训练记录：环境、数据、命令、loss、训练曲线。所有数字来自实际运行日志和 SwanLab，可复核。

## 环境与数据

```text
日期：2026-06-09
GPU：NVIDIA，CUDA driver 12.2
torch 2.5.1+cu121 / transformers 4.57.6 / datasets 3.6.0
训练精度：dtype=bfloat16
```

pretrain 数据 `pretrain_t2t.jsonl`，7,970,519 条样本，字段 `text`（第一行 JSON 异常，修复后替换）。

## Pretrain

```text
命令：trainer/train_pretrain.py，hidden_size=768、num_hidden_layers=8、batch_size=32、
     accumulation_steps=8、max_seq_len=380、epochs=2、learning_rate=5e-4、dtype=bfloat16
SwanLab run id：tcvebdyzbdj9xn2d1bs3e
进度：Epoch [2/2](249079/249079)
输出：out/pretrain_768.pth（约 132M）；resume checkpoint 约 619M、step=249079
```

训练曲线（run `tcvebdyzbdj9xn2d1bs3e`）：loss 从 ~7.3 快速降到 ~2，再缓慢收敛到 ~1.5–1.7；`logits_loss` 同步；`aux_loss=0`（dense，[02-model/06-moe](../02-model/06-moe.md) 讲过 dense 时 aux_loss 恒为 0）。`learning_rate` 和 `epoch_time` 可见两段，对应 resume 续训。

![Pretrain 训练曲线](../../images/swanlab/MiniMind-Pretrain-Epoch-2-BatchSize-32-LearningRate-0.0005.png)

## Full SFT

```text
命令：trainer/train_full_sft.py，Epoch=2、BatchSize=64、LearningRate=1e-5（起训权重 pretrain_768.pth）
SwanLab run id：5r4qfa5jtw3m8nayn9xie
最终 loss：~1.25–1.4（末值 ~1.3，从 ~1.75 缓慢下降）；aux_loss=0
输出：out/full_sft_768.pth
```

训练曲线（run `5r4qfa5jtw3m8nayn9xie`）：loss 从 ~1.75 噪声下降到 ~1.25–1.4；`epoch_time` 两段=2 epoch。

![Full SFT 训练曲线](../../images/swanlab/MiniMind-Full-SFT-Epoch-2-BatchSize-64-LearningRate-1e-05.png)

## DPO

```text
Epoch=1、BatchSize=4、LearningRate=4e-8
```

`dpo_loss` 在初始值 ~0.69（=−log 0.5，[06-dpo/02](../06-dpo/02-dpo-loss-and-math.md) 的 −logsigmoid 在 logits≈0 时正是这个值）附近震荡（0.4–0.85），末值 ~0.45；记录点很少（~42）、lr 极小（4e-8）。

**边界**：步数少 + lr 极小，loss 变化大部分是噪声，**不能据此说已学到稳定偏好**，需后续固定 prompt eval 验证。这正是 [05-optimizer](../08-training-mechanics/05-optimizer-adamw-scheduler.md) 提到的——对齐阶段 lr 故意调到极小，避免破坏 SFT 已有能力。

![DPO 训练曲线](../../images/swanlab/MiniMind-DPO-Epoch-1-BatchSize-4-LR-4e-08.png)

## PPO

```text
Epoch=1、BatchSize=2、LearningRate=3e-7
SwanLab run id：xtcfq7mqy5orjzy32l123
```

reward 噪声很大、**无清晰上升趋势**；`kl_ref` 随训练升到 ~0.1–0.25；`approx_kl` / `clipfrac` 很低（更新受控）；`critic_loss` ~0.2–0.5。

**边界**：约 10k step，reward 未见明显改善，符合 README 所述「PPO reward 提升缓慢」。

![PPO 训练曲线](../../images/swanlab/MiniMind-PPO-Epoch-1-BS-2-LR-3e-07.png)

## GRPO

```text
Epoch=1、BatchSize=4、LearningRate=3e-7、loss_type=cispo（8 层 768）
```

（约 5k step；以下为低分辨率截图读出的趋势，精确数值以 SwanLab 为准）reward 噪声大、**无明显单调上升**；`kl_ref` 从 ~0 缓慢下漂到 ~−0.15；`advantages_std` ~1（偶掉到 ~0.6）、`advantages_mean ≈ 0`（[03-grpo](../07-ppo-grpo/03-grpo.md) 的组内归一，均值本就接近 0）；`avg_response_len` 噪声大（约 100–600）。

**边界**：仍是训练奖励/统计、非能力评测。reward 不明显上升，需结合 [下一节](03-eval-conclusions-sft-vs-rl.md) 的 8 层 eval（输出反而更长更繁复）一起看。

![GRPO 训练曲线](../../images/swanlab/MiniMind-GRPO-Epoch-1-BS-4-LR-3e-07.png)

## 小结：训练曲线只是过程证据

- pretrain / SFT 的 loss 正常收敛；DPO 因步数少 + 极小 lr 几乎没动；PPO / GRPO 的 reward 都是平噪声、无明显上升。
- 这些只支撑「训练过程证据」。**要谈「能力 / 效果」，必须对权重做固定 prompt eval 对比**——这是下一节的内容。把训练曲线（reward 升没升）和能力评测（答得对不对）分开，是这一章最重要的纪律。

## 练习

1. 这一节的训练曲线（loss / reward）能用来说明「模型能力提升」吗？为什么？
2. DPO 的 `dpo_loss` 初始值为什么约 0.69？为什么说它的曲线「大部分是噪声」？
3. pretrain 曲线里 `learning_rate` 和 `epoch_time` 为什么呈两段？

<details>
<summary>参考答案</summary>

1. 不能。它们是训练过程证据（loss 收敛、reward 升降），不等于能力；要谈能力须做固定 prompt eval 对比。
2. `dpo_loss = −logsigmoid(β·logits)`，初始 policy≈reference 时 logits≈0，`−logsigmoid(0)=−log0.5≈0.69`；DPO 步数少（~42 点）+ lr 极小（4e-8），loss 变化大部分是噪声，不能据此说学到稳定偏好。
3. 因为 resume 续训——训练中断后从 checkpoint 恢复，lr 调度和计时各成一段，对应 2 个 epoch。
</details>
