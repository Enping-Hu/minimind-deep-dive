# Pretrain 主循环

`trainer/train_pretrain.py` 是整个项目最基础的训练入口。后面的 SFT、DPO、PPO/GRPO 目标函数各异，但这套工程骨架——解析参数、建模型和数据、跑训练循环、存 checkpoint——几乎一致。先吃透这份脚本，等于吃透项目的训练骨架。

这一节走它的整体结构，到「一步训练做了什么」为止。`loss → backward → optimizer.step` 这条更新链的数学细节集中在 [第 8 章](../08-training-mechanics/01-update-skeleton.md)；学习率调度、断点续训、混合精度、DDP 这些工程件在本节随主流程一并点到，不单独展开。

## 脚本分两块

- `train_epoch(...)`：单个 epoch 内的 step 级训练。
- `if __name__ == "__main__":`：训练前准备、状态恢复、组织 epoch 循环。

建议先看 `main` 搭起整体，再看 `train_epoch` 看每步更新。

## main：训练前怎么搭起来

按执行顺序：

1. 初始化分布式环境 + 随机种子（`init_distributed_mode`、`setup_seed(42)`）。
2. 构造 `MiniMindConfig`，按 `from_resume` 检查是否有 checkpoint。
3. 设混合精度上下文 `autocast_ctx` 和 `GradScaler`。
4. （可选）初始化 wandb / swanlab 日志。
5. 建五大对象：`model, tokenizer = init_model(...)`、`PretrainDataset`、`GradScaler`、`AdamW`。
6. 若有 checkpoint，恢复 model / optimizer / scaler / epoch / step——**恢复的是完整训练现场，不只是参数**（AdamW 的一二阶动量、scaler 比例都要恢复，否则训练动力学会变）。
7. 分布式时用 `DistributedDataParallel` 包模型。
8. 进入 epoch 循环，调用 `train_epoch`。

注意区分两种「加载」：`from_weight` 是从已有参数出发训练，`from_resume` 是恢复训练现场（参数 + 优化器 + scaler + 进度）。

## train_epoch：一步训练做什么

核心循环：

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    input_ids = input_ids.to(args.device)          # 1. 数据搬到 GPU
    labels = labels.to(args.device)
    lr = get_lr(epoch * iters + step, ...)          # 2. 按进度算学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr                      #    手动写进 optimizer

    with autocast_ctx:                              # 3. 混合精度前向
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss              #    主损失 + MoE 辅助损失
        loss = loss / args.accumulation_steps       #    梯度累积：按步数缩放

    scaler.scale(loss).backward()                   # 4. 反向，累积梯度

    if (step + 1) % args.accumulation_steps == 0:   # 5. 每 accumulation_steps 步更新一次
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    # 6. 定期日志、保存
```

八件事：取 batch → 搬 GPU → 算 lr → autocast 前向 → `loss/accumulation_steps` → backward 累积 → 满足条件才 step → 日志和保存。

注意「数据构造」和「数据搬运」是两件事：`PretrainDataset` 在 CPU 侧把文本变张量（[01-data-and-labels](01-data-and-labels.md)），训练循环每步再 `.to(device)` 搬上 GPU。学习率也不是优化器自动调的，是脚本每步用 `get_lr` 算好手动写进 `param_groups`（`get_lr` 的 cosine 调度见 [08-training-mechanics/05-optimizer](../08-training-mechanics/05-optimizer-adamw-scheduler.md)）。

## 关键默认值（核对自源码 argparse）

| 参数 | 默认 | 含义 |
|---|---|---|
| `epochs` | 1 | 训练轮数（zero 模型 1 轮，充分训练 2–6 轮） |
| `batch_size` | 32 | 每批样本数 |
| `learning_rate` | 5e-4 | AdamW 初始 lr |
| `accumulation_steps` | 8 | 累积 8 步更新一次 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |
| `max_seq_len` | 340 | 单样本最大 token 数 |
| `dtype` | bfloat16 | 混合精度类型 |

优化器是 `AdamW`，数据默认 `../dataset/pretrain_hq.jsonl`。

## 梯度累积：为什么除以 accumulation_steps

`loss = loss / accumulation_steps` 让累积 8 个小步的总梯度，量级接近「一个 8 倍大 batch 直接算」的梯度。不除等于人为放大梯度，容易训练不稳。于是参数不是每步更新，而是每 `accumulation_steps` 步更新一次——用小显存模拟大 batch。这条链（backward 累积 → unscale → clip → step → zero_grad）的逐环节解释在 [08-training-mechanics/01-update-skeleton](../08-training-mechanics/01-update-skeleton.md)。

## 保存

每 `save_interval` 步、且是主进程时：把权重 `.half()` 存成 `out/pretrain_512.pth`，同时用 `lm_checkpoint` 存一份完整续训快照到 `checkpoints/`。保存前先 `raw_model = model.module`（拆 DDP 包装）再 `getattr(raw_model, '_orig_mod', raw_model)`（拆 `torch.compile` 包装），拿到原始模型再 `state_dict()`。

## 练习

1. `train_pretrain.py` 分哪两块？建议的阅读顺序是什么？
2. `from_weight` 和 `from_resume` 有什么区别？为什么续训要恢复优化器和 scaler 状态？
3. `train_epoch` 一步里，参数是每步都更新吗？由哪行控制？
4. 数据是在哪里变成张量、又在哪里搬上 GPU 的？

<details>
<summary>参考答案</summary>

1. `train_epoch`（step 级训练）和 `main`（准备 + epoch 循环）；建议先看 `main` 搭起整体，再看 `train_epoch` 看每步更新。
2. `from_weight` 从已有参数出发训练；`from_resume` 恢复完整训练现场（参数 + 优化器 + scaler + epoch/step）。AdamW 维护一二阶动量、scaler 维护缩放比例，不恢复它们训练动力学会变。
3. 不是。`if (step + 1) % args.accumulation_steps == 0:` 控制每累积 8 步才 `scaler.step` 更新一次。
4. `PretrainDataset` 在 CPU 侧把文本编码成张量；训练循环里 `input_ids.to(args.device)` 每步把当前 batch 搬上 GPU。
</details>
