# 环境与快速开始

目标：先把 MiniMind 跑起来看到输出，再决定从哪一章深入。两条路——直接复用已训练权重（几分钟），或从零训练最小链路（几小时）。

## 环境

需要 `python>=3.10`。在 MiniMind 源码根目录：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

先确认 CUDA 可用，否则训练会退回 CPU、慢到不可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 方式一：跑别人训练好的权重

最快看到对话效果。下载 Transformers 格式的 MiniMind2 权重，直接用 `eval_llm.py` 加载：

```bash
git clone https://huggingface.co/jingyaogong/MiniMind2
python -u eval_llm.py --load_from ./MiniMind2
```

`--load_from` 指向一个 Transformers 格式目录时，按目录加载；指向 `model` 时则加载 `./out/` 下的原生 `.pth` 权重（方式二会用到）。

## 方式二：从零训练最小链路

下载数据、依次跑预训练和 SFT，得到自己的权重。

**1. 下载数据** 到 `./dataset/`。最快复现一个对话模型，只需两份（来自 [minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)）：

- `pretrain_hq.jsonl`（预训练语料）
- `sft_mini_512.jsonl`（SFT 对话数据）

**2. 预训练**，输出 `./out/pretrain_*.pth`（`*` 为模型维度，默认 512）：

```bash
python -u trainer/train_pretrain.py 2>&1 | tee logs/pretrain.log
```

**3. 监督微调**，从预训练权重续训，输出 `./out/full_sft_*.pth`：

```bash
python -u trainer/train_full_sft.py 2>&1 | tee logs/full_sft.log
```

训练脚本默认每 100 步把权重覆盖保存到 `./out/`。LoRA、蒸馏、DPO/PPO/GRPO 等阶段在对应章节展开。

**4. 测自己的权重**。确认 `*.pth` 在 `./out/` 下，`--weight` 指定权重名前缀：

```bash
python -u eval_llm.py --weight full_sft   # 也可 pretrain / dpo / ppo_actor / grpo / spo
```

## 断点续训

所有训练脚本加 `--from_resume 1` 即可从 `./checkpoints/` 自动恢复（模型、优化器、step 进度都在内），适合长训练或不稳定环境：

```bash
python -u trainer/train_pretrain.py --from_resume 1 2>&1 | tee -a logs/pretrain.log
```

## 一个日志习惯

上面的命令都带 `python -u ... 2>&1 | tee 日志文件`：

- `-u` 关闭 stdout 缓冲，训练进度实时刷出来，而不是攒一批才显示；
- `2>&1 | tee` 把屏幕输出同时存进日志文件，长训练中断后还能回看 loss 曲线和报错。

后面章节涉及训练/评测命令时会沿用这个写法。

## 练习

1. `eval_llm.py` 的 `--load_from ./MiniMind2` 和 `--weight full_sft` 两种用法，分别加载的是哪种格式、放在哪里的权重？
2. 训练中途机器重启，怎么不丢进度接着训？

<details>
<summary>参考答案</summary>

1. `--load_from ./MiniMind2` 加载一个 Transformers 格式目录；`--weight full_sft` 配合默认 `--load_from model`，加载 `./out/full_sft_*.pth` 这样的原生 torch 权重。
2. 给训练脚本加 `--from_resume 1`，从 `./checkpoints/` 下的 `<权重名>_<维度>_resume.pth` 自动恢复模型、优化器与 step。
</details>
