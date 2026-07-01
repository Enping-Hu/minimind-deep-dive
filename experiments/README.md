# experiments

本目录存放可复现的教学小实验，CPU 即可运行，每个都从书中已有锚点长出来。

## 运行依赖

- Python ≥ 3.8
- torch、numpy、matplotlib

```bash
pip install torch numpy matplotlib
```

## 实验列表

### rope_multi_freq.py

**锚点**：ch02/03 (RoPE 旋转编码)。

可视化 RoPE 的 d/2 组频率从快到慢，以及不同位置上的旋转角速度。验证：高频组在短距离敏感、低频组在长距离保持。

```bash
python experiments/rope_multi_freq.py
```

输出 `outputs/rope_frequencies.png` 和 `outputs/rope_rotation.png`。

### pre_norm_vs_post_norm.py

**锚点**：ch02/01 (Block + RMSNorm) + ch02/07 (Normalization Evolution)。

用最小 2 层 Transformer 分别跑 Pre-Norm 和 Post-Norm，对比 loss 收敛和梯度范数。验证：为何 Pre-Norm 更稳定、为何 Post-Norm 在深层更容易梯度爆炸/消失。

```bash
python experiments/pre_norm_vs_post_norm.py
```

输出 `outputs/pre_vs_post_norm.png`。

## 结构

```text
experiments/
├── _base.py          # 公共基础：种子、设备、存图
├── rope_multi_freq.py
├── pre_norm_vs_post_norm.py
└── README.md
```

运行前不需要修改任何路径。`_base.py` 自动把图片存到 `experiments/outputs/`。

## 说明

- 这些脚本只生成图，不提交图到仓库（`.gitignore` 忽略 `outputs/`）。
- 读者拿到仓库后本地运行，自己生成图，保证可复现。
- 若环境确实无法运行（如纯 Web 环境），可引用书中已有的图，但脚本本身仍是规范实现。
