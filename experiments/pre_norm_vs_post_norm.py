# experiments/pre_norm_vs_post_norm.py
"""Pre-Norm vs Post-Norm 对比：跑最小网络对比训练稳定性。

锚点：ch02/01 (Block + RMSNorm) + ch02/07 (Normalization Evolution)。
设计一个 2 层小 Transformer，分别用 Pre-Norm 和 Post-Norm 跑相同步数，
观察 loss 收敛速度和梯度爆炸/消失。
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from _base import save_fig, setup

logger = logging.getLogger(__name__)

# 常量
SEED = 42
DIM = 64
HEADS = 2
LAYERS = 2
SEQ_LEN = 16
VOCAB = 256
BATCH = 32
STEPS = 500
LR = 1e-3
DEVICE = torch.device("cpu")  # 明确 CPU，教学 demo 不需要 GPU


def make_synthetic_data(n: int = 1000) -> DataLoader:
    """生成随机序列数据：从 0..VOCAB-1 随机 token，预测下一个 token。"""
    np.random.seed(SEED)
    x = torch.randint(0, VOCAB, (n, SEQ_LEN))
    y = torch.randint(0, VOCAB, (n, SEQ_LEN))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=BATCH, shuffle=True)


class MinimalBlock(nn.Module):
    """最小 Attention + FFN 块，可切换 Pre/Post-Norm。"""

    def __init__(self, dim: int, heads: int, pre_norm: bool = True):
        super().__init__()
        self.pre_norm = pre_norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向。"""
        # Attention 分支
        if self.pre_norm:
            attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        else:
            attn_out, _ = self.attn(x, x, x)
            attn_out = self.norm1(x + attn_out)
        x = x + attn_out

        # FFN 分支
        if self.pre_norm:
            ffn_out = self.ffn(self.norm2(x))
        else:
            ffn_out = self.ffn(x)
            ffn_out = self.norm2(x + ffn_out)
        x = x + ffn_out
        return x


class MinimalTransformer(nn.Module):
    """2 层小 Transformer，用于教学对比。"""

    def __init__(self, vocab: int, dim: int, heads: int, layers: int, pre_norm: bool = True):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([MinimalBlock(dim, heads, pre_norm) for _ in range(layers)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向，返回 logits。"""
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


def train_one(model: nn.Module, loader: DataLoader, steps: int, lr: float) -> Tuple[list, list]:
    """训练指定步数，返回 (loss_history, grad_norm_history)。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses = []
    grad_norms = []

    it = iter(loader)
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        opt.zero_grad()
        loss.backward()

        # 记录梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms.append(total_norm ** 0.5)

        opt.step()
        losses.append(loss.item())

    return losses, grad_norms


def plot_comparison(
    pre_loss: list, post_loss: list,
    pre_grad: list, post_grad: list,
    out_name: str = "pre_vs_post_norm",
) -> str:
    """绘制 Pre/Post-Norm 的 loss 和梯度范数对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(pre_loss, label="Pre-Norm", color="#4472C4", lw=1.2)
    ax.plot(post_loss, label="Post-Norm", color="#ED7D31", lw=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("训练 Loss 曲线")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, lw=0.3)

    ax = axes[1]
    ax.plot(pre_grad, label="Pre-Norm", color="#4472C4", lw=1.2)
    ax.plot(post_grad, label="Post-Norm", color="#ED7D31", lw=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("梯度 L2 范数")
    ax.set_title("梯度范数")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, lw=0.3)

    return save_fig(fig, out_name)


def main() -> None:
    """入口。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    setup()

    logger.info("数据: 合成随机序列, vocab=%d, seq_len=%d", VOCAB, SEQ_LEN)
    loader = make_synthetic_data()

    logger.info("Pre-Norm 训练...")
    model_pre = MinimalTransformer(VOCAB, DIM, HEADS, LAYERS, pre_norm=True).to(DEVICE)
    loss_pre, grad_pre = train_one(model_pre, loader, STEPS, LR)
    logger.info("  末 loss: %.4f, 末梯度范数: %.4f", loss_pre[-1], grad_pre[-1])

    logger.info("Post-Norm 训练...")
    model_post = MinimalTransformer(VOCAB, DIM, HEADS, LAYERS, pre_norm=False).to(DEVICE)
    loss_post, grad_post = train_one(model_post, loader, STEPS, LR)
    logger.info("  末 loss: %.4f, 末梯度范数: %.4f", loss_post[-1], grad_post[-1])

    path = plot_comparison(loss_pre, loss_post, grad_pre, grad_post)
    logger.info("输出: %s", path)


if __name__ == "__main__":
    main()
