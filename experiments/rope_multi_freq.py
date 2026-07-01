# experiments/rope_multi_freq.py
"""RoPE 多频可视化：验证 d/2 组频率从快到慢的设计。"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# 让 _base 在 experiments 目录内可导
sys.path.insert(0, str(Path(__file__).parent))
from _base import save_fig, setup

logger = logging.getLogger(__name__)

# 常量，与 minimind-3 默认 hidden_size=768 一致
DIM = 768
BASE = 10000.0


def compute_rope_frequencies(dim: int, base: float) -> torch.Tensor:
    """计算 RoPE 的 d/2 个频率，从高频到低频。"""
    half = dim // 2
    # theta_i = 1 / (base^(2i/dim)), i=0..half-1
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / dim))
    return freqs


def plot_frequencies(freqs: torch.Tensor, out_name: str = "rope_frequencies") -> str:
    """绘制频率分布图，保存并返回路径。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(freqs))
    ax.bar(x[:64], freqs.numpy()[:64], color="#4472C4", width=0.8)
    ax.set_xlabel("频率组索引 i (0 → d/2-1)")
    ax.set_ylabel(r"频率 $\theta_i$")
    ax.set_title(f"RoPE 频率分布 (dim={DIM}, base={BASE:.0f}) — 前 64 组")
    ax.set_yscale("log")
    # 标注两端
    ax.annotate(
        f"最快: {freqs[0].item():.4f}",
        xy=(0, freqs[0].item()),
        xytext=(10, freqs[0].item() * 2),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
    )
    ax.annotate(
        f"最慢: {freqs[-1].item():.6f}",
        xy=(len(freqs) - 1, freqs[-1].item()),
        xytext=(len(freqs) - 20, freqs[-1].item() * 2),
        arrowprops=dict(arrowstyle="->", color="green"),
        color="green",
    )
    return save_fig(fig, out_name)


def plot_rotation_matrix(m: int, dim: int, base: float, out_name: str = "rope_rotation") -> str:
    """绘制位置 m 上的旋转矩阵示意，展示不同频率的旋转速度。"""
    freqs = compute_rope_frequencies(dim, base)
    # 只取前 16 维做可视化，2 维一组
    show_dims = 16
    thetas = freqs[: show_dims // 2].numpy()
    angles = m * thetas  # 位置 m 上的旋转角

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        theta = thetas[idx]
        angle = angles[idx]
        # 画单位圆
        circle = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(circle), np.sin(circle), "k--", lw=0.5)
        # 画旋转后的向量
        ax.arrow(0, 0, np.cos(angle), np.sin(angle), head_width=0.05, color="#4472C4")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_title(f"组 {idx}: θ={theta:.4f}\nm={m}, 转角={angle:.2f}rad")
        ax.grid(True, lw=0.3)
    fig.suptitle(f"RoPE 位置 m={m} 的旋转示意 (前 8 组, 2 维一组)", fontsize=14)
    return save_fig(fig, out_name)


def main() -> None:
    """入口。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    setup()

    freqs = compute_rope_frequencies(DIM, BASE)
    logger.info("dim=%d, half=%d, 频率范围: %.4f → %.6f", DIM, len(freqs), freqs[0].item(), freqs[-1].item())

    p1 = plot_frequencies(freqs)
    p2 = plot_rotation_matrix(m=10, dim=DIM, base=BASE)
    logger.info("输出: %s, %s", p1, p2)


if __name__ == "__main__":
    main()
