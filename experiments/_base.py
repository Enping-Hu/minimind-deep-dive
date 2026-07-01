# experiments/_base.py
"""教学实验公共基础：设备检测、种子设置、图片保存。"""

import logging
import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")  # 无头环境，不写交互后端

# 中文字体配置：macOS 优先 PingFang/STHeiti，Linux 优先 Noto Sans CJK
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "STHeiti",
    "Heiti SC",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False  # 减号用 ASCII，不转 Unicode

logger = logging.getLogger(__name__)

# 常量
DEFAULT_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def setup(seed: int = DEFAULT_SEED) -> torch.device:
    """设置随机种子并返回可用设备。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: %s, 种子: %d", device, seed)
    return device


def save_fig(
    fig: plt.Figure,
    name: str,
    out_dir: Optional[str] = None,
    dpi: int = 150,
) -> str:
    """保存图片到输出目录，返回保存路径。"""
    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("图片已保存: %s", path)
    return path
