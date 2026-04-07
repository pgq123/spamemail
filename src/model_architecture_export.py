"""Export publication-ready neural network architecture diagrams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _add_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#2d3748",
    fontsize: int = 10,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
        family="SimSun",
    )


def _add_arrow(
    ax: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    text: str | None = None,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.1,
        color="#2d3748",
    )
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.03
        ax.text(mid_x, mid_y, text, ha="center", va="bottom", fontsize=9, color="#2d3748", family="SimSun")


def _setup_canvas(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, title, ha="center", va="center", fontsize=16, weight="bold", family="SimSun")
    return fig, ax


def build_mlp_architecture_figure() -> plt.Figure:
    """Build MLP architecture and mechanism figure."""

    fig, ax = _setup_canvas("TF-IDF + MLP 垃圾短信分类架构")
    y = 0.58
    w = 0.13
    h = 0.18
    x_positions = [0.04, 0.19, 0.34, 0.49, 0.64, 0.79]

    _add_box(ax, x=x_positions[0], y=y, w=w, h=h, text="输入短信文本", facecolor="#edf2ff")
    _add_box(
        ax,
        x=x_positions[1],
        y=y,
        w=w,
        h=h,
        text="文本清洗\n小写化/去URL\n去停用词",
        facecolor="#e6fffa",
    )
    _add_box(
        ax,
        x=x_positions[2],
        y=y,
        w=w,
        h=h,
        text="TF-IDF向量化\nngram=(1,2)\nmax_features=6000",
        facecolor="#ebf8ff",
    )
    _add_box(
        ax,
        x=x_positions[3],
        y=y,
        w=w,
        h=h,
        text="隐藏层1\nDense(256)\nReLU",
        facecolor="#fff5f5",
    )
    _add_box(
        ax,
        x=x_positions[4],
        y=y,
        w=w,
        h=h,
        text="隐藏层2\nDense(128)\nReLU",
        facecolor="#fff5f5",
    )
    _add_box(
        ax,
        x=x_positions[5],
        y=y,
        w=w,
        h=h,
        text="输出层\nDense(2)+Softmax\n垃圾/正常",
        facecolor="#fefcbf",
    )

    for left, right in zip(x_positions[:-1], x_positions[1:]):
        _add_arrow(ax, start=(left + w, y + h / 2), end=(right, y + h / 2))

    _add_box(
        ax,
        x=0.37,
        y=0.22,
        w=0.27,
        h=0.16,
        text="损失函数: CrossEntropy\n优化器: Adam\n机制: 反向传播更新两层全连接权重",
        facecolor="#faf5ff",
    )
    _add_arrow(ax, start=(0.79 + w / 2, y), end=(0.51, 0.38), text="计算损失")
    _add_arrow(ax, start=(0.50, 0.22), end=(0.41, y), text="梯度回传")

    ax.text(
        0.5,
        0.08,
        "工作机制: 先把短文本映射到稀疏统计特征空间, 再由两层感知机学习非线性分类边界。",
        ha="center",
        va="center",
        fontsize=10,
        family="SimSun",
    )
    fig.tight_layout()
    return fig


def build_distilbert_architecture_figure() -> plt.Figure:
    """Build DistilBERT architecture and mechanism figure."""

    fig, ax = _setup_canvas("预训练 DistilBERT 微调架构")
    y = 0.58
    w = 0.13
    h = 0.18
    x_positions = [0.04, 0.19, 0.34, 0.49, 0.64, 0.79]

    _add_box(
        ax,
        x=x_positions[0],
        y=y,
        w=w,
        h=h,
        text="输入短信文本\nTokenizer编码\n[input_ids,mask]",
        facecolor="#edf2ff",
    )
    _add_box(
        ax,
        x=x_positions[1],
        y=y,
        w=w,
        h=h,
        text="Token/位置嵌入\nEmbedding",
        facecolor="#e6fffa",
    )
    _add_box(
        ax,
        x=x_positions[2],
        y=y,
        w=w,
        h=h,
        text="多头自注意力\n(Multi-Head\nSelf-Attention)",
        facecolor="#ebf8ff",
    )
    _add_box(
        ax,
        x=x_positions[3],
        y=y,
        w=w,
        h=h,
        text="前馈网络 FFN\n+ 残差连接\n+ LayerNorm",
        facecolor="#ebf8ff",
    )
    _add_box(
        ax,
        x=x_positions[4],
        y=y,
        w=w,
        h=h,
        text="取 [CLS] 向量\nDropout",
        facecolor="#fff5f5",
    )
    _add_box(
        ax,
        x=x_positions[5],
        y=y,
        w=w,
        h=h,
        text="线性分类头\nLinear(隐藏维->2)\nSoftmax",
        facecolor="#fefcbf",
    )

    for left, right in zip(x_positions[:-1], x_positions[1:]):
        _add_arrow(ax, start=(left + w, y + h / 2), end=(right, y + h / 2))

    ax.text(
        0.415,
        0.50,
        "重复堆叠6层编码器(预训练知识迁移)",
        ha="center",
        va="center",
        fontsize=9,
        color="#2d3748",
        family="SimSun",
    )

    _add_box(
        ax,
        x=0.36,
        y=0.22,
        w=0.30,
        h=0.16,
        text="损失函数: CrossEntropy\n优化器: AdamW + 线性Warmup调度\n机制: 端到端微调预训练参数",
        facecolor="#faf5ff",
    )
    _add_arrow(ax, start=(0.79 + w / 2, y), end=(0.52, 0.38), text="监督信号")
    _add_arrow(ax, start=(0.50, 0.22), end=(0.41, y), text="梯度反传")

    ax.text(
        0.5,
        0.08,
        "工作机制: 自注意力在全局上下文中建模词间依赖, 利用预训练语义表征提升垃圾短信识别精度。",
        ha="center",
        va="center",
        fontsize=10,
        family="SimSun",
    )
    fig.tight_layout()
    return fig


def _save_figure(fig: plt.Figure, output_base: Path) -> dict[str, str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png_path": str(png_path), "pdf_path": str(pdf_path)}


def export_model_architecture_figures(output_dir: Path) -> dict[str, str]:
    """Export MLP and DistilBERT architecture figures as PNG/PDF."""

    output_dir = Path(output_dir)
    mlp_outputs = _save_figure(
        build_mlp_architecture_figure(),
        output_dir / "mlp_neural_network_architecture",
    )
    distilbert_outputs = _save_figure(
        build_distilbert_architecture_figure(),
        output_dir / "distilbert_neural_network_architecture",
    )
    return {
        "mlp_png_path": mlp_outputs["png_path"],
        "mlp_pdf_path": mlp_outputs["pdf_path"],
        "distilbert_png_path": distilbert_outputs["png_path"],
        "distilbert_pdf_path": distilbert_outputs["pdf_path"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MLP and DistilBERT architecture figures.")
    parser.add_argument("--output-dir", default="docs/flowcharts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = export_model_architecture_figures(Path(args.output_dir))
    print(json.dumps(payload, ensure_ascii=False))
