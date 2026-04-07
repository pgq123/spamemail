"""Plotting utilities for training and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

# Use non-interactive backend to support headless test/CI environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import pandas as pd
import seaborn as sns

from metrics import MODEL_DISPLAY_NAMES

FONT_SIZE_PT = 12
LINE_SPACING = 1.5
TITLE_PAD = 12
LABEL_PAD = 8
MODEL_COLUMN_FONT_SIZE_PT = 10

PALETTE = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "teal": "#2a9d8f",
    "red": "#d62728",
    "gray": "#4c566a",
}

MODEL_PALETTE = {
    MODEL_DISPLAY_NAMES["mlp"]: "#1f77b4",
    MODEL_DISPLAY_NAMES["bert_pretrained"]: "#ff7f0e",
    MODEL_DISPLAY_NAMES["bert_zero_trained"]: "#2a9d8f",
}

METRIC_DISPLAY = {
    "accuracy": "\u51c6\u786e\u7387",
    "precision": "\u7cbe\u786e\u7387",
    "recall": "\u53ec\u56de\u7387",
    "f1": "F1 \u503c",
    "roc_auc": "ROC-AUC",
    "response_time_ms": "\u54cd\u5e94\u65f6\u95f4(ms)",
}


def apply_ieee_style() -> None:
    """Apply a consistent publication-style theme for all figures."""

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            # Keep latin glyphs in Times New Roman and Chinese glyphs in SimSun by fallback.
            "font.family": ["Times New Roman", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "font.size": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "lines.linewidth": 1.8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
        }
    )


def _apply_layout_and_typography(fig: plt.Figure) -> None:
    """Enforce typography and spacing spec on all text/axes elements."""

    for ax in fig.axes:
        title_text = ax.get_title()
        if title_text:
            ax.set_title(title_text, fontsize=FONT_SIZE_PT, pad=TITLE_PAD)
            ax.title.set_linespacing(LINE_SPACING)
        ax.xaxis.label.set_fontsize(FONT_SIZE_PT)
        ax.yaxis.label.set_fontsize(FONT_SIZE_PT)
        ax.xaxis.labelpad = LABEL_PAD
        ax.yaxis.labelpad = LABEL_PAD
        ax.tick_params(axis="both", pad=6)

    for text_obj in fig.findobj(match=Text):
        text_obj.set_fontsize(FONT_SIZE_PT)
        text_obj.set_linespacing(LINE_SPACING)


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Persist figure to disk and release pyplot resources."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _apply_layout_and_typography(fig)
    fig.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    return output_path


def plot_training_curves(history: pd.DataFrame, output_path: Path) -> Path:
    """Plot side-by-side loss and accuracy curves across epochs."""

    apply_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="\u8bad\u7ec3\u635f\u5931", color=PALETTE["blue"])
    axes[0].plot(history["epoch"], history["val_loss"], label="\u9a8c\u8bc1\u635f\u5931", color=PALETTE["orange"])
    axes[0].set_title("\u8bad\u7ec3\u4e0e\u9a8c\u8bc1\u635f\u5931\u66f2\u7ebf")
    axes[0].set_xlabel("\u8f6e\u6b21")
    axes[0].set_ylabel("\u635f\u5931")
    axes[0].legend(frameon=False)
    axes[1].plot(history["epoch"], history["train_accuracy"], label="\u8bad\u7ec3\u51c6\u786e\u7387", color=PALETTE["teal"])
    axes[1].plot(history["epoch"], history["val_accuracy"], label="\u9a8c\u8bc1\u51c6\u786e\u7387", color=PALETTE["red"])
    axes[1].set_title("\u8bad\u7ec3\u4e0e\u9a8c\u8bc1\u51c6\u786e\u7387\u66f2\u7ebf")
    axes[1].set_xlabel("\u8f6e\u6b21")
    axes[1].set_ylabel("\u51c6\u786e\u7387")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_attention_heatmap(
    tokens: Sequence[str],
    attention_map,
    output_path: Path,
    title: str = "\u6ce8\u610f\u529b\u6743\u91cd\u70ed\u529b\u56fe",
) -> Path:
    """Render token-level attention heatmap for qualitative inspection."""

    apply_ieee_style()
    display_tokens = list(tokens)
    max_tokens = 36
    if len(display_tokens) > max_tokens:
        display_tokens = display_tokens[:max_tokens]
        attention_map = attention_map[:max_tokens, :max_tokens]

    fig_width = min(12, max(6, len(display_tokens) * 0.35))
    fig_height = min(10, max(5, len(display_tokens) * 0.30))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        attention_map,
        cmap="Blues",
        square=True,
        cbar=True,
        xticklabels=display_tokens,
        yticklabels=display_tokens,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("\u952e\u8bcd\u5143")
    ax.set_ylabel("\u67e5\u8be2\u8bcd\u5143")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_metric_bars(metric_frame: pd.DataFrame, output_path: Path, title: str = "\u6d4b\u8bd5\u96c6\u6307\u6807\u5bf9\u6bd4") -> Path:
    """Visualize key metrics as a bar chart with numeric annotations."""

    apply_ieee_style()
    metric_frame = metric_frame.copy()
    metric_frame["metric_display"] = metric_frame["metric"].map(METRIC_DISPLAY).fillna(metric_frame["metric"])
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["teal"], PALETTE["red"], PALETTE["gray"]]
    sns.barplot(
        data=metric_frame.reset_index(drop=True),
        x="metric_display",
        y="value",
        hue="metric_display",
        palette=colors[: len(metric_frame)],
        dodge=False,
        ax=ax,
    )
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    max_value = float(metric_frame["value"].max()) if len(metric_frame) else 1.0
    y_upper = max_value + max(0.03, max_value * 0.08)
    ax.set_ylim(0, y_upper)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("\u6307\u6807\u503c")
    for patch, value in zip(ax.patches, metric_frame["value"]):
        # Add labels on bars to avoid manual value lookup from axes.
        ax.annotate(
            f"{value:.3f}",
            (patch.get_x() + patch.get_width() / 2.0, min(value + y_upper * 0.012, y_upper * 0.985)),
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_PT,
        )
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_confusion_matrix(confusion_frame: pd.DataFrame, output_path: Path, title: str = "\u6df7\u6dc6\u77e9\u9635") -> Path:
    """Plot confusion matrix with integer annotations."""

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    color_upper = max(int(confusion_frame.to_numpy().sum()), 1)
    sns.heatmap(
        confusion_frame,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        linewidths=0.8,
        linecolor="white",
        vmin=0,
        vmax=color_upper,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("\u9884\u6d4b\u6807\u7b7e")
    ax.set_ylabel("\u771f\u5b9e\u6807\u7b7e")
    ax.set_aspect("equal")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_three_model_metric_comparison(metric_frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot grouped bars for three-model horizontal metric comparison."""

    apply_ieee_style()
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    working = metric_frame.copy()
    working = working[working["metric"].isin(metric_keys)]
    working["metric_display"] = working["metric"].map(METRIC_DISPLAY)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    sns.barplot(
        data=working,
        x="metric_display",
        y="value",
        hue="model_display",
        hue_order=[
            MODEL_DISPLAY_NAMES["mlp"],
            MODEL_DISPLAY_NAMES["bert_pretrained"],
            MODEL_DISPLAY_NAMES["bert_zero_trained"],
        ],
        palette=MODEL_PALETTE,
        ax=ax,
    )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("")
    ax.set_ylabel("\u6307\u6807\u503c")
    ax.tick_params(axis="x", rotation=8)
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        title="\u6a21\u578b",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("\u4e09\u6a21\u578b\u5173\u952e\u6d4b\u8bd5\u6307\u6807\u603b\u5bf9\u6bd4", fontsize=FONT_SIZE_PT, y=0.985)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.98)
    return _save_figure(fig, output_path)


def plot_chapter_bridge_chart(summary_frame: pd.DataFrame, output_path: Path) -> Path:
    """Plot chapter-bridge chart for transitioning from analysis to conclusion."""

    apply_ieee_style()
    working = summary_frame.copy()
    working = working.sort_values(by="f1", ascending=False, kind="stable").reset_index(drop=True)
    working["display"] = working["model_display"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})

    sns.barplot(
        data=working,
        x="display",
        y="f1",
        hue="display",
        palette=MODEL_PALETTE,
        dodge=False,
        ax=axes[0],
    )
    legend = axes[0].get_legend()
    if legend is not None:
        legend.remove()
    axes[0].set_ylim(0, 1)
    axes[0].set_title("\u6a21\u578b\u4e3b\u6307\u6807\u6392\u5e8f\uff08F1\uff09")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("F1 \u503c")
    axes[0].tick_params(axis="x", rotation=10)

    scatter = working[["model_display", "accuracy", "roc_auc"]].copy()
    for _, row in scatter.iterrows():
        axes[1].scatter(
            row["accuracy"],
            row["roc_auc"],
            s=90,
            color=MODEL_PALETTE.get(row["model_display"], PALETTE["gray"]),
            label=row["model_display"],
        )
        axes[1].annotate(row["model_display"], (row["accuracy"], row["roc_auc"]), xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("\u51c6\u786e\u7387\u4e0e ROC-AUC \u627f\u63a5\u5173\u7cfb\u56fe")
    axes[1].set_xlabel("\u51c6\u786e\u7387")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_three_model_metric_dashboard(
    metric_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Render a linked dashboard that combines histograms with a quantitative table.

    The top panels keep bar-style comparisons, while the bottom panel embeds the
    exact table values from the same summary data source.
    """

    apply_ieee_style()
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    working = metric_frame.copy()
    working = working[working["metric"].isin(metric_keys)]
    working["metric_display"] = working["metric"].map(METRIC_DISPLAY)

    summary = summary_frame.copy()
    summary = summary.sort_values(by="rank", ascending=True, kind="stable")

    fig = plt.figure(figsize=(15, 9.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], width_ratios=[2.0, 1.15], hspace=0.22, wspace=0.28)
    bar_ax = fig.add_subplot(grid[0, 0])
    latency_ax = fig.add_subplot(grid[0, 1])
    table_ax = fig.add_subplot(grid[1, :])

    sns.barplot(
        data=working,
        x="metric_display",
        y="value",
        hue="model_display",
        hue_order=[
            MODEL_DISPLAY_NAMES["mlp"],
            MODEL_DISPLAY_NAMES["bert_pretrained"],
            MODEL_DISPLAY_NAMES["bert_zero_trained"],
        ],
        palette=MODEL_PALETTE,
        ax=bar_ax,
    )
    bar_ax.set_ylim(0, 1)
    bar_ax.set_title("\u5173\u952e\u6027\u80fd\u6307\u6807\u76f4\u65b9\u5bf9\u6bd4")
    bar_ax.set_xlabel("")
    bar_ax.set_ylabel("\u6307\u6807\u503c")
    bar_ax.tick_params(axis="x", rotation=8)
    bar_handles, bar_labels = bar_ax.get_legend_handles_labels()
    bar_legend = bar_ax.get_legend()
    if bar_legend is not None:
        bar_legend.remove()
    fig.legend(
        bar_handles,
        bar_labels,
        title="\u6a21\u578b",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )

    latency_sorted = summary.sort_values(by="response_time_ms", ascending=True, kind="stable")
    sns.barplot(
        data=latency_sorted,
        x="response_time_ms",
        y="model_display",
        hue="model_display",
        palette=MODEL_PALETTE,
        dodge=False,
        ax=latency_ax,
    )
    latency_legend = latency_ax.get_legend()
    if latency_legend is not None:
        latency_legend.remove()
    latency_ax.set_title("\u5e73\u5747\u54cd\u5e94\u65f6\u95f4\u5bf9\u6bd4")
    latency_ax.set_xlabel("\u6beb\u79d2/\u6837\u672c")
    latency_ax.set_ylabel("")
    latency_upper = float(latency_sorted["response_time_ms"].max()) if len(latency_sorted) else 1.0
    latency_ax.set_xlim(0, latency_upper * 1.25 + 0.02)

    for patch, value in zip(latency_ax.patches, latency_sorted["response_time_ms"]):
        latency_ax.annotate(
            f"{value:.2f}",
            (value, patch.get_y() + patch.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=FONT_SIZE_PT,
            xytext=(4, 0),
            textcoords="offset points",
        )

    table_ax.axis("off")
    table_cols = [
        "rank",
        "model_display",
        "accuracy",
        "recall",
        "f1",
        "roc_auc",
        "response_time_ms",
        "composite_score",
    ]
    table_view = summary[table_cols].copy()
    table_view = table_view.rename(
        columns={
            "rank": "\u6392\u540d",
            "model_display": "\u6a21\u578b",
            "accuracy": "\u51c6\u786e\u7387",
            "recall": "\u53ec\u56de\u7387",
            "f1": "F1",
            "roc_auc": "ROC-AUC",
            "response_time_ms": "\u54cd\u5e94(ms)",
            "composite_score": "\u7efc\u5408\u5f97\u5206",
        }
    )
    for col in ["\u51c6\u786e\u7387", "\u53ec\u56de\u7387", "F1", "ROC-AUC", "\u7efc\u5408\u5f97\u5206"]:
        table_view[col] = table_view[col].map(lambda value: f"{float(value):.3f}")
    table_view["\u54cd\u5e94(ms)"] = table_view["\u54cd\u5e94(ms)"].map(lambda value: f"{float(value):.3f}")
    table_view["\u6a21\u578b"] = table_view["\u6a21\u578b"].replace(
        {
            "\u9884\u8bad\u7ec3 DistilBERT": "\u9884\u8bad\u7ec3\nDistilBERT",
            "TF-IDF+MLP": "TF-IDF+\nMLP",
            "\u4ece\u96f6\u8bad\u7ec3\u7d27\u51d1\u578b BERT": "\u4ece\u96f6\u8bad\u7ec3\n\u7d27\u51d1\u578b BERT",
        }
    )

    table = table_ax.table(
        cellText=table_view.values,
        colLabels=table_view.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.09, 0.18, 0.11, 0.11, 0.11, 0.12, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE_PT)
    table.scale(1, 1.95)
    for row_idx in range(1, len(table_view) + 1):
        table[(row_idx, 1)].get_text().set_fontsize(MODEL_COLUMN_FONT_SIZE_PT)

    fig.suptitle(
        "\u4e09\u6a21\u578b\u6027\u80fd\u5bf9\u6bd4",
        fontsize=FONT_SIZE_PT,
        y=0.98,
    )
    fig.subplots_adjust(top=0.86, bottom=0.06, left=0.06, right=0.98)
    return _save_figure(fig, output_path)
