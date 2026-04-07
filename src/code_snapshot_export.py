"""Render key code snippets into image snapshots for paper figures."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SnapshotSpec:
    name: str
    source_path: Path
    start_marker: str
    end_marker: str
    title: str
    include_after: int = 0


def _extract_snippet(lines: list[str], start_marker: str, end_marker: str, include_after: int = 0) -> list[str]:
    start_idx = next((idx for idx, line in enumerate(lines) if start_marker in line), None)
    if start_idx is None:
        raise ValueError(f"Start marker not found: {start_marker}")

    end_idx = next((idx for idx in range(start_idx, len(lines)) if end_marker in lines[idx]), None)
    if end_idx is None:
        raise ValueError(f"End marker not found: {end_marker}")
    if end_idx < start_idx:
        raise ValueError("Invalid snippet marker range.")
    stop_idx = min(len(lines), end_idx + 1 + max(include_after, 0))
    return lines[start_idx:stop_idx]


def _render_snapshot(*, title: str, snippet_lines: list[str], output_path: Path, line_offset: int) -> Path:
    numbered_lines = [f"{line_offset + idx:>4}: {line.rstrip()}" for idx, line in enumerate(snippet_lines)]
    content = "\n".join(numbered_lines)

    max_chars = max((len(line) for line in numbered_lines), default=80)
    width = min(24.0, max(11.0, max_chars * 0.11))
    height = min(32.0, max(6.0, len(numbered_lines) * 0.33 + 1.8))

    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.axis("off")
    ax.text(
        0.01,
        0.99,
        title,
        va="top",
        ha="left",
        fontsize=12,
        color="#f8fafc",
        family="SimSun",
        weight="bold",
    )
    ax.text(
        0.01,
        0.95,
        content,
        va="top",
        ha="left",
        fontsize=8.8,
        color="#e2e8f0",
        family="DejaVu Sans Mono",
        linespacing=1.23,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_path


def export_core_code_snapshots(output_dir: Path, project_root: Path | None = None) -> dict[str, str]:
    """Generate PNG snapshots for key implementation code blocks."""

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir)

    specs = [
        SnapshotSpec(
            name="distilbert_model_core",
            source_path=root / "src" / "model.py",
            start_marker="class BertSpamClassifier(nn.Module):",
            end_marker="attentions=getattr(backbone_outputs, \"attentions\", None),",
            title="核心代码截图1: DistilBERT 分类器结构",
            include_after=1,
        ),
        SnapshotSpec(
            name="mlp_pipeline_core",
            source_path=root / "src" / "mlp_train.py",
            start_marker="pipeline = Pipeline(",
            end_marker="pipeline.fit(x_train, y_train)",
            title="核心代码截图2: TF-IDF + MLP 管线",
            include_after=1,
        ),
        SnapshotSpec(
            name="bert_training_loop_core",
            source_path=root / "src" / "bert_train.py",
            start_marker="for epoch in range(1, runtime_cfg.epochs + 1):",
            end_marker="model.load_state_dict(best_state)",
            title="核心代码截图3: DistilBERT 训练循环",
            include_after=1,
        ),
    ]

    outputs: dict[str, str] = {}
    for spec in specs:
        lines = spec.source_path.read_text(encoding="utf-8").splitlines()
        start_idx = next((idx for idx, line in enumerate(lines) if spec.start_marker in line), None)
        if start_idx is None:
            raise ValueError(f"Start marker not found for {spec.name}")
        snippet_lines = _extract_snippet(lines, spec.start_marker, spec.end_marker, include_after=spec.include_after)
        out_path = output_dir / f"{spec.name}.png"
        _render_snapshot(title=spec.title, snippet_lines=snippet_lines, output_path=out_path, line_offset=start_idx + 1)
        outputs[f"{spec.name}_path"] = str(out_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-ready core code snapshots.")
    parser.add_argument("--output-dir", default="docs/code_snapshots")
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = export_core_code_snapshots(Path(args.output_dir), project_root=Path(args.project_root).resolve())
    print(json.dumps(payload, ensure_ascii=False))
