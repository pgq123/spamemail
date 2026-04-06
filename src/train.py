"""统一训练入口：一条命令完成 BERT 双模式与 MLP 的对比实验。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from bert_train import run_experiment
from config import ExperimentConfig
from metrics import standardize_model_display_name
from mlp_train import MLPExperimentConfig, run_mlp_baseline
from visualize import (
    plot_chapter_bridge_chart,
    plot_three_model_metric_comparison,
    plot_three_model_metric_dashboard,
)

METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc", "response_time_ms"]
HIGHER_IS_BETTER = {
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "roc_auc": True,
    "response_time_ms": False,
}
COMPOSITE_WEIGHTS = {
    "accuracy": 0.2,
    "recall": 0.2,
    "f1": 0.25,
    "roc_auc": 0.2,
    "response_time_ms": 0.15,
}


def parse_args() -> argparse.Namespace:
    """解析统一训练入口参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Run one-shot comparison experiments across pretrained BERT, "
            "zero-trained BERT, and MLP baseline."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-path", default="data/sms_spam_collection.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--bert-result-dir", default="results/bert")
    parser.add_argument("--mlp-result-dir", default="results/mlp")
    parser.add_argument("--comparison-result-dir", default="results/comparison")
    parser.add_argument("--result-time-format", default="%y_%m_%d_%H_%M_%S")
    parser.add_argument("--no-timestamped-results", action="store_true")
    parser.add_argument(
        "--run-scope",
        choices=("all", "bert", "mlp"),
        default="all",
        help="all(default)=run pretrained BERT + zero-trained BERT + MLP in one command.",
    )
    parser.add_argument(
        "--bert-modes",
        choices=("both", "pretrained", "zero_trained"),
        default="both",
        help="BERT variants to run when run-scope is all/bert. Default runs both variants.",
    )

    parser.add_argument("--bert-epochs", type=int, default=40)
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--bert-max-length", type=int, default=128)
    parser.add_argument("--bert-learning-rate", type=float, default=3e-5)
    parser.add_argument("--bert-pretrained-model-name", default="distilbert-base-uncased")
    parser.add_argument("--bert-pretrained-local-files-only", action="store_true")
    parser.add_argument("--bert-enable-early-stopping", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")

    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--mlp-max-iter", type=int, default=40)
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    return parser.parse_args()


def _resolve_run_tag(args: argparse.Namespace) -> str | None:
    """根据开关决定是否生成时间戳目录名。"""

    return None if args.no_timestamped_results else datetime.now().strftime(args.result_time_format)


def _join_with_optional_tag(base: Path, run_tag: str | None) -> Path:
    """按需拼接运行标签目录。"""

    return (base / run_tag) if run_tag else base


def _bert_modes_to_run(args: argparse.Namespace) -> List[str]:
    """将 BERT 模式参数统一成列表形式。"""

    return ["pretrained", "zero_trained"] if args.bert_modes == "both" else [args.bert_modes]


def _build_bert_config(args: argparse.Namespace, use_pretrained_backbone: bool, run_tag: str | None) -> ExperimentConfig:
    """构建 BERT 训练配置。"""

    project_root = Path(args.project_root).resolve()
    return ExperimentConfig(
        project_root=project_root,
        data_path=Path(args.data_path),
        result_dir=_join_with_optional_tag(project_root / args.bert_result_dir, run_tag),
        model_dir=_join_with_optional_tag(project_root / args.model_dir, run_tag),
        epochs=args.bert_epochs,
        batch_size=args.bert_batch_size,
        max_length=args.bert_max_length,
        learning_rate=args.bert_learning_rate,
        use_pretrained_backbone=use_pretrained_backbone,
        pretrained_model_name=args.bert_pretrained_model_name,
        pretrained_local_files_only=args.bert_pretrained_local_files_only,
        enable_early_stopping=args.bert_enable_early_stopping,
        force_cpu=args.force_cpu,
        timestamped_results=False,
    )


def _build_mlp_config(args: argparse.Namespace, run_tag: str | None) -> MLPExperimentConfig:
    """构建 MLP 训练配置。"""

    project_root = Path(args.project_root).resolve()
    return MLPExperimentConfig(
        project_root=project_root,
        data_path=Path(args.data_path),
        result_dir=_join_with_optional_tag(project_root / args.mlp_result_dir, run_tag),
        model_dir=_join_with_optional_tag(project_root / args.model_dir, run_tag),
        learning_rate_init=args.mlp_learning_rate,
        batch_size=args.mlp_batch_size,
        max_iter=args.mlp_max_iter,
        alpha=args.mlp_alpha,
        timestamped_results=False,
    )


def _load_metric_file(path: str) -> Dict[str, float]:
    """读取单模型指标文件并筛选数值型字段。"""

    metric_dict = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: float(value) for key, value in metric_dict.items() if isinstance(value, (int, float))}


def _fill_metric_defaults(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required metric columns exist and have numeric values."""

    working = metrics_frame.copy()
    for metric in METRIC_ORDER:
        if metric not in working.columns:
            working[metric] = float("nan")
        working[metric] = pd.to_numeric(working[metric], errors="coerce")

    # Keep ranking stable even when some historical runs do not provide latency.
    if working["response_time_ms"].notna().any():
        replacement = float(working["response_time_ms"].median())
    else:
        replacement = 0.0
    working["response_time_ms"] = working["response_time_ms"].fillna(replacement)
    return working


def _normalized_gap_to_best(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Compute per-model relative gap to best performer for one metric."""

    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if higher_is_better:
        best = float(values.max())
        if best <= 0:
            return pd.Series([0.0] * len(values), index=series.index)
        return (best - values) / best
    best = float(values.min())
    if best <= 0:
        return values - best
    return (values - best) / best


def _normalize_metric(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Min-max normalize a metric into [0, 1], adjusting direction if needed."""

    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_value = float(values.min())
    max_value = float(values.max())
    if abs(max_value - min_value) < 1e-12:
        normalized = pd.Series([1.0] * len(values), index=series.index)
    else:
        normalized = (values - min_value) / (max_value - min_value)
    return normalized if higher_is_better else 1.0 - normalized


def _build_method_comparison_table(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """Build a table with three evaluation methods and their rankings."""

    working = metrics_frame.copy()
    working["method_1_raw_rank_score"] = working["f1"]
    working["method_1_raw_rank"] = (
        working["method_1_raw_rank_score"].rank(method="min", ascending=False).astype(int)
    )

    gap_columns = []
    for metric, higher_is_better in HIGHER_IS_BETTER.items():
        gap_col = f"{metric}_gap_to_best"
        working[gap_col] = _normalized_gap_to_best(working[metric], higher_is_better=higher_is_better)
        gap_columns.append(gap_col)
    working["method_2_gap_score"] = 1.0 - working[gap_columns].mean(axis=1)
    working["method_2_gap_rank"] = working["method_2_gap_score"].rank(method="min", ascending=False).astype(int)

    working["composite_score"] = 0.0
    for metric, weight in COMPOSITE_WEIGHTS.items():
        working["composite_score"] += _normalize_metric(
            working[metric], higher_is_better=HIGHER_IS_BETTER[metric]
        ) * weight
    working["method_3_composite_rank"] = working["composite_score"].rank(method="min", ascending=False).astype(int)

    working["consensus_rank_score"] = (
        working["method_1_raw_rank"] + working["method_2_gap_rank"] + working["method_3_composite_rank"]
    ) / 3.0
    working["rank"] = working["consensus_rank_score"].rank(method="min", ascending=True).astype(int)
    return working


def _build_metric_statistics_table(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean/std/best/worst statistics for each key metric."""

    rows: List[Dict[str, object]] = []
    for metric in METRIC_ORDER + ["composite_score"]:
        series = pd.to_numeric(metrics_frame[metric], errors="coerce")
        if metric == "response_time_ms":
            best_index = int(series.idxmin())
            worst_index = int(series.idxmax())
        else:
            best_index = int(series.idxmax())
            worst_index = int(series.idxmin())

        rows.append(
            {
                "metric": metric,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "best_value": float(series.iloc[best_index]),
                "best_model": str(metrics_frame.iloc[best_index]["model"]),
                "worst_value": float(series.iloc[worst_index]),
                "worst_model": str(metrics_frame.iloc[worst_index]["model"]),
            }
        )
    return pd.DataFrame(rows)


def _append_comparison_row(
    rows: List[Dict[str, object]],
    *,
    model_key: str,
    family: str,
    variant: str,
    metrics: Dict[str, float],
    metrics_path: str,
) -> None:
    """追加一行模型比较记录，统一字段结构。"""

    rows.append(
        {
            "model": model_key,
            "family": family,
            "variant": variant,
            **metrics,
            "metrics_path": metrics_path,
        }
    )


def _write_comparison_artifacts(
    args: argparse.Namespace,
    run_tag: str | None,
    rows: List[Dict[str, object]],
) -> Dict[str, str]:
    """
    导出跨模型比较产物。

    生成内容：
    - CSV/JSON 指标对比表
    - 跨模型总对比图
    - 章节承接图（4->5）
    """

    comparison_dir = _join_with_optional_tag(Path(args.project_root).resolve() / args.comparison_result_dir, run_tag)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    metrics_frame = _fill_metric_defaults(pd.DataFrame(rows))
    metrics_frame = _build_method_comparison_table(metrics_frame)
    metrics_frame = metrics_frame.sort_values(by=["rank", "composite_score"], ascending=[True, False], kind="stable").reset_index(drop=True)
    metrics_frame["is_best"] = metrics_frame["rank"] == metrics_frame["rank"].min()
    metrics_frame["model_display"] = metrics_frame["model"].map(standardize_model_display_name)
    stats_frame = _build_metric_statistics_table(metrics_frame)

    method_table = metrics_frame[
        [
            "model",
            "model_display",
            "method_1_raw_rank_score",
            "method_1_raw_rank",
            "method_2_gap_score",
            "method_2_gap_rank",
            "composite_score",
            "method_3_composite_rank",
            "consensus_rank_score",
            "rank",
        ]
    ].copy()

    comparison_csv = comparison_dir / "three_model_metrics_comparison.csv"
    comparison_json = comparison_dir / "three_model_metrics_comparison.json"
    method_csv = comparison_dir / "three_model_performance_methods.csv"
    stats_csv = comparison_dir / "three_model_metric_statistics.csv"
    cross_model_fig = comparison_dir / "three_model_metrics_comparison.png"
    dashboard_fig = comparison_dir / "three_model_metrics_dashboard.png"
    chapter_bridge_fig = comparison_dir / "chapter_4_to_5_bridge.png"

    metrics_frame.to_csv(comparison_csv, index=False)
    method_table.to_csv(method_csv, index=False)
    stats_frame.to_csv(stats_csv, index=False)
    comparison_json.write_text(
        json.dumps(
            {
                "run_scope": args.run_scope,
                "run_tag": run_tag,
                "best_model": metrics_frame.iloc[0]["model"] if len(metrics_frame) else None,
                "evaluation_methods": [
                    "method_1_raw_rank: based on F1 absolute value (higher is better)",
                    "method_2_gap_score: relative gap to best across all core metrics",
                    "method_3_composite_rank: weighted normalized score including response time",
                ],
                "models": metrics_frame.to_dict(orient="records"),
                "metric_statistics": stats_frame.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plot_ready = metrics_frame.melt(
        id_vars=["model", "model_display"],
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="value",
    )
    plot_three_model_metric_comparison(plot_ready, cross_model_fig)
    plot_three_model_metric_dashboard(plot_ready, metrics_frame, dashboard_fig)
    plot_chapter_bridge_chart(metrics_frame, chapter_bridge_fig)

    return {
        "comparison_csv_path": str(comparison_csv),
        "comparison_json_path": str(comparison_json),
        "method_csv_path": str(method_csv),
        "stats_csv_path": str(stats_csv),
        "cross_model_plot_path": str(cross_model_fig),
        "dashboard_plot_path": str(dashboard_fig),
        "chapter_bridge_plot_path": str(chapter_bridge_fig),
    }


def run_unified_experiment(args: argparse.Namespace) -> Dict[str, object]:
    """
    执行统一实验并输出比较结果。

    模型独立性保证：
    - BERT 与 MLP 使用各自独立配置对象。
    - 结果目录按模型家族隔离，防止产物互相覆盖。
    - 比较阶段仅聚合指标文件，不复用训练中间状态。
    """

    run_tag = _resolve_run_tag(args)
    run_outputs: Dict[str, Dict[str, str]] = {}
    comparison_rows: List[Dict[str, object]] = []

    if args.run_scope in {"all", "bert"}:
        for mode in _bert_modes_to_run(args):
            run_key = f"bert_{mode}"
            outputs = run_experiment(_build_bert_config(args, mode == "pretrained", run_tag))
            run_outputs[run_key] = outputs
            _append_comparison_row(
                comparison_rows,
                model_key=run_key,
                family="bert",
                variant=mode,
                metrics=_load_metric_file(outputs["metrics_path"]),
                metrics_path=outputs["metrics_path"],
            )

    if args.run_scope in {"all", "mlp"}:
        mlp_outputs = run_mlp_baseline(_build_mlp_config(args, run_tag))
        run_outputs["mlp"] = mlp_outputs
        _append_comparison_row(
            comparison_rows,
            model_key="mlp",
            family="mlp",
            variant="baseline",
            metrics=_load_metric_file(mlp_outputs["metrics_path"]),
            metrics_path=mlp_outputs["metrics_path"],
        )

    comparison_outputs = _write_comparison_artifacts(args, run_tag, comparison_rows)
    return {"run_tag": run_tag, "runs": run_outputs, "comparison": comparison_outputs}


if __name__ == "__main__":
    payload = run_unified_experiment(parse_args())
    print(json.dumps(payload, indent=2, ensure_ascii=False))
