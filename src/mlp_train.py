"""TF-IDF + MLP baseline training script."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from metrics import (
    build_confusion_dataframe,
    build_metric_bar_frame,
    build_prediction_frame,
    classification_report_frame,
    compute_classification_metrics,
    save_experiment_artifacts,
)
from preprocess import SpamDataAdapter, get_text_label_pairs
from runtime_utils import resolve_run_paths
from visualize import plot_confusion_matrix, plot_metric_bars, plot_training_curves


@dataclass
class MLPExperimentConfig:
    """Configuration for TF-IDF + MLP baseline."""

    project_root: Path = Path(".")
    data_path: Path = Path("data/sms_spam_collection.csv")
    model_dir: Path = Path("models")
    result_dir: Path = Path("results/mlp")
    result_time_format: str = "%y_%m_%d_%H_%M_%S"
    timestamped_results: bool = True

    learning_rate_init: float = 1e-3
    batch_size: int = 64
    max_iter: int = 40
    alpha: float = 1e-4
    random_seed: int = 42
    max_features: int = 6000
    ngram_min: int = 1
    ngram_max: int = 2
    hidden_layer_sizes: tuple[int, int] = (256, 128)


def run_mlp_baseline(config: MLPExperimentConfig) -> dict[str, str]:
    """Train/evaluate MLP baseline and export artifacts."""

    project_root, data_path, model_dir, result_dir = resolve_run_paths(
        project_root=config.project_root,
        data_path=config.data_path,
        model_dir=config.model_dir,
        result_dir=config.result_dir,
        timestamped_results=config.timestamped_results,
        result_time_format=config.result_time_format,
    )
    runtime_cfg = replace(config, project_root=project_root, data_path=data_path, model_dir=model_dir, result_dir=result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = _to_adapter_config(runtime_cfg)
    adapter = SpamDataAdapter(adapter_config)
    prepared = adapter.prepare_dataframe()
    splits = adapter.train_val_test_split(prepared)

    x_train, y_train = get_text_label_pairs(splits.train, adapter_config.text_column, adapter_config.target_column)
    x_test, y_test = get_text_label_pairs(splits.test, adapter_config.text_column, adapter_config.target_column)
    ids_test = splits.test[adapter_config.id_column].astype(str).tolist()
    texts_test = list(x_test)

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=runtime_cfg.max_features,
                    ngram_range=(runtime_cfg.ngram_min, runtime_cfg.ngram_max),
                ),
            ),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=runtime_cfg.hidden_layer_sizes,
                    activation="relu",
                    alpha=runtime_cfg.alpha,
                    learning_rate_init=runtime_cfg.learning_rate_init,
                    batch_size=runtime_cfg.batch_size,
                    max_iter=runtime_cfg.max_iter,
                    random_state=runtime_cfg.random_seed,
                    early_stopping=False,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    prob = pipeline.predict_proba(x_test)[:, 1]
    pred = pipeline.predict(x_test)
    metrics = compute_classification_metrics(y_test, pred, prob)
    metrics["response_time_ms"] = 0.0

    metric_frame = build_metric_bar_frame(metrics)
    confusion_frame = build_confusion_dataframe(y_test, pred)
    prediction_frame = build_prediction_frame(ids_test, texts_test, y_test, pred.tolist(), prob.tolist())
    report_frame = classification_report_frame(y_test, pred)

    loss_curve = list(getattr(pipeline.named_steps["mlp"], "loss_curve_", []))
    if loss_curve:
        history = pd.DataFrame(
            {
                "epoch": list(range(1, len(loss_curve) + 1)),
                "train_loss": loss_curve,
                "val_loss": loss_curve,
                "train_accuracy": [metrics["accuracy"]] * len(loss_curve),
                "val_accuracy": [metrics["accuracy"]] * len(loss_curve),
            }
        )
    else:
        history = pd.DataFrame(
            {
                "epoch": [1],
                "train_loss": [0.0],
                "val_loss": [0.0],
                "train_accuracy": [metrics["accuracy"]],
                "val_accuracy": [metrics["accuracy"]],
            }
        )

    outputs = save_experiment_artifacts(
        result_dir=result_dir,
        prefix="mlp",
        history=history,
        metrics=metrics,
        report_frame=report_frame,
        prediction_frame=prediction_frame,
        confusion_frame=confusion_frame,
        config_payload=asdict(runtime_cfg),
    )

    curve_path = plot_training_curves(history, result_dir / "mlp_training_curves.png")
    metric_path = plot_metric_bars(metric_frame, result_dir / "mlp_metric_bars.png", title="TF-IDF+MLP 测试集指标图")
    cm_path = plot_confusion_matrix(confusion_frame, result_dir / "mlp_confusion_matrix.png")
    outputs["training_curve_path"] = str(curve_path)
    outputs["metric_bar_path"] = str(metric_path)
    outputs["confusion_plot_path"] = str(cm_path)

    model_path = model_dir / "mlp_spam_classifier.pkl"
    joblib.dump(pipeline, model_path)
    outputs["model_path"] = str(model_path)
    return outputs


def _to_adapter_config(config: MLPExperimentConfig):
    from config import ExperimentConfig

    return ExperimentConfig(
        project_root=config.project_root,
        data_path=config.data_path,
        model_dir=config.model_dir,
        result_dir=config.result_dir,
        result_time_format=config.result_time_format,
        timestamped_results=config.timestamped_results,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF + MLP baseline experiment.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-path", default="data/sms_spam_collection.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--result-dir", default="results/mlp")
    parser.add_argument("--result-time-format", default="%y_%m_%d_%H_%M_%S")
    parser.add_argument("--no-timestamped-results", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=1e-4)
    return parser.parse_args()


def _args_to_config(args: argparse.Namespace) -> MLPExperimentConfig:
    return MLPExperimentConfig(
        project_root=Path(args.project_root),
        data_path=Path(args.data_path),
        model_dir=Path(args.model_dir),
        result_dir=Path(args.result_dir),
        result_time_format=args.result_time_format,
        timestamped_results=not args.no_timestamped_results,
        learning_rate_init=args.learning_rate,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    payload = run_mlp_baseline(_args_to_config(parse_args()))
    print(json.dumps(payload, indent=2, ensure_ascii=False))

