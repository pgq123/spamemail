"""Metrics and artifact helpers for spam-classification experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from runtime_utils import serialize_path_values

MODEL_DISPLAY_NAMES = {
    "mlp": "TF-IDF+MLP",
    "bert_pretrained": "预训练 DistilBERT",
    "bert_zero_trained": "从零训练紧凑型 BERT",
}


def standardize_model_display_name(model_key: str) -> str:
    """Map canonical model key to human-readable display name."""

    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute a stable metric bundle for binary classification."""

    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    prob = np.asarray(y_prob, dtype=float) if y_prob is not None else pred.astype(float)

    metrics = {
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(truth, prob))
    except ValueError:
        metrics["roc_auc"] = 0.5
    return metrics


def average_attention_map(attentions: Any, seq_len: int, sample_index: int = 0) -> np.ndarray | None:
    """Average multi-layer/multi-head attention into one square map."""

    if not attentions:
        return None

    layer_maps: list[np.ndarray] = []
    for layer_attention in attentions:
        if isinstance(layer_attention, torch.Tensor):
            layer_array = layer_attention.detach().float().cpu().numpy()
        else:
            layer_array = np.asarray(layer_attention)
        if layer_array.ndim != 4 or layer_array.size == 0:
            continue
        if sample_index >= layer_array.shape[0]:
            continue
        sample_attention = layer_array[sample_index]  # [heads, seq, seq]
        if sample_attention.ndim != 3:
            continue
        layer_maps.append(sample_attention.mean(axis=0))

    if not layer_maps:
        return None
    arr = np.mean(np.stack(layer_maps, axis=0), axis=0)
    if arr.ndim != 2 or arr.shape[0] < seq_len or arr.shape[1] < seq_len:
        return None
    return arr[:seq_len, :seq_len]


def build_metric_bar_frame(metrics: dict[str, float]) -> pd.DataFrame:
    """Convert metric dict to plotting dataframe."""

    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = [{"metric": key, "value": float(metrics.get(key, 0.0))} for key in keys]
    return pd.DataFrame(rows)


def build_confusion_dataframe(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    """Build labeled confusion-matrix dataframe."""

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(matrix, index=["Ham", "Spam"], columns=["Ham", "Spam"])


def build_prediction_frame(
    ids: Sequence[str],
    texts: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> pd.DataFrame:
    """Build detailed prediction result dataframe."""

    return pd.DataFrame(
        {
            "Email No.": list(ids),
            "text": list(texts),
            "y_true": list(y_true),
            "y_pred": list(y_pred),
            "y_prob_spam": list(y_prob),
        }
    )


def classification_report_frame(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    """Build classification report dataframe from sklearn summary."""

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


def save_experiment_artifacts(
    *,
    result_dir: Path,
    prefix: str,
    history: pd.DataFrame | None,
    metrics: dict[str, float],
    report_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    confusion_frame: pd.DataFrame,
    config_payload: dict[str, Any] | None = None,
    extra_json_payloads: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Persist common tabular/json artifacts and return generated file paths."""

    result_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    metrics_path = result_dir / f"{prefix}_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["metrics_path"] = str(metrics_path)

    report_path = result_dir / f"{prefix}_classification_report.csv"
    report_frame.to_csv(report_path)
    outputs["classification_report_path"] = str(report_path)

    prediction_path = result_dir / f"{prefix}_test_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    outputs["prediction_path"] = str(prediction_path)

    confusion_path = result_dir / f"{prefix}_confusion_matrix.csv"
    confusion_frame.to_csv(confusion_path)
    outputs["confusion_csv_path"] = str(confusion_path)

    if history is not None:
        history_path = result_dir / f"{prefix}_training_history.csv"
        history.to_csv(history_path, index=False)
        outputs["history_path"] = str(history_path)

    if config_payload is not None:
        config_path = result_dir / f"{prefix}_experiment_config.json"
        config_path.write_text(
            json.dumps(serialize_path_values(config_payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        outputs["config_path"] = str(config_path)

    for key, value in (extra_json_payloads or {}).items():
        out = result_dir / f"{prefix}_{key}.json"
        out.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs[f"{key}_path"] = str(out)

    return outputs
