"""BERT experiment training entrypoint and helpers."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from config import ExperimentConfig
from metrics import (
    build_confusion_dataframe,
    build_metric_bar_frame,
    build_prediction_frame,
    classification_report_frame,
    compute_classification_metrics,
    save_experiment_artifacts,
)
from model import BertSpamClassifier
from preprocess import SpamDataAdapter, get_text_label_pairs
from runtime_utils import resolve_run_paths
from visualize import (
    plot_attention_heatmap,
    plot_confusion_matrix,
    plot_metric_bars,
    plot_training_curves,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokenize_texts(tokenizer: Any, texts: Iterable[str], max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def build_dataloader(
    frame: pd.DataFrame,
    tokenizer: Any,
    *,
    text_column: str,
    target_column: str,
    batch_size: int,
    max_length: int,
    shuffle: bool,
) -> DataLoader:
    """Build lightweight dataloader from dataframe."""

    texts, labels = get_text_label_pairs(frame, text_column=text_column, target_column=target_column)
    encoded = _tokenize_texts(tokenizer, texts, max_length=max_length)
    dataset = []
    for idx in range(len(labels)):
        dataset.append(
            {
                "input_ids": encoded["input_ids"][idx],
                "attention_mask": encoded["attention_mask"][idx],
                "labels": torch.tensor(int(labels[idx]), dtype=torch.long),
            }
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def epoch_pass(
    model: BertSpamClassifier,
    dataloader: DataLoader,
    *,
    device: torch.device,
    optimizer: AdamW | None = None,
    scheduler: Any | None = None,
    train: bool,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict[str, float]:
    """Run one train/validation epoch and return loss + accuracy."""

    model.train(mode=train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = dataloader
    if show_progress:
        iterator = tqdm(
            dataloader,
            desc=progress_desc or ("Train" if train else "Eval"),
            leave=False,
            dynamic_ncols=True,
        )

    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if train and optimizer is not None:
            optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)
        preds = torch.argmax(outputs.logits, dim=-1)

        if train and optimizer is not None:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((preds == labels).sum().item())
        total_samples += batch_size

    denom = max(total_samples, 1)
    return {"loss": total_loss / denom, "accuracy": total_correct / denom}


def predict(
    model: BertSpamClassifier,
    dataloader: DataLoader,
    frame: pd.DataFrame,
    *,
    tokenizer: Any,
    text_column: str,
    target_column: str,
    id_column: str,
    max_length: int,
    device: torch.device,
) -> tuple[list[str], list[str], list[int], list[int], list[float], np.ndarray | None, list[str], dict[str, float]]:
    """Run model inference over test set and return result bundles."""

    model.eval()
    ids = frame[id_column].astype(str).tolist()
    texts = frame[text_column].astype(str).tolist()
    y_true = frame[target_column].astype(int).tolist()
    y_pred: list[int] = []
    y_prob: list[float] = []
    attention_capture = None
    attention_tokens: list[str] = []

    start = time.perf_counter()
    iterator = tqdm(dataloader, desc="Predict", leave=False, dynamic_ncols=True)
    sample_offset = 0
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            probs = outputs.probabilities[:, 1]
            preds = torch.argmax(outputs.logits, dim=-1)
            y_prob.extend(probs.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
            if attention_capture is None:
                attention_capture, attention_tokens = _select_attention_view(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    attentions=outputs.attentions,
                    tokenizer=tokenizer,
                    batch_texts=texts[sample_offset : sample_offset + int(input_ids.shape[0])],
                    max_length=max_length,
                    min_tokens=2,
                )
            sample_offset += int(input_ids.shape[0])
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    runtime = {"response_time_ms": elapsed_ms / max(len(texts), 1)}
    return ids, texts, y_true, y_pred, y_prob, attention_capture, attention_tokens, runtime


def _unpack_predict_payload(payload: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(payload) == 8:
        return payload
    raise ValueError("predict() must return 8 values.")


def _extract_last_layer_attention(
    attentions: Any,
    *,
    sample_index: int,
    seq_len: int,
) -> np.ndarray | None:
    """Extract one sample attention map from last layer and average over heads."""

    if not attentions:
        return None
    layer = attentions[-1]
    if isinstance(layer, torch.Tensor):
        arr = layer.detach().float().cpu().numpy()
    else:
        arr = np.asarray(layer)
    if arr.ndim != 4 or arr.size == 0:
        return None
    if sample_index >= arr.shape[0]:
        return None
    sample = arr[sample_index]  # [num_heads, seq, seq]
    if sample.ndim != 3:
        return None
    mean_map = sample.mean(axis=0)
    if mean_map.ndim != 2:
        return None
    return mean_map[:seq_len, :seq_len]


def _normalize_token_piece(token: str) -> tuple[str | None, bool]:
    """Normalize one token piece and return (piece, is_continuation)."""

    if token in {"[PAD]", "[CLS]", "[SEP]", "[UNK]"}:
        return None, False

    piece = token.strip()
    continuation = False
    if piece.startswith("##"):
        continuation = True
        piece = piece[2:]
    elif piece.startswith("#"):
        continuation = True
        piece = piece.lstrip("#")

    piece = piece.lstrip("Ġ▁")
    piece = re.sub(r"^[^\w]+|[^\w]+$", "", piece)
    if not piece:
        return None, continuation
    if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", piece):
        return None, continuation
    return piece, continuation


def _aggregate_subwords_to_words(
    token_map: np.ndarray,
    raw_tokens: list[str],
) -> tuple[np.ndarray | None, list[str]]:
    """Merge subwords into whole words and aggregate token attention into word attention."""

    groups: list[list[int]] = []
    words: list[str] = []

    for idx, token in enumerate(raw_tokens):
        piece, continuation = _normalize_token_piece(token)
        if piece is None:
            continue
        if continuation and groups:
            groups[-1].append(idx)
            words[-1] = f"{words[-1]}{piece}"
        else:
            groups.append([idx])
            words.append(piece)

    if not groups:
        return None, []

    size = len(groups)
    word_map = np.zeros((size, size), dtype=float)
    for i, row_group in enumerate(groups):
        for j, col_group in enumerate(groups):
            block = token_map[np.ix_(row_group, col_group)]
            word_map[i, j] = float(block.mean()) if block.size else 0.0

    return _renormalize_rows(word_map), words


def _fallback_words_from_text(text: str) -> list[str]:
    """Extract displayable words directly from original text for UNK-heavy cases."""

    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\u4e00-\u9fff]+", text)
    return [word.strip() for word in words if word.strip()]


def _renormalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row to keep attention scale interpretable after filtering."""

    row_sums = matrix.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 1e-12
    normalized = matrix.copy()
    if np.any(nonzero):
        normalized[nonzero] = normalized[nonzero] / row_sums[nonzero]
    return normalized


def _select_attention_view(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    attentions: Any,
    tokenizer: Any,
    batch_texts: list[str],
    max_length: int,
    min_tokens: int = 2,
) -> tuple[np.ndarray | None, list[str]]:
    """Pick one sample in a batch that has enough meaningful tokens."""

    batch_size = int(input_ids.shape[0])
    best_map: np.ndarray | None = None
    best_tokens: list[str] = []
    for sample_idx in range(batch_size):
        valid_len = int(attention_mask[sample_idx].detach().sum().item())
        seq_len = max(1, min(max_length, valid_len))
        token_map = _extract_last_layer_attention(attentions, sample_index=sample_idx, seq_len=seq_len)
        if token_map is None:
            continue

        token_ids = input_ids[sample_idx].detach().cpu().tolist()[:seq_len]
        raw_tokens = (
            tokenizer.convert_ids_to_tokens(token_ids)
            if hasattr(tokenizer, "convert_ids_to_tokens")
            else []
        )
        if not raw_tokens:
            continue

        word_map, words = _aggregate_subwords_to_words(token_map, raw_tokens)
        if word_map is not None:
            if len(words) >= min_tokens:
                return word_map, words
            if len(words) > len(best_tokens):
                best_map = word_map
                best_tokens = words

        # Text-based fallback for UNK-heavy tokenization.
        raw_text = batch_texts[sample_idx] if sample_idx < len(batch_texts) else ""
        fallback_words = _fallback_words_from_text(raw_text)
        non_special_indices = [
            idx for idx, token in enumerate(raw_tokens) if token not in {"[PAD]", "[CLS]", "[SEP]"}
        ]
        usable = min(len(non_special_indices), len(fallback_words))
        if usable >= min_tokens:
            keep_indices = non_special_indices[:usable]
            fallback_map = token_map[np.ix_(keep_indices, keep_indices)]
            return _renormalize_rows(fallback_map), fallback_words[:usable]
    return best_map, best_tokens


def run_experiment(config: ExperimentConfig) -> dict[str, str]:
    """Run one BERT experiment and export metrics, charts, and predictions."""

    set_seed(config.random_seed)
    project_root, data_path, model_dir, result_dir = resolve_run_paths(
        project_root=config.project_root,
        data_path=config.data_path,
        model_dir=config.model_dir,
        result_dir=config.result_dir,
        timestamped_results=config.timestamped_results,
        result_time_format=config.result_time_format,
    )
    runtime_cfg = replace(config, project_root=project_root, data_path=data_path, model_dir=model_dir, result_dir=result_dir)

    variant = "pretrained" if runtime_cfg.use_pretrained_backbone else "zero_trained"
    variant_prefix = f"bert_{variant}"
    variant_result_dir = runtime_cfg.result_dir / variant
    variant_model_dir = runtime_cfg.model_dir / variant
    variant_result_dir.mkdir(parents=True, exist_ok=True)
    variant_model_dir.mkdir(parents=True, exist_ok=True)

    adapter = SpamDataAdapter(runtime_cfg)
    prepared = adapter.prepare_dataframe()
    splits = adapter.train_val_test_split(prepared)

    vocab_path = None
    if not runtime_cfg.use_pretrained_backbone:
        vocab_path = variant_model_dir / f"{variant_prefix}_vocab.txt"
        adapter.write_vocabulary(prepared, vocab_path)
    tokenizer = adapter.build_tokenizer(vocab_path=vocab_path)

    train_loader = build_dataloader(
        splits.train,
        tokenizer,
        text_column=runtime_cfg.text_column,
        target_column=runtime_cfg.target_column,
        batch_size=runtime_cfg.batch_size,
        max_length=runtime_cfg.max_length,
        shuffle=True,
    )
    val_loader = build_dataloader(
        splits.val,
        tokenizer,
        text_column=runtime_cfg.text_column,
        target_column=runtime_cfg.target_column,
        batch_size=runtime_cfg.batch_size,
        max_length=runtime_cfg.max_length,
        shuffle=False,
    )
    test_loader = build_dataloader(
        splits.test,
        tokenizer,
        text_column=runtime_cfg.text_column,
        target_column=runtime_cfg.target_column,
        batch_size=runtime_cfg.batch_size,
        max_length=runtime_cfg.max_length,
        shuffle=False,
    )

    device = torch.device("cpu" if runtime_cfg.force_cpu or not torch.cuda.is_available() else "cuda")
    vocab_size = int(getattr(tokenizer, "vocab_size", 30522))
    model = BertSpamClassifier(config=runtime_cfg, vocab_size=vocab_size).to(device)

    optimizer = AdamW(model.parameters(), lr=runtime_cfg.learning_rate, weight_decay=runtime_cfg.weight_decay)
    total_steps = max(len(train_loader) * max(runtime_cfg.epochs, 1), 1)
    warmup_steps = int(total_steps * runtime_cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history_rows: list[dict[str, float]] = []
    best_state = model.state_dict()
    best_val_loss = float("inf")
    patience_wait = 0

    print(
        f"[{variant_prefix}] Start training: epochs={runtime_cfg.epochs}, "
        f"batch_size={runtime_cfg.batch_size}, device={device.type}, "
        f"train_samples={len(splits.train)}, val_samples={len(splits.val)}, test_samples={len(splits.test)}"
    )
    for epoch in range(1, runtime_cfg.epochs + 1):
        train_stats = epoch_pass(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            train=True,
            show_progress=True,
            progress_desc=f"Epoch {epoch}/{runtime_cfg.epochs} Train",
        )
        val_stats = epoch_pass(
            model,
            val_loader,
            device=device,
            optimizer=None,
            scheduler=None,
            train=False,
            show_progress=True,
            progress_desc=f"Epoch {epoch}/{runtime_cfg.epochs} Val",
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_stats["loss"]),
                "val_loss": float(val_stats["loss"]),
                "train_accuracy": float(train_stats["accuracy"]),
                "val_accuracy": float(val_stats["accuracy"]),
            }
        )

        current_val_loss = float(val_stats["loss"])
        print(
            f"[{variant_prefix}] Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['accuracy']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f}"
        )
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_state = model.state_dict()
            patience_wait = 0
        else:
            patience_wait += 1
        if runtime_cfg.enable_early_stopping and patience_wait >= runtime_cfg.patience:
            print(
                f"[{variant_prefix}] Early stopping triggered at epoch {epoch} "
                f"(patience={runtime_cfg.patience}, best_val_loss={best_val_loss:.4f})."
            )
            break

    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    prediction_payload = _unpack_predict_payload(
        predict(
            model,
            test_loader,
            splits.test,
            tokenizer=tokenizer,
            text_column=runtime_cfg.text_column,
            target_column=runtime_cfg.target_column,
            id_column=runtime_cfg.id_column,
            max_length=runtime_cfg.max_length,
            device=device,
        )
    )
    ids, texts, y_true, y_pred, y_prob, attention_map, attention_tokens, runtime_info = prediction_payload

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    metrics.update({"response_time_ms": float(runtime_info.get("response_time_ms", 0.0))})
    metric_frame = build_metric_bar_frame(metrics)
    confusion_frame = build_confusion_dataframe(y_true, y_pred)
    prediction_frame = build_prediction_frame(ids, texts, y_true, y_pred, y_prob)
    report_frame = classification_report_frame(y_true, y_pred)

    split_manifest_path = adapter.export_split_manifest(splits, variant_result_dir / f"{variant_prefix}_split_manifest.json")
    split_manifest_payload: dict[str, Any]
    if isinstance(split_manifest_path, Path) and split_manifest_path.exists():
        split_manifest_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    else:
        split_manifest_payload = {
            "train": {"rows": len(splits.train)},
            "val": {"rows": len(splits.val)},
            "test": {"rows": len(splits.test)},
        }

    artifact_outputs = save_experiment_artifacts(
        result_dir=variant_result_dir,
        prefix=variant_prefix,
        history=history,
        metrics=metrics,
        report_frame=report_frame,
        prediction_frame=prediction_frame,
        confusion_frame=confusion_frame,
        config_payload=asdict(runtime_cfg),
        extra_json_payloads={"split_manifest": split_manifest_payload},
    )
    outputs = artifact_outputs if isinstance(artifact_outputs, dict) else {}

    curve_path = plot_training_curves(history, variant_result_dir / f"{variant_prefix}_training_curves.png")
    bars_path = plot_metric_bars(metric_frame, variant_result_dir / f"{variant_prefix}_metric_bars.png")
    cm_path = plot_confusion_matrix(confusion_frame, variant_result_dir / f"{variant_prefix}_confusion_matrix.png")
    outputs["training_curve_path"] = str(curve_path)
    outputs["metric_bar_path"] = str(bars_path)
    outputs["confusion_plot_path"] = str(cm_path)

    if attention_map is not None and len(attention_tokens) > 0:
        heatmap_title = (
            "DistilBERT的注意力权重图"
            if variant == "pretrained"
            else "紧凑型BERT的注意力权重图"
        )
        heatmap_path = plot_attention_heatmap(
            attention_tokens[: attention_map.shape[0]],
            attention_map,
            variant_result_dir / f"{variant_prefix}_attention_heatmap.png",
            title=heatmap_title,
        )
        outputs["attention_heatmap_path"] = str(heatmap_path)
    else:
        print(f"[{variant_prefix}] Attention heatmap skipped: no valid tokens after filtering.")

    model_path = variant_model_dir / f"{variant_prefix}_spam_classifier.pt"
    torch.save(model.state_dict(), model_path)
    outputs["model_path"] = str(model_path)
    print(f"[{variant_prefix}] Training finished. Metrics saved to: {outputs.get('metrics_path', 'N/A')}")
    if "metrics_path" not in outputs:
        fallback_metrics_path = variant_result_dir / f"{variant_prefix}_test_metrics.json"
        fallback_metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs["metrics_path"] = str(fallback_metrics_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERT spam experiment.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-path", default="data/sms_spam_collection.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--result-dir", default="results/bert")
    parser.add_argument("--result-time-format", default="%y_%m_%d_%H_%M_%S")
    parser.add_argument("--no-timestamped-results", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--enable-early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--pretrained-model-name", default="distilbert-base-uncased")
    parser.add_argument("--pretrained-local-files-only", action="store_true")
    parser.add_argument("--from-scratch-backbone", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def _args_to_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        project_root=Path(args.project_root),
        data_path=Path(args.data_path),
        model_dir=Path(args.model_dir),
        result_dir=Path(args.result_dir),
        result_time_format=args.result_time_format,
        timestamped_results=not args.no_timestamped_results,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        enable_early_stopping=args.enable_early_stopping,
        patience=args.patience,
        pretrained_model_name=args.pretrained_model_name,
        pretrained_local_files_only=args.pretrained_local_files_only,
        use_pretrained_backbone=not args.from_scratch_backbone,
        force_cpu=args.force_cpu,
    )


if __name__ == "__main__":
    payload = run_experiment(_args_to_config(parse_args()))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
