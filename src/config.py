"""Experiment configuration objects for BERT and data preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration used by BERT experiments and shared data adapters."""

    project_root: Path = Path(".")
    data_path: Path = Path("data/sms_spam_collection.csv")
    model_dir: Path = Path("models")
    result_dir: Path = Path("results/bert")
    result_time_format: str = "%y_%m_%d_%H_%M_%S"
    timestamped_results: bool = True

    target_column: str = "Prediction"
    text_column: str = "text"
    id_column: str = "Email No."
    target_aliases: tuple[str, ...] = ("Prediction", "target", "label", "v1")
    text_aliases: tuple[str, ...] = ("text", "message", "sms", "v2")
    id_aliases: tuple[str, ...] = ("Email No.", "id")

    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_seed: int = 42

    repeat_clip: int = 8
    max_reconstructed_tokens: int = 256

    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128
    dropout: float = 0.2
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 256

    patience: int = 3
    enable_early_stopping: bool = False
    use_class_weights: bool = True
    force_cpu: bool = False

    use_pretrained_backbone: bool = True
    pretrained_model_name: str = "distilbert-base-uncased"
    pretrained_local_files_only: bool = False

