"""BERT-based spam classification research pipeline."""

from importlib import import_module

__all__ = ["ExperimentConfig", "BertSpamClassifier", "SpamDataAdapter"]


def __getattr__(name: str):
    """Lazily expose selected public APIs to keep import overhead small."""

    if name == "ExperimentConfig":
        return import_module(f"{__name__}.config").ExperimentConfig
    if name == "SpamDataAdapter":
        return import_module(f"{__name__}.preprocess").SpamDataAdapter
    if name == "BertSpamClassifier":
        return import_module(f"{__name__}.model").BertSpamClassifier
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
