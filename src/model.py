"""BERT spam-classifier model wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, BertConfig, BertModel

from config import ExperimentConfig


@dataclass
class ModelOutput:
    """Unified model forward output."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    loss: torch.Tensor | None
    attentions: tuple[torch.Tensor, ...] | None


class BertSpamClassifier(nn.Module):
    """Binary spam classifier with configurable BERT backbone."""

    def __init__(self, config: ExperimentConfig, vocab_size: int | None = None) -> None:
        super().__init__()
        self.config = config

        if config.use_pretrained_backbone:
            self.backbone = self._load_pretrained_backbone(config)
            hidden_size = int(getattr(self.backbone.config, "hidden_size", 768))
        else:
            bert_cfg = BertConfig(
                vocab_size=int(vocab_size or 30522),
                max_position_embeddings=max(512, config.max_length + 2),
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                hidden_dropout_prob=config.dropout,
                attention_probs_dropout_prob=config.dropout,
                output_attentions=True,
            )
            # Force eager attention implementation so attention weights are always returned.
            setattr(bert_cfg, "_attn_implementation", "eager")
            self.backbone = BertModel(bert_cfg)
            hidden_size = config.hidden_size

        if hasattr(self.backbone.config, "_attn_implementation"):
            self.backbone.config._attn_implementation = "eager"

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def _load_pretrained_backbone(self, config: ExperimentConfig):
        kwargs = {
            "local_files_only": config.pretrained_local_files_only,
            "output_attentions": True,
        }
        try:
            return AutoModel.from_pretrained(
                config.pretrained_model_name,
                attn_implementation="eager",
                **kwargs,
            )
        except TypeError:
            return AutoModel.from_pretrained(config.pretrained_model_name, **kwargs)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        pooled = backbone_outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        probs = torch.softmax(logits, dim=-1)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return ModelOutput(
            logits=logits,
            probabilities=probs,
            loss=loss,
            attentions=getattr(backbone_outputs, "attentions", None),
        )
