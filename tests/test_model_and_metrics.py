"""模型前向与指标工具的单元测试。"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ExperimentConfig
from metrics import average_attention_map, compute_classification_metrics
from model import BertSpamClassifier


class ModelAndMetricTests(unittest.TestCase):
    """验证模型输出形状与核心指标计算逻辑。"""

    def test_model_forward_shape(self) -> None:
        """前向传播后应返回正确维度的 logits/probabilities 与可选 loss。"""

        config = ExperimentConfig(
            max_length=16,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=128,
            use_pretrained_backbone=False,
        )
        model = BertSpamClassifier(config=config, vocab_size=32)
        input_ids = torch.randint(0, 32, (2, 16))
        attention_mask = torch.ones((2, 16), dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.assertEqual(outputs.logits.shape, (2, 2))
        self.assertEqual(outputs.probabilities.shape, (2, 2))
        self.assertIsNotNone(outputs.loss)
        self.assertIsNotNone(outputs.attentions)

    def test_metrics_are_computed(self) -> None:
        """指标函数应输出数值合理的 accuracy/precision/roc_auc 等字段。"""

        metrics = compute_classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertGreater(metrics["roc_auc"], 0.5)

    def test_average_attention_map_handles_missing_attention(self) -> None:
        """注意力为空时应安全返回 None，避免可视化阶段报错。"""

        self.assertIsNone(average_attention_map(None, seq_len=8))
        self.assertIsNone(average_attention_map(tuple(), seq_len=8))


if __name__ == "__main__":
    unittest.main()
