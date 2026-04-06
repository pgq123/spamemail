"""可视化模块冒烟测试。"""

import sys
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.visualize import (
    plot_attention_heatmap,
    plot_confusion_matrix,
    plot_metric_bars,
    plot_three_model_metric_dashboard,
    plot_training_curves,
)


class VisualizationTests(unittest.TestCase):
    """验证绘图函数能正确生成非空图像文件。"""

    def setUp(self) -> None:
        # 为每个用例创建独立输出目录，避免文件名冲突。
        self.output_dir = Path.cwd() / f"tmp_visualize_{uuid4().hex}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        # 递归删除测试产物，防止污染工作区。
        for path in sorted(self.output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        self.output_dir.rmdir()

    def test_plot_generation(self) -> None:
        """给定合法输入时，四类图表都应成功写出文件。"""

        history = pd.DataFrame(
            {
                "epoch": [1, 2],
                "train_loss": [0.8, 0.5],
                "val_loss": [0.7, 0.55],
                "train_accuracy": [0.7, 0.82],
                "val_accuracy": [0.68, 0.79],
            }
        )
        metric_frame = pd.DataFrame({"metric": ["accuracy", "f1"], "value": [0.9, 0.88]})
        comparison_metric_frame = pd.DataFrame(
            {
                "model": ["mlp", "mlp", "bert_pretrained", "bert_pretrained", "bert_zero_trained", "bert_zero_trained"],
                "model_display": ["TF-IDF+MLP", "TF-IDF+MLP", "\u9884\u8bad\u7ec3 DistilBERT", "\u9884\u8bad\u7ec3 DistilBERT", "\u4ece\u96f6\u8bad\u7ec3\u7d27\u51d1\u578b BERT", "\u4ece\u96f6\u8bad\u7ec3\u7d27\u51d1\u578b BERT"],
                "metric": ["accuracy", "f1", "accuracy", "f1", "accuracy", "f1"],
                "value": [0.76, 0.76, 0.91, 0.91, 0.83, 0.83],
            }
        )
        summary_frame = pd.DataFrame(
            {
                "rank": [1, 2, 3],
                "model_display": ["\u9884\u8bad\u7ec3 DistilBERT", "\u4ece\u96f6\u8bad\u7ec3\u7d27\u51d1\u578b BERT", "TF-IDF+MLP"],
                "accuracy": [0.91, 0.83, 0.76],
                "recall": [0.91, 0.83, 0.76],
                "f1": [0.91, 0.83, 0.76],
                "roc_auc": [0.91, 0.83, 0.76],
                "response_time_ms": [3.2, 7.1, 1.4],
                "composite_score": [0.95, 0.64, 0.71],
            }
        )
        confusion = pd.DataFrame([[40, 2], [3, 15]], index=["Ham", "Spam"], columns=["Ham", "Spam"])
        attention = np.array([[1.0, 0.4], [0.2, 0.9]])

        curve_path = plot_training_curves(history, self.output_dir / "curves.png")
        bar_path = plot_metric_bars(metric_frame, self.output_dir / "bars.png")
        matrix_path = plot_confusion_matrix(confusion, self.output_dir / "cm.png")
        heatmap_path = plot_attention_heatmap(["free", "win"], attention, self.output_dir / "attn.png")
        dashboard_path = plot_three_model_metric_dashboard(
            comparison_metric_frame, summary_frame, self.output_dir / "dashboard.png"
        )

        for path in [curve_path, bar_path, matrix_path, heatmap_path, dashboard_path]:
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
