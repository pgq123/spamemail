"""训练行为回归测试。"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bert_train import run_experiment
from config import ExperimentConfig
from mlp_train import MLPExperimentConfig, run_mlp_baseline


class _FakeBertModel:
    """用于隔离训练循环行为的最小 BERT 假模型。"""

    def __init__(self, *args, **kwargs) -> None:
        self._state = {"weight": torch.tensor([1.0])}
        self._param = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def to(self, device):
        return self

    def parameters(self):
        return [self._param]

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state


class TrainingBehaviorTests(unittest.TestCase):
    """验证训练轮次控制、早停逻辑与图表标题规范。"""

    def _build_split_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Email No.": ["id_1", "id_2"],
                "text": ["hello world", "win prize now"],
                "Prediction": [0, 1],
            }
        )

    def _patch_bert_dependencies(self, epoch_pass_mock):
        split_frame = self._build_split_frame()
        splits = SimpleNamespace(train=split_frame, val=split_frame, test=split_frame)
        adapter = MagicMock()
        adapter.prepare_dataframe.return_value = split_frame
        adapter.train_val_test_split.return_value = splits
        adapter.write_vocabulary.return_value = Path("bert_vocab.txt")
        adapter.build_tokenizer.return_value = SimpleNamespace(vocab_size=32)

        return patch.multiple(
            "bert_train",
            set_seed=MagicMock(),
            SpamDataAdapter=MagicMock(return_value=adapter),
            build_dataloader=MagicMock(return_value=[{"sample": 1}]),
            BertSpamClassifier=_FakeBertModel,
            AdamW=MagicMock(return_value=MagicMock()),
            get_linear_schedule_with_warmup=MagicMock(return_value=MagicMock()),
            epoch_pass=epoch_pass_mock,
            predict=MagicMock(return_value=(["id_1"], ["hello"], [0], [0], [0.1], None, [], {"response_time_ms": 1.2})),
            compute_classification_metrics=MagicMock(
                return_value={"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0}
            ),
            build_metric_bar_frame=MagicMock(return_value=pd.DataFrame({"metric": ["accuracy"], "value": [1.0]})),
            build_confusion_dataframe=MagicMock(return_value=pd.DataFrame([[1, 0], [0, 1]])),
            build_prediction_frame=MagicMock(return_value=pd.DataFrame({"id": ["id_1"]})),
            classification_report_frame=MagicMock(return_value=pd.DataFrame({"precision": [1.0]}, index=["0"])),
            save_experiment_artifacts=MagicMock(),
            plot_training_curves=MagicMock(),
            plot_metric_bars=MagicMock(),
            plot_confusion_matrix=MagicMock(),
            plot_attention_heatmap=MagicMock(),
        )

    def _make_temp_root(self) -> Path:
        root = Path.cwd() / f"tmp_training_behaviors_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _cleanup_path(self, root: Path) -> None:
        if not root.exists():
            return
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        root.rmdir()

    def test_bert_runs_full_epochs_when_early_stopping_disabled(self) -> None:
        """关闭早停时，BERT 应完整运行到配置的最大轮次。"""

        root = self._make_temp_root()
        try:
            cfg = ExperimentConfig(
                project_root=root,
                data_path=Path("spam.csv"),
                result_dir=Path("results/bert"),
                model_dir=Path("models"),
                epochs=40,
                enable_early_stopping=False,
                timestamped_results=False,
                use_class_weights=False,
            )

            epoch_pass_mock = MagicMock(
                side_effect=[
                    {"loss": 1.0, "accuracy": 0.5} if i % 2 == 0 else {"loss": 1.1, "accuracy": 0.5}
                    for i in range(cfg.epochs * 2)
                ]
            )

            with self._patch_bert_dependencies(epoch_pass_mock):
                run_experiment(cfg)

            self.assertEqual(epoch_pass_mock.call_count, cfg.epochs * 2)
        finally:
            self._cleanup_path(root)

    def test_bert_early_stopping_triggers_when_enabled(self) -> None:
        """开启早停后，当验证损失连续恶化应提前停止。"""

        root = self._make_temp_root()
        try:
            cfg = ExperimentConfig(
                project_root=root,
                data_path=Path("spam.csv"),
                result_dir=Path("results/bert"),
                model_dir=Path("models"),
                epochs=40,
                patience=2,
                enable_early_stopping=True,
                timestamped_results=False,
                use_class_weights=False,
            )

            metrics_sequence = [
                {"loss": 0.50, "accuracy": 0.5},  # epoch 1 train
                {"loss": 0.40, "accuracy": 0.5},  # epoch 1 val(best)
                {"loss": 0.50, "accuracy": 0.5},  # epoch 2 train
                {"loss": 0.41, "accuracy": 0.5},  # epoch 2 val(wait=1)
                {"loss": 0.50, "accuracy": 0.5},  # epoch 3 train
                {"loss": 0.42, "accuracy": 0.5},  # epoch 3 val(wait=2=>stop)
            ]
            epoch_pass_mock = MagicMock(side_effect=metrics_sequence)

            with self._patch_bert_dependencies(epoch_pass_mock):
                run_experiment(cfg)

            self.assertEqual(epoch_pass_mock.call_count, 6)
        finally:
            self._cleanup_path(root)

    def test_mlp_metric_plot_uses_standardized_chinese_title(self) -> None:
        """MLP 指标图标题应使用标准模型名 + 中文图题格式。"""

        root = self._make_temp_root()
        try:
            data_path = root / "sms_spam_collection.csv"
            rows = []
            for index in range(20):
                label = "spam" if index % 2 else "ham"
                text = "win money now" if label == "spam" else "meeting at noon"
                rows.append({"v1": label, "v2": text, "id": f"row_{index}"})
            pd.DataFrame(rows).to_csv(data_path, index=False)

            cfg = MLPExperimentConfig(
                project_root=root,
                data_path=Path("sms_spam_collection.csv"),
                result_dir=Path("results/mlp"),
                model_dir=Path("models"),
                max_iter=1,
                batch_size=4,
                timestamped_results=False,
            )

            with patch("mlp_train.plot_metric_bars") as metric_plot_mock:
                run_mlp_baseline(cfg)

            self.assertTrue(metric_plot_mock.called)
            self.assertEqual(
                metric_plot_mock.call_args.kwargs.get("title"),
                "TF-IDF+MLP \u6d4b\u8bd5\u96c6\u6307\u6807\u56fe",
            )
        finally:
            self._cleanup_path(root)


if __name__ == "__main__":
    unittest.main()
