"""统一训练入口（train.py）回归测试。"""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import run_unified_experiment


class UnifiedTrainEntrypointTests(unittest.TestCase):
    """验证一键训练入口能正确串联三模型并生成比较产物。"""

    def _make_temp_root(self) -> Path:
        root = Path.cwd() / f"tmp_unified_train_{uuid4().hex}"
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

    def _write_metric_file(self, path: Path, score: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "accuracy": score,
                    "precision": score,
                    "recall": score,
                    "f1": score,
                    "roc_auc": score,
                }
            ),
            encoding="utf-8",
        )

    def test_default_all_scope_runs_three_models_and_writes_comparison(self) -> None:
        """默认 all 模式应生成三模型对比表与两张核心对比图。"""

        root = self._make_temp_root()
        try:
            bert_pretrained_metrics = root / "fake_outputs" / "bert_pretrained_metrics.json"
            bert_zero_metrics = root / "fake_outputs" / "bert_zero_metrics.json"
            mlp_metrics = root / "fake_outputs" / "mlp_metrics.json"
            self._write_metric_file(bert_pretrained_metrics, 0.91)
            self._write_metric_file(bert_zero_metrics, 0.83)
            self._write_metric_file(mlp_metrics, 0.76)

            args = argparse.Namespace(
                project_root=str(root),
                data_path="data/sms_spam_collection.csv",
                model_dir="models",
                bert_result_dir="results/bert",
                mlp_result_dir="results/mlp",
                comparison_result_dir="results/comparison",
                result_time_format="%d-%H-%M",
                no_timestamped_results=True,
                run_scope="all",
                bert_modes="both",
                bert_epochs=1,
                bert_batch_size=2,
                bert_max_length=16,
                bert_learning_rate=3e-5,
                bert_pretrained_model_name="distilbert-base-uncased",
                bert_pretrained_local_files_only=False,
                bert_enable_early_stopping=False,
                force_cpu=True,
                mlp_learning_rate=1e-3,
                mlp_batch_size=8,
                mlp_max_iter=1,
                mlp_alpha=1e-4,
            )

            with (
                patch(
                    "train.run_experiment",
                    side_effect=[
                        {"metrics_path": str(bert_pretrained_metrics)},
                        {"metrics_path": str(bert_zero_metrics)},
                    ],
                ) as bert_run_mock,
                patch("train.run_mlp_baseline", return_value={"metrics_path": str(mlp_metrics)}) as mlp_run_mock,
            ):
                payload = run_unified_experiment(args)

            self.assertEqual(bert_run_mock.call_count, 2)
            self.assertEqual(mlp_run_mock.call_count, 1)

            first_bert_cfg = bert_run_mock.call_args_list[0].args[0]
            second_bert_cfg = bert_run_mock.call_args_list[1].args[0]
            self.assertTrue(first_bert_cfg.use_pretrained_backbone)
            self.assertFalse(second_bert_cfg.use_pretrained_backbone)

            comparison_csv = Path(payload["comparison"]["comparison_csv_path"])
            method_csv = Path(payload["comparison"]["method_csv_path"])
            stats_csv = Path(payload["comparison"]["stats_csv_path"])
            cross_model_plot = Path(payload["comparison"]["cross_model_plot_path"])
            dashboard_plot = Path(payload["comparison"]["dashboard_plot_path"])
            chapter_bridge_plot = Path(payload["comparison"]["chapter_bridge_plot_path"])
            self.assertTrue(comparison_csv.exists())
            self.assertTrue(method_csv.exists())
            self.assertTrue(stats_csv.exists())
            self.assertTrue(cross_model_plot.exists())
            self.assertTrue(dashboard_plot.exists())
            self.assertTrue(chapter_bridge_plot.exists())

            comparison_frame = pd.read_csv(comparison_csv)
            self.assertEqual(len(comparison_frame), 3)
            self.assertSetEqual(
                set(comparison_frame["model"].tolist()),
                {"bert_pretrained", "bert_zero_trained", "mlp"},
            )
            self.assertListEqual(
                comparison_frame["model"].tolist(),
                ["bert_pretrained", "bert_zero_trained", "mlp"],
            )
            self.assertListEqual(comparison_frame["rank"].tolist(), [1, 2, 3])
            self.assertListEqual(comparison_frame["is_best"].tolist(), [True, False, False])
            self.assertIn("response_time_ms", comparison_frame.columns)
            self.assertIn("composite_score", comparison_frame.columns)
        finally:
            self._cleanup_path(root)


if __name__ == "__main__":
    unittest.main()
