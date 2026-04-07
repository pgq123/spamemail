"""Core code snapshot export smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from code_snapshot_export import export_core_code_snapshots


class CodeSnapshotExportTests(unittest.TestCase):
    """Verify code snapshot exporter emits non-empty PNG files."""

    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.output_dir = Path.cwd() / f"tmp_code_snapshot_{uuid4().hex}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for path in sorted(self.output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        self.output_dir.rmdir()

    def test_export_code_snapshots(self) -> None:
        outputs = export_core_code_snapshots(self.output_dir, project_root=self.repo_root)
        keys = [
            "distilbert_model_core_path",
            "mlp_pipeline_core_path",
            "bert_training_loop_core_path",
        ]
        for key in keys:
            artifact = Path(outputs[key])
            self.assertTrue(artifact.exists())
            self.assertGreater(artifact.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
