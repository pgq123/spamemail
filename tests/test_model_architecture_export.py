"""Model architecture figure export smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_architecture_export import export_model_architecture_figures


class ModelArchitectureExportTests(unittest.TestCase):
    """Verify MLP and DistilBERT architecture figures can be exported."""

    def setUp(self) -> None:
        self.output_dir = Path.cwd() / f"tmp_model_arch_{uuid4().hex}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for path in sorted(self.output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        self.output_dir.rmdir()

    def test_export_architecture_png_and_pdf(self) -> None:
        outputs = export_model_architecture_figures(self.output_dir)
        keys = [
            "mlp_png_path",
            "mlp_pdf_path",
            "distilbert_png_path",
            "distilbert_pdf_path",
        ]
        for key in keys:
            artifact = Path(outputs[key])
            self.assertTrue(artifact.exists())
            self.assertGreater(artifact.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
