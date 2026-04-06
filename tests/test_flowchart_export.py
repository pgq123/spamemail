"""Flowchart export smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowchart_export import export_system_flowchart


class FlowchartExportTests(unittest.TestCase):
    """Verify flowchart exporter writes high-resolution artifacts."""

    def setUp(self) -> None:
        self.output_dir = Path.cwd() / f"tmp_flowchart_{uuid4().hex}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for path in sorted(self.output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        self.output_dir.rmdir()

    def test_export_png_and_pdf(self) -> None:
        outputs = export_system_flowchart(self.output_dir)
        png_path = Path(outputs["png_path"])
        pdf_path = Path(outputs["pdf_path"])

        self.assertTrue(png_path.exists())
        self.assertTrue(pdf_path.exists())
        self.assertGreater(png_path.stat().st_size, 0)
        self.assertGreater(pdf_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
