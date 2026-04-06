"""预处理与数据适配行为测试。"""

import sys
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ExperimentConfig
from preprocess import SpamDataAdapter


class PreprocessTests(unittest.TestCase):
    """验证数据列标准化、文本重建与词表导出逻辑。"""

    def setUp(self) -> None:
        # 为每个用例创建隔离目录，避免文件副作用相互影响。
        self.temp_dir = Path.cwd() / f"tmp_preprocess_{uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.config = ExperimentConfig(project_root=self.temp_dir, data_path=Path("spam.csv"))
        self.adapter = SpamDataAdapter(self.config)

    def tearDown(self) -> None:
        # 测试结束后递归清理临时目录，保证仓库整洁。
        for path in sorted(self.temp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        self.temp_dir.rmdir()

    def test_prepare_dataframe_reconstructs_text(self) -> None:
        """当缺少原始文本列时，应能根据词频列重建 text 字段。"""

        frame = pd.DataFrame(
            [
                {"Email No.": "Email 1", "free": 2, "win": 1, "hello": 0, "Prediction": 1},
                {"Email No.": "Email 2", "free": 0, "win": 0, "hello": 3, "Prediction": 0},
            ]
        )
        prepared = self.adapter.prepare_dataframe(frame)
        self.assertIn("text", prepared.columns)
        self.assertEqual(prepared.loc[0, "text"], "free free win")
        self.assertEqual(prepared.loc[1, "text"], "hello hello hello")

    def test_prepare_dataframe_prefers_raw_sms_columns(self) -> None:
        """若输入为 v1/v2 原始短信结构，应优先直接使用原文本，不重建伪文本。"""

        frame = pd.DataFrame(
            [
                {"v1": "ham", "v2": "Hello there", "id": "row_1"},
                {"v1": "spam", "v2": "Free entry in 2 a wkly comp", "id": "row_2"},
            ]
        )
        prepared = self.adapter.prepare_dataframe(frame)
        self.assertEqual(prepared.loc[0, "text"], "Hello there")
        self.assertEqual(prepared.loc[1, "text"], "Free entry in 2 a wkly comp")
        self.assertEqual(prepared.loc[0, "Prediction"], 0)
        self.assertEqual(prepared.loc[1, "Prediction"], 1)
        self.assertEqual(prepared.loc[0, "Email No."], "row_1")

    def test_write_vocabulary_creates_vocab_file(self) -> None:
        """导出词表时应包含特殊 token 与特征 token。"""

        frame = pd.DataFrame([{"Email No.": "Email 1", "free": 2, "win": 1, "Prediction": 1}])
        vocab_path = self.temp_dir / "vocab.txt"
        self.adapter.write_vocabulary(frame, vocab_path)
        content = vocab_path.read_text(encoding="utf-8")
        self.assertIn("[CLS]", content)
        self.assertIn("free", content)

    def test_write_vocabulary_uses_text_tokens_when_available(self) -> None:
        """存在文本列时，词表应从真实文本抽取 token。"""

        frame = pd.DataFrame(
            [
                {"v1": "ham", "v2": "Hello friend", "id": "row_1"},
                {"v1": "spam", "v2": "Win cash now", "id": "row_2"},
            ]
        )
        vocab_path = self.temp_dir / "text_vocab.txt"
        self.adapter.write_vocabulary(frame, vocab_path)
        content = vocab_path.read_text(encoding="utf-8")
        self.assertIn("hello", content)
        self.assertIn("win", content)


if __name__ == "__main__":
    unittest.main()
