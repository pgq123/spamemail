"""数据适配与预处理工具，统一不同垃圾短信数据格式。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertTokenizerFast

from config import ExperimentConfig
from runtime_utils import first_existing, normalize_binary_label

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


@dataclass
class DatasetSplits:
    """训练/验证/测试划分结果容器。"""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class SpamDataAdapter:
    """短信数据适配器，负责列名标准化、文本重建、划分与词表导出。"""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def load_dataframe(self) -> pd.DataFrame:
        """从配置路径读取原始 CSV 数据。"""

        df = pd.read_csv(self.config.data_path)
        return df

    def prepare_dataframe(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        标准化数据列并确保生成可训练的 text/target 字段。

        若缺少 text 列，则会从词频列重建伪文本，保证下游流程可运行。
        """

        source_df = self.load_dataframe() if df is None else df.copy()
        source_df = self.normalize_standard_columns(source_df)
        if self.config.text_column not in source_df.columns:
            feature_columns = self.infer_feature_columns(source_df)
            source_df[self.config.text_column] = source_df.apply(lambda row: self.reconstruct_text_from_row(row, feature_columns), axis=1)
        source_df[self.config.text_column] = source_df[self.config.text_column].fillna("empty").astype(str)
        source_df[self.config.target_column] = source_df[self.config.target_column].astype(int)
        return source_df

    def normalize_standard_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将外部字段名映射为项目标准字段名。

        必需：目标列（Prediction 别名）至少存在一个。
        可选：text/id 可从别名推断，不存在时会自动补齐 id。
        """

        target_column = self.pick_first_existing(df, self.config.target_aliases)
        if target_column is None:
            raise KeyError(f"Missing target column. Expected one of: {self.config.target_aliases}")

        rename_map = self.build_rename_map(
            df=df,
            alias_to_target={
                self.config.target_aliases: self.config.target_column,
                self.config.text_aliases: self.config.text_column,
                self.config.id_aliases: self.config.id_column,
            },
        )
        normalized = df.rename(columns=rename_map)
        if self.config.target_column in normalized.columns:
            normalized[self.config.target_column] = normalized[self.config.target_column].map(self.normalize_label)
        if self.config.id_column not in normalized.columns:
            # Ensure every sample has a stable identifier for result export.
            normalized[self.config.id_column] = [f"sample_{idx}" for idx in range(len(normalized))]
        return normalized

    def build_rename_map(
        self,
        df: pd.DataFrame,
        alias_to_target: dict[tuple[str, ...], str],
    ) -> dict[str, str]:
        """根据别名组生成重命名字典。"""

        rename_map: dict[str, str] = {}
        for aliases, target in alias_to_target.items():
            existing = self.pick_first_existing(df, aliases)
            if existing is not None and existing != target:
                rename_map[existing] = target
        return rename_map

    def pick_first_existing(self, df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
        """在候选列中返回 DataFrame 内第一个存在的列名。"""

        return first_existing(candidates, df.columns)

    def normalize_label(self, value):
        """将多种标签编码归一化为 0/1。"""

        return normalize_binary_label(value)

    def infer_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """推断词频特征列（排除 id/target/text）。"""

        ignored = {self.config.id_column, self.config.target_column, self.config.text_column}
        return [column for column in df.columns if column not in ignored]

    def reconstruct_text_from_row(self, row: pd.Series, feature_columns: Sequence[str]) -> str:
        """
        从词频列重建文本。

        算法说明：
        - 按 token 频次降序扩展词序列
        - 单 token 重复次数上限为 repeat_clip
        - 总 token 数上限为 max_reconstructed_tokens
        """

        tokens: List[str] = []
        counts = self.extract_positive_token_counts(row, feature_columns)
        counts.sort(key=lambda item: item[1], reverse=True)
        for token, count in counts:
            tokens.extend([token] * min(count, self.config.repeat_clip))
            if len(tokens) >= self.config.max_reconstructed_tokens:
                break
        if not tokens:
            tokens = ["empty"]
        return " ".join(tokens[: self.config.max_reconstructed_tokens])

    def extract_positive_token_counts(self, row: pd.Series, feature_columns: Sequence[str]) -> List[tuple[str, int]]:
        # 提前过滤非法词频，保证重建阶段仅处理有效 (token, count) 对。
        counts: List[tuple[str, int]] = []
        for token in feature_columns:
            value = row.get(token, 0)
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count > 0:
                counts.append((token, count))
        return counts

    def train_val_test_split(self, df: pd.DataFrame) -> DatasetSplits:
        """按配置比例进行分层抽样划分（train/val/test）。"""

        train_df, temp_df = train_test_split(
            df,
            train_size=self.config.train_size,
            stratify=df[self.config.target_column],
            random_state=self.config.random_seed,
        )
        relative_val_size = self.config.val_size / (self.config.val_size + self.config.test_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=relative_val_size,
            stratify=temp_df[self.config.target_column],
            random_state=self.config.random_seed,
        )
        return DatasetSplits(
            train=train_df.reset_index(drop=True),
            val=val_df.reset_index(drop=True),
            test=test_df.reset_index(drop=True),
        )

    def write_vocabulary(self, df: pd.DataFrame, vocab_path: Path) -> Path:
        """导出 tokenizer 词表文件（特殊符号 + 数据集词元）。"""

        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        tokens = self.collect_vocabulary_tokens(df)
        with vocab_path.open("w", encoding="utf-8") as handle:
            for token in SPECIAL_TOKENS + sorted(tokens):
                handle.write(f"{token}\n")
        return vocab_path

    def collect_vocabulary_tokens(self, df: pd.DataFrame) -> List[str]:
        """
        收集词表 token。

        优先从文本列抽取；若无文本列则回退到词频列名。
        """

        normalized = self.normalize_standard_columns(df.copy())
        if self.config.text_column in normalized.columns:
            text_series = normalized[self.config.text_column].fillna("").astype(str)
            tokens = self.extract_text_vocabulary(text_series.tolist())
            if tokens:
                return tokens
        return self.infer_feature_columns(normalized)

    def extract_text_vocabulary(self, texts: Sequence[str]) -> List[str]:
        """从自由文本中提取去重后的小写 token 列表。"""

        unique_tokens: set[str] = set()
        for text in texts:
            for token in re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower()):
                unique_tokens.add(token)
        return sorted(unique_tokens)

    def build_tokenizer(self, vocab_path: Path | None = None):
        """根据配置构建 tokenizer（预训练或本地词表）。"""

        if self.config.use_pretrained_backbone:
            return AutoTokenizer.from_pretrained(
                self.config.pretrained_model_name,
                use_fast=True,
                local_files_only=self.config.pretrained_local_files_only,
            )

        if vocab_path is None:
            raise ValueError("vocab_path is required when use_pretrained_backbone=False")
        return BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)

    def export_split_manifest(self, splits: DatasetSplits, output_path: Path) -> Path:
        """导出数据划分清单（样本数与类别占比），便于实验追踪。"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: {
                "rows": len(frame),
                "class_balance": frame[self.config.target_column].value_counts(normalize=True).to_dict(),
            }
            for name, frame in {"train": splits.train, "val": splits.val, "test": splits.test}.items()
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path


def get_text_label_pairs(
    frame: pd.DataFrame,
    text_column: str,
    target_column: str,
) -> Tuple[Sequence[str], Sequence[int]]:
    """从划分数据中提取文本序列与整数标签序列。"""

    return frame[text_column].tolist(), frame[target_column].astype(int).tolist()
