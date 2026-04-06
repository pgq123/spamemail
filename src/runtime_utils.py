"""训练运行期公共工具函数。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def resolve_run_paths(
    *,
    project_root: Path,
    data_path: Path,
    model_dir: Path,
    result_dir: Path,
    timestamped_results: bool,
    result_time_format: str,
) -> tuple[Path, Path, Path, Path]:
    """
    解析项目路径并按需附加时间戳目录。

    返回值顺序：
    1. project_root 绝对路径
    2. data_path 绝对路径
    3. model_dir 绝对路径
    4. result_dir 绝对路径（可含时间戳）
    """

    root = project_root.resolve()
    resolved_data_path = (root / data_path).resolve()
    resolved_model_dir = (root / model_dir).resolve()
    base_result_dir = (root / result_dir).resolve()
    resolved_result_dir = (
        (base_result_dir / datetime.now().strftime(result_time_format)).resolve()
        if timestamped_results
        else base_result_dir
    )
    return root, resolved_data_path, resolved_model_dir, resolved_result_dir


def serialize_path_values(data: dict[str, Any]) -> dict[str, Any]:
    """将字典中的 Path 转为字符串，便于 JSON 序列化。"""

    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


def normalize_binary_label(value: Any) -> int:
    """将常见 spam/ham 标签统一编码为 0/1。"""

    if pd.isna(value):
        raise ValueError("Encountered missing label value.")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"ham", "non-spam", "not spam"}:
            return 0
        if lowered in {"spam", "junk"}:
            return 1
    return int(value)


def first_existing(candidates: Iterable[str], available: Iterable[str]) -> str | None:
    """在候选列名中返回第一个实际存在的列名。"""

    available_set = set(available)
    for candidate in candidates:
        if candidate in available_set:
            return candidate
    return None
