"""
工具函數
提供通用的工具函數
"""

import os
import shutil
from typing import List, Dict, Any
from pathlib import Path


def ensure_directory(path: str):
    """
    確保目錄存在，如果不存在則創建

    Args:
        path: 目錄路徑
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def cleanup_temp_directories(temp_paths: List[str]):
    """
    清理臨時目錄

    Args:
        temp_paths: 臨時目錄路徑列表
    """
    for path in temp_paths:
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"清理臨時目錄失敗 {path}: {e}")


def format_duration(seconds: float) -> str:
    """
    格式化持續時間

    Args:
        seconds: 秒數

    Returns:
        格式化的時間字串
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def safe_float_convert(value: Any, default: float = 0.0) -> float:
    """
    安全地轉換為浮點數

    Args:
        value: 要轉換的值
        default: 預設值

    Returns:
        浮點數值
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int_convert(value: Any, default: int = 0) -> int:
    """
    安全地轉換為整數

    Args:
        value: 要轉換的值
        default: 預設值

    Returns:
        整數值
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    驗證配置檔案

    Args:
        config: 配置字典

    Returns:
        錯誤訊息列表
    """
    errors = []

    # 檢查必要欄位
    required_fields = ['parameters', 'strategy']
    for field in required_fields:
        if field not in config:
            errors.append(f"缺少必要欄位: {field}")

    # 檢查參數配置
    if 'parameters' in config:
        parameters = config['parameters']
        if not isinstance(parameters, dict):
            errors.append("parameters 必須是字典")
        else:
            for param_name, param_config in parameters.items():
                if not isinstance(param_config, (list, dict)):
                    errors.append(f"參數 {param_name} 的配置格式無效")

    # 檢查策略配置
    if 'strategy' in config:
        strategy = config['strategy']
        if not isinstance(strategy, dict):
            errors.append("strategy 必須是字典")
        else:
            valid_strategies = ['grid_search', 'random_search', 'hierarchical']
            if 'type' in strategy and strategy['type'] not in valid_strategies:
                errors.append(f"無效的策略類型: {strategy['type']}")

    return errors


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合併配置字典

    Args:
        base_config: 基礎配置
        override_config: 覆蓋配置

    Returns:
        合併後的配置
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
