"""
參數組合生成器
負責根據配置生成測試參數的組合
"""

import itertools
import random
from typing import Dict, List, Any, Union
import yaml


class ParameterGenerator:
    """參數組合生成器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化參數生成器

        Args:
            config: 測試配置字典
        """
        self.config = config
        self.parameters = config.get('parameters', {})
        self.strategy = config.get('strategy', {})

    def generate_combinations(self) -> List[Dict[str, Any]]:
        """
        生成參數組合列表

        Returns:
            參數組合列表
        """
        strategy_type = self.strategy.get('type', 'grid_search')

        if strategy_type == 'grid_search':
            return self._generate_grid_search()
        elif strategy_type == 'random_search':
            return self._generate_random_search()
        elif strategy_type == 'hierarchical':
            return self._generate_hierarchical()
        else:
            raise ValueError(f"不支持的測試策略: {strategy_type}")

    def _generate_grid_search(self) -> List[Dict[str, Any]]:
        """生成網格搜索的參數組合"""
        param_values = {}

        for param_name, param_config in self.parameters.items():
            param_values[param_name] = self._parse_parameter_config(param_config)

        # 生成所有組合
        param_names = list(param_values.keys())
        param_value_lists = [param_values[name] for name in param_names]

        combinations = []
        for values in itertools.product(*param_value_lists):
            combo = dict(zip(param_names, values))
            combinations.append(combo)

        # 限制組合數量
        max_combinations = self.strategy.get('max_combinations', len(combinations))
        if len(combinations) > max_combinations:
            random.seed(self.strategy.get('random_seed', 42))
            combinations = random.sample(combinations, max_combinations)

        # 添加重複測試
        repetitions = self.strategy.get('repetitions', 1)
        if repetitions > 1:
            repeated_combinations = []
            for combo in combinations:
                for i in range(repetitions):
                    repeated_combo = combo.copy()
                    repeated_combo['_repetition'] = i + 1
                    repeated_combinations.append(repeated_combo)
            combinations = repeated_combinations

        return combinations

    def _generate_random_search(self) -> List[Dict[str, Any]]:
        """生成隨機搜索的參數組合"""
        max_combinations = self.strategy.get('max_combinations', 50)
        random.seed(self.strategy.get('random_seed', 42))

        combinations = []
        for _ in range(max_combinations):
            combo = {}
            for param_name, param_config in self.parameters.items():
                values = self._parse_parameter_config(param_config)
                combo[param_name] = random.choice(values)
            combinations.append(combo)

        # 添加重複測試
        repetitions = self.strategy.get('repetitions', 1)
        if repetitions > 1:
            repeated_combinations = []
            for combo in combinations:
                for i in range(repetitions):
                    repeated_combo = combo.copy()
                    repeated_combo['_repetition'] = i + 1
                    repeated_combinations.append(repeated_combo)
            combinations = repeated_combinations

        return combinations

    def _generate_hierarchical(self) -> List[Dict[str, Any]]:
        """生成分層測試的參數組合"""
        # 簡化實現：先測試單一參數，然後組合
        combinations = []

        # 第一層：單一參數測試（使用中間值）
        base_values = {}
        for param_name, param_config in self.parameters.items():
            values = self._parse_parameter_config(param_config)
            base_values[param_name] = values[len(values) // 2]  # 中間值

        # 第二層：變更單一參數
        for param_name, param_config in self.parameters.items():
            values = self._parse_parameter_config(param_config)
            for value in values:
                combo = base_values.copy()
                combo[param_name] = value
                combinations.append(combo)

        # 第三層：關鍵參數組合（可擴展）
        # TODO: 實作關鍵參數交互測試

        return combinations

    def _parse_parameter_config(self, param_config: Union[List, Dict]) -> List[Any]:
        """
        解析參數配置

        Args:
            param_config: 參數配置，可以是列表或字典

        Returns:
            參數值列表
        """
        if isinstance(param_config, list):
            # 明確列表
            return param_config
        elif isinstance(param_config, dict):
            # 區間 + 間隔
            if 'range' in param_config and 'step' in param_config:
                start, end = param_config['range']
                step = param_config['step']
                values = []
                current = start
                while current <= end:
                    values.append(round(current, 6))  # 避免浮點精度問題
                    current += step
                return values
            else:
                raise ValueError(f"無效的參數配置格式: {param_config}")
        else:
            # 單一值
            return [param_config]

    def get_parameter_ranges(self) -> Dict[str, List[Any]]:
        """
        獲取各參數的可能值範圍

        Returns:
            參數範圍字典
        """
        ranges = {}
        for param_name, param_config in self.parameters.items():
            ranges[param_name] = self._parse_parameter_config(param_config)
        return ranges


def load_config(config_file: str) -> Dict[str, Any]:
    """
    載入測試配置檔案

    Args:
        config_file: 配置檔案路徑

    Returns:
        配置字典
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
