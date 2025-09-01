"""
YAML 配置生成器
負責基於模板生成測試用的臨時配置檔案
"""

import os
import tempfile
import yaml
from typing import Dict, Any, List
from pathlib import Path


class ConfigGenerator:
    """配置檔案生成器"""

    def __init__(self, template_path: str):
        """
        初始化配置生成器

        Args:
            template_path: 模板檔案路徑
        """
        self.template_path = template_path
        self.template = self._load_template()

    def _load_template(self) -> Dict[str, Any]:
        """載入模板配置"""
        with open(self.template_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def generate_config(self, parameters: Dict[str, Any]) -> str:
        """
        生成臨時配置檔案

        Args:
            parameters: 要修改的參數字典

        Returns:
            臨時配置檔案路徑
        """
        # 複製模板
        config = self.template.copy()

        # 更新參數
        for param_name, param_value in parameters.items():
            if param_name in config:
                config[param_name] = param_value
            else:
                # 如果參數不在頂層，嘗試在嵌套結構中查找
                self._update_nested_param(config, param_name, param_value)

        # 生成臨時檔案
        temp_file = self._create_temp_config_file(config)
        return temp_file

    def _update_nested_param(self, config: Dict[str, Any], param_name: str, param_value: Any):
        """
        更新嵌套參數

        Args:
            config: 配置字典
            param_name: 參數名稱
            param_value: 參數值
        """
        # 簡單實現：假設參數在頂層
        # 可以擴展為支援嵌套結構
        pass

    def _create_temp_config_file(self, config: Dict[str, Any]) -> str:
        """
        創建臨時配置檔案

        Args:
            config: 配置字典

        Returns:
            臨時檔案路徑
        """
        # 創建臨時目錄
        temp_dir = tempfile.mkdtemp(prefix='botsort_test_')

        # 生成檔案名稱
        config_file = os.path.join(temp_dir, 'custom_botsort.yaml')

        # 寫入配置
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return config_file

    def cleanup_temp_files(self, config_paths: List[str]):
        """
        清理臨時檔案

        Args:
            config_paths: 要清理的配置檔案路徑列表
        """
        for config_path in config_paths:
            try:
                temp_dir = os.path.dirname(config_path)
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"清理臨時檔案失敗 {config_path}: {e}")

    def get_template_parameters(self) -> Dict[str, Any]:
        """
        獲取模板中的參數

        Returns:
            參數字典
        """
        return self.template
