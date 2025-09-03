"""
數據驗證和品質檢查模組
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import warnings


class DataValidator:
    """數據驗證器"""

    def __init__(self):
        self.validation_results = {}

    def validate_test_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        驗證測試結果的完整性和品質

        Args:
            results: 測試結果列表

        Returns:
            驗證報告
        """
        validation_report = {
            "total_results": len(results),
            "valid_results": 0,
            "issues": [],
            "data_quality": {},
            "recommendations": []
        }

        valid_results = []

        for i, result in enumerate(results):
            issues = self._validate_single_result(result, i)
            if not issues:
                valid_results.append(result)
            else:
                validation_report["issues"].extend(issues)

        validation_report["valid_results"] = len(valid_results)
        validation_report["data_quality"] = self._assess_data_quality(valid_results)
        validation_report["recommendations"] = self._generate_recommendations(validation_report)

        return validation_report

    def validate_test_result(self, result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        驗證單個測試結果

        Args:
            result: 測試結果字典

        Returns:
            (is_valid, error_messages)
        """
        issues = self._validate_single_result(result, 0)
        return len(issues) == 0, issues

    def validate_test_results_batch(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量驗證測試結果

        Args:
            results: 測試結果列表

        Returns:
            驗證摘要報告
        """
        return self.validate_test_results(results)

    def _validate_single_result(self, result: Dict[str, Any], index: int) -> List[str]:
        """驗證單個測試結果"""
        issues = []

        # 檢查必要欄位
        required_fields = ["run_id", "parameters", "performance_metrics", "execution_info"]
        for field in required_fields:
            if field not in result:
                issues.append(f"結果 {index}: 缺少必要欄位 '{field}'")

        # 檢查效能指標
        if "performance_metrics" in result:
            perf = result["performance_metrics"]

            # 檢查 FPS 合理性
            fps = perf.get("avg_fps")
            if fps is not None:
                if fps <= 0 or fps > 1000:
                    issues.append(f"結果 {index}: FPS 值異常 ({fps})")

            # 檢查推理時間合理性
            inference_time = perf.get("avg_inference_time")
            if inference_time is not None:
                if inference_time <= 0 or inference_time > 10:
                    issues.append(f"結果 {index}: 推理時間異常 ({inference_time})")

        # 檢查參數合理性
        if "parameters" in result:
            params = result["parameters"]

            # 檢查閾值參數範圍
            thresholds = ["track_high_thresh", "track_low_thresh", "new_track_thresh",
                         "match_thresh", "proximity_thresh", "appearance_thresh"]

            for thresh in thresholds:
                if thresh in params:
                    value = params[thresh]
                    if value < 0 or value > 1:
                        issues.append(f"結果 {index}: {thresh} 值超出範圍 [0,1] ({value})")

        return issues

    def _assess_data_quality(self, valid_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """評估數據品質"""
        if not valid_results:
            return {"quality_score": 0, "issues": ["沒有有效的測試結果"]}

        quality_report = {
            "quality_score": 0,
            "completeness": 0,
            "consistency": 0,
            "coverage": 0,
            "issues": []
        }

        # 評估完整性
        total_fields = ["performance_metrics", "tracking_metrics", "execution_info"]
        complete_results = 0

        for result in valid_results:
            if all(field in result and result[field] for field in total_fields):
                complete_results += 1

        quality_report["completeness"] = complete_results / len(valid_results)

        # 評估一致性（檢查相同參數的結果是否一致）
        param_combinations = {}
        for result in valid_results:
            param_key = str(sorted(result["parameters"].items()))
            if param_key in param_combinations:
                param_combinations[param_key].append(result)
            else:
                param_combinations[param_key] = [result]

        consistent_groups = 0
        for param_key, group in param_combinations.items():
            if len(group) > 1:
                fps_values = [r["performance_metrics"].get("avg_fps") for r in group
                             if r["performance_metrics"].get("avg_fps") is not None]
                if fps_values and len(set(fps_values)) == 1:
                    consistent_groups += 1

        quality_report["consistency"] = consistent_groups / len(param_combinations) if param_combinations else 0

        # 評估參數覆蓋度
        all_params = set()
        for result in valid_results:
            all_params.update(result["parameters"].keys())

        param_coverage = {}
        for param in all_params:
            values = set()
            for result in valid_results:
                if param in result["parameters"]:
                    values.add(result["parameters"][param])
            param_coverage[param] = len(values)

        avg_coverage = sum(param_coverage.values()) / len(param_coverage) if param_coverage else 0
        quality_report["coverage"] = min(avg_coverage / 5, 1.0)  # 假設每個參數最多5個值

        # 計算總體品質分數
        quality_report["quality_score"] = (
            quality_report["completeness"] * 0.4 +
            quality_report["consistency"] * 0.3 +
            quality_report["coverage"] * 0.3
        )

        return quality_report

    def _generate_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """生成改進建議"""
        recommendations = []

        if validation_report["valid_results"] == 0:
            recommendations.append("所有測試結果都有問題，建議檢查測試配置和執行流程")
            return recommendations

        quality = validation_report["data_quality"]

        if quality["completeness"] < 0.8:
            recommendations.append("數據完整性不足，建議確保所有必要欄位都被正確收集")

        if quality["consistency"] < 0.9:
            recommendations.append("相同參數的測試結果不一致，建議檢查測試環境的穩定性")

        if quality["coverage"] < 0.6:
            recommendations.append("參數覆蓋度不足，建議增加更多參數組合的測試")

        if len(validation_report["issues"]) > validation_report["total_results"] * 0.1:
            recommendations.append("測試結果錯誤率過高，建議檢查測試執行邏輯")

        return recommendations

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """清理數據，移除異常值和無效數據"""
        cleaning_log = []
        original_shape = df.shape

        # 移除完全重複的行
        df = df.drop_duplicates()
        if df.shape[0] < original_shape[0]:
            cleaning_log.append(f"移除 {original_shape[0] - df.shape[0]} 個重複行")

        # 處理數值型欄位的異常值
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # 使用 IQR 方法檢測異常值
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                cleaning_log.append(f"欄位 {col}: 發現 {len(outliers)} 個異常值")
                # 選擇性移除異常值（可以配置）
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        return df, cleaning_log
