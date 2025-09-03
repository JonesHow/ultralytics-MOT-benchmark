"""
結果收集器
負責收集測試結果並進行初步分析
"""

import os
import json
import re
from typing import Dict, Any, List
from datetime import datetime
import statistics
from statistics_analyzer import StatisticsAnalyzer
from loguru import logger


class NumpyEncoder(json.JSONEncoder):
    """自定義 JSON 編碼器，用於處理 numpy 類型"""
    def default(self, obj):
        if hasattr(obj, 'item'):  # numpy 類型
            return obj.item()
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            return list(obj)
        return super().default(obj)


class ResultCollector:
    """結果收集器"""

    def __init__(self):
        self.results = []

    def collect_result(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集單個測試結果

        Args:
            test_result: 測試結果字典

        Returns:
            處理後的結果
        """
        processed_result = self._process_test_result(test_result)
        self.results.append(processed_result)
        return processed_result

    def _process_test_result(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        處理測試結果，提取關鍵指標

        Args:
            test_result: 原始測試結果

        Returns:
            處理後的結果
        """
        processed = test_result.copy()

        # 提取效能指標
        performance_metrics = self._extract_performance_metrics(test_result)
        processed["performance_metrics"] = performance_metrics

        # 提取追蹤指標
        tracking_metrics = self._extract_tracking_metrics(test_result)
        processed["tracking_metrics"] = tracking_metrics

        return processed

    def _extract_performance_metrics(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        從測試結果中提取效能指標

        Args:
            test_result: 測試結果

        Returns:
            效能指標字典
        """
        metrics = {
            "avg_inference_time": None,
            "min_inference_time": None,
            "max_inference_time": None,
            "median_inference_time": None,
            "avg_fps": None,
            "frames_processed": 0
        }

        # 從 metadata 中提取
        if "metadata" in test_result:
            metadata = test_result["metadata"]
            if isinstance(metadata, dict):
                # 嘗試從 metadata 中提取指標
                # 根據實際的 metadata 結構調整
                if "stats" in metadata:
                    stats = metadata["stats"]
                    metrics.update({
                        "avg_inference_time": stats.get("avg_inference_time"),
                        "frames_processed": stats.get("frames_processed", 0)
                    })
                    # 計算 FPS
                    if stats.get("avg_inference_time"):
                        metrics["avg_fps"] = 1.0 / stats["avg_inference_time"]

        # 從 logs 中提取（如果 metadata 不可用）
        if not any(metrics.values()):
            log_path = test_result.get("file_paths", {}).get("log_path")
            if log_path and os.path.exists(log_path):
                log_metrics = self._parse_log_file(log_path)
                metrics.update(log_metrics)

        return metrics

    def _extract_tracking_metrics(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        從測試結果中提取追蹤指標

        Args:
            test_result: 測試結果

        Returns:
            追蹤指標字典
        """
        # 從 metadata 中提取新的追蹤指標
        if "metadata" in test_result:
            metadata = test_result["metadata"]
            if isinstance(metadata, dict) and "tracking_metrics" in metadata:
                tracking_metrics = metadata["tracking_metrics"]
                if isinstance(tracking_metrics, dict):
                    return tracking_metrics.copy()

        # 如果沒有找到新的結構，回退到舊結構或返回默認值
        if "metadata" in test_result:
            metadata = test_result["metadata"]
            if isinstance(metadata, dict):
                if "tracking" in metadata:
                    tracking = metadata["tracking"]
                    return {
                        "total_tracks": tracking.get("total_tracks", 0),
                        "avg_track_length": tracking.get("avg_track_length", 0),
                        "track_switches": tracking.get("track_switches", 0)
                    }

        # 返回默認的空指標
        return {
            "total_tracks": 0,
            "avg_track_length": 0,
            "max_track_length": 0,
            "min_track_length": 0,
            "track_lengths": [],
            "continuity_metrics": {},
            "fragmentation_metrics": {},
            "new_track_frequency": {},
            "id_switch_stats": {},
            "frames_processed": 0,
            "avg_active_tracks_per_frame": 0
        }

    def _parse_log_file(self, log_path: str) -> Dict[str, Any]:
        """
        解析日誌檔案提取效能指標

        Args:
            log_path: 日誌檔案路徑

        Returns:
            效能指標字典
        """
        metrics = {}

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取 FPS 資訊
            fps_pattern = r'avg_fps[:\s]+([\d.]+)'
            fps_match = re.search(fps_pattern, content, re.IGNORECASE)
            if fps_match:
                metrics["avg_fps"] = float(fps_match.group(1))

            # 提取推理時間
            time_pattern = r'avg_inference_time[:\s]+([\d.]+)'
            time_match = re.search(time_pattern, content, re.IGNORECASE)
            if time_match:
                metrics["avg_inference_time"] = float(time_match.group(1))

            # 提取處理幀數
            frames_pattern = r'frames_processed[:\s]+(\d+)'
            frames_match = re.search(frames_pattern, content, re.IGNORECASE)
            if frames_match:
                metrics["frames_processed"] = int(frames_match.group(1))

        except Exception as e:
            logger.warning(f"解析日誌檔案失敗 {log_path}: {e}")

        return metrics

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        獲取總結統計

        Returns:
            統計摘要
        """
        if not self.results:
            return {}

        # 使用統計分析器進行深入分析
        analyzer = StatisticsAnalyzer(self.results)

        summary = {
            "total_tests": len(self.results),
            "successful_tests": len([r for r in self.results if "error" not in r]),
            "failed_tests": len([r for r in self.results if "error" in r]),
            "performance_summary": self._calculate_performance_summary(),
            "parameter_sensitivity": analyzer.parameter_sensitivity_analysis(),
            "optimal_parameters": analyzer.find_optimal_parameters(),
            "performance_comparison": analyzer.performance_comparison_report(),
            "recommendations": analyzer._generate_recommendations()
        }

        return summary

    def _calculate_performance_summary(self) -> Dict[str, Any]:
        """計算效能摘要"""
        perf_data = [r["performance_metrics"] for r in self.results if r["performance_metrics"]["avg_fps"]]

        if not perf_data:
            return {}

        fps_values = [p["avg_fps"] for p in perf_data if p["avg_fps"] is not None]

        summary = {}
        if fps_values:
            summary.update({
                "avg_fps_mean": statistics.mean(fps_values),
                "avg_fps_median": statistics.median(fps_values),
                "avg_fps_std": statistics.stdev(fps_values) if len(fps_values) > 1 else 0,
                "avg_fps_min": min(fps_values),
                "avg_fps_max": max(fps_values)
            })

        return summary

    def _analyze_parameters(self) -> Dict[str, Any]:
        """分析參數影響 - 已由 StatisticsAnalyzer 替換"""
        # 此方法已棄用，請使用 StatisticsAnalyzer.parameter_sensitivity_analysis()
        return {}

    def save_results(self, output_path: str):
        """
        保存結果到檔案

        Args:
            output_path: 輸出檔案路徑
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    def generate_analysis_report(self, output_dir: str = "reports") -> str:
        """
        生成完整的分析報告

        Args:
            output_dir: 輸出目錄

        Returns:
            報告檔案路徑
        """
        analyzer = StatisticsAnalyzer(self.results)
        return analyzer.generate_analysis_report(output_dir)
