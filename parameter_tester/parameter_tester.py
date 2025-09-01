#!/usr/bin/env python3
"""
參數測試系統主入口
負責協調各個模組執行參數測試
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# 添加當前目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from parameter_generator import ParameterGenerator, load_config
from config_generator import ConfigGenerator
from test_executor import TestExecutor
from result_collector import ResultCollector
from utils import ensure_directory, cleanup_temp_directories, validate_config, format_duration


class ParameterTester:
    """參數測試器主類"""

    def __init__(self, config_file: str):
        """
        初始化參數測試器

        Args:
            config_file: 測試配置檔案路徑
        """
        self.config_file = config_file
        self.config = load_config(config_file)

        # 驗證配置
        errors = validate_config(self.config)
        if errors:
            raise ValueError(f"配置檔案錯誤:\n" + "\n".join(errors))

        # 初始化組件
        self.param_generator = ParameterGenerator(self.config)
        self.config_generator = ConfigGenerator(
            os.path.join(current_dir, "templates", "botsort_template.yaml")
        )
        self.test_executor = TestExecutor(
            os.path.join(os.path.dirname(current_dir), "ultralytics_inference_video.py"),
            self.config.get("test_settings", {})
        )
        self.result_collector = ResultCollector()

        # 臨時檔案追蹤
        self.temp_configs = []

    def run_tests(self) -> List[Dict[str, Any]]:
        """
        執行參數測試

        Returns:
            測試結果列表
        """
        print("=== 參數測試系統啟動 ===")
        print(f"配置檔案: {self.config_file}")
        print(f"測試策略: {self.config.get('strategy', {}).get('type', 'grid_search')}")

        # 生成參數組合
        print("\n1. 生成參數組合...")
        combinations = self.param_generator.generate_combinations()
        print(f"生成 {len(combinations)} 個參數組合")

        # 執行測試
        print("\n2. 開始執行測試...")
        results = []
        total_tests = len(combinations)

        for i, params in enumerate(combinations, 1):
            print(f"\n執行測試 {i}/{total_tests}")
            print(f"參數: {params}")

            try:
                # 生成配置檔案
                config_path = self.config_generator.generate_config(params)
                self.temp_configs.append(config_path)

                # 執行測試
                test_result = self.test_executor.execute_test(config_path, params)

                # 收集結果
                processed_result = self.result_collector.collect_result(test_result)
                results.append(processed_result)

                # 顯示結果摘要
                self._print_test_summary(processed_result)

            except Exception as e:
                print(f"測試失敗: {e}")
                error_result = {
                    "parameters": params,
                    "error": str(e),
                    "run_id": f"error_{i}"
                }
                results.append(error_result)

        # 清理臨時檔案
        if self.config.get("test_settings", {}).get("cleanup_temp_files", True):
            print("\n3. 清理臨時檔案...")
            cleanup_temp_directories([os.path.dirname(p) for p in self.temp_configs])

        print("\n=== 測試完成 ===")
        print(f"總共執行 {len(results)} 個測試")
        print(f"成功: {len([r for r in results if 'error' not in r])}")
        print(f"失敗: {len([r for r in results if 'error' in r])}")

        return results

    def _print_test_summary(self, result: Dict[str, Any]):
        """列印測試摘要"""
        if "error" in result:
            print(f"❌ 失敗: {result['error']}")
            return

        perf = result.get("performance_metrics", {})
        tracking = result.get("tracking_metrics", {})

        duration = result.get("execution_info", {}).get("duration", 0)
        print(f"✅ 成功 (耗時: {format_duration(duration)})")

        if perf.get("avg_fps"):
            print(f"   FPS: {perf['avg_fps']:.2f}")
        if tracking.get("total_tracks"):
            print(f"   軌跡數: {tracking['total_tracks']}")

    def generate_report(self, results: List[Dict[str, Any]], output_dir: str = "reports"):
        """
        生成測試報告

        Args:
            results: 測試結果列表
            output_dir: 輸出目錄
        """
        ensure_directory(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"parameter_test_report_{timestamp}.json")
        summary_file = os.path.join(output_dir, f"parameter_test_summary_{timestamp}.json")

        # 保存詳細結果
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 保存摘要
        summary = self.result_collector.get_summary_statistics()
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n報告已生成:")
        print(f"詳細結果: {report_file}")
        print(f"摘要報告: {summary_file}")

        # 列印摘要
        self._print_summary_report(summary)

    def _print_summary_report(self, summary: Dict[str, Any]):
        """列印摘要報告"""
        print("\n=== 測試摘要 ===")
        print(f"總測試數: {summary.get('total_tests', 0)}")
        print(f"成功測試: {summary.get('successful_tests', 0)}")
        print(f"失敗測試: {summary.get('failed_tests', 0)}")

        perf_summary = summary.get('performance_summary', {})
        if perf_summary:
            print("\n效能指標:")
            print(f"  平均 FPS: {perf_summary.get('avg_fps_mean', 0):.2f}")
            print(f"  中位 FPS: {perf_summary.get('avg_fps_median', 0):.2f}")
            print(f"  FPS 標準差: {perf_summary.get('avg_fps_std', 0):.2f}")
            print(f"  最小 FPS: {perf_summary.get('avg_fps_min', 0):.2f}")
            print(f"  最大 FPS: {perf_summary.get('avg_fps_max', 0):.2f}")
        param_analysis = summary.get('parameter_analysis', {})
        if param_analysis:
            print("\n參數影響分析:")
            for param_name, values in param_analysis.items():
                print(f"  {param_name}:")
                for param_value, avg_fps in values.items():
                    if avg_fps is not None:
                        print(f"    {param_value}: {avg_fps:.2f} FPS")
                    else:
                        print(f"    {param_value}: 無資料")
def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="BoTSORT 參數測試系統")
    parser.add_argument(
        "--config",
        type=str,
        default="test_config.yaml",
        help="測試配置檔案路徑"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="報告輸出目錄"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="僅分析現有結果，不執行測試"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        help="要分析的結果檔案路徑"
    )

    args = parser.parse_args()

    try:
        if args.analyze_only:
            if not args.results_file:
                print("錯誤: --analyze-only 需要指定 --results-file")
                sys.exit(1)

            # 僅分析模式
            collector = ResultCollector()
            collector.load_results(args.results_file)
            summary = collector.get_summary_statistics()

            ensure_directory(args.output_dir)
            summary_file = os.path.join(args.output_dir, "analysis_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"分析完成，結果保存至: {summary_file}")
            return

        # 執行測試
        tester = ParameterTester(args.config)
        results = tester.run_tests()
        tester.generate_report(results, args.output_dir)

    except Exception as e:
        print(f"錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
