"""
報告生成器
創建專業的 HTML 和 Markdown 格式報告
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
import base64


class ReportGenerator:
    """報告生成器"""

    def __init__(self, analysis_results: Dict[str, Any], output_dir: str = "reports"):
        self.analysis_results = analysis_results
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_html_report(self) -> str:
        """生成 HTML 格式報告"""
        html_template = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BoTSORT 參數測試分析報告</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .summary-box {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .metric-label {{ font-size: 0.9em; color: #7f8c8d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .chart-container {{ text-align: center; margin: 30px 0; }}
        .recommendation {{ background: #d5f4e6; padding: 15px; border-left: 4px solid #27ae60; margin: 10px 0; }}
        .warning {{ background: #fdf2e9; padding: 15px; border-left: 4px solid #f39c12; margin: 10px 0; }}
        .error {{ background: #fadbd8; padding: 15px; border-left: 4px solid #e74c3c; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BoTSORT 參數測試分析報告</h1>
        <p><strong>生成時間:</strong> {timestamp}</p>

        {executive_summary}

        {test_overview}

        {parameter_analysis}

        {performance_analysis}

        {recommendations}

        {technical_details}
    </div>
</body>
</html>
"""

        # 構建各個部分
        executive_summary = self._build_executive_summary()
        test_overview = self._build_test_overview()
        parameter_analysis = self._build_parameter_analysis()
        performance_analysis = self._build_performance_analysis()
        recommendations = self._build_recommendations()
        technical_details = self._build_technical_details()

        # 填入模板
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            executive_summary=executive_summary,
            test_overview=test_overview,
            parameter_analysis=parameter_analysis,
            performance_analysis=performance_analysis,
            recommendations=recommendations,
            technical_details=technical_details
        )

        # 保存檔案
        html_path = os.path.join(self.output_dir, f"analysis_report_{self.timestamp}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path

    def _build_executive_summary(self) -> str:
        """構建執行摘要"""
        total_tests = self.analysis_results.get("total_tests", 0)
        optimal_params = self.analysis_results.get("optimal_parameters", {})

        summary = f"""
        <div class="summary-box">
            <h2>執行摘要</h2>
            <div class="metric">
                <div class="metric-value">{total_tests}</div>
                <div class="metric-label">測試總數</div>
            </div>
        """

        if optimal_params and "best_metric_value" in optimal_params:
            summary += f"""
            <div class="metric">
                <div class="metric-value">{optimal_params['best_metric_value']:.2f}</div>
                <div class="metric-label">最佳 FPS</div>
            </div>
            """

        summary += """
        </div>
        """

        return summary

    def _build_test_overview(self) -> str:
        """構建測試概覽"""
        performance_comp = self.analysis_results.get("performance_comparison", {})
        overall_stats = performance_comp.get("overall_statistics", {})

        if not overall_stats:
            return "<h2>測試概覽</h2><p>沒有可用的測試數據。</p>"

        overview = """
        <h2>測試概覽</h2>
        <table>
            <tr><th>指標</th><th>平均值</th><th>中位數</th><th>標準差</th><th>最小值</th><th>最大值</th></tr>
        """

        for metric, stats in overall_stats.items():
            if isinstance(stats, dict):
                overview += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{stats.get('mean', 'N/A'):.4f if isinstance(stats.get('mean'), (int, float)) else 'N/A'}</td>
                    <td>{stats.get('median', 'N/A'):.4f if isinstance(stats.get('median'), (int, float)) else 'N/A'}</td>
                    <td>{stats.get('std', 'N/A'):.4f if isinstance(stats.get('std'), (int, float)) else 'N/A'}</td>
                    <td>{stats.get('min', 'N/A'):.4f if isinstance(stats.get('min'), (int, float)) else 'N/A'}</td>
                    <td>{stats.get('max', 'N/A'):.4f if isinstance(stats.get('max'), (int, float)) else 'N/A'}</td>
                </tr>
                """

        overview += "</table>"
        return overview

    def _build_parameter_analysis(self) -> str:
        """構建參數分析部分"""
        sensitivity = self.analysis_results.get("parameter_sensitivity", {})

        if not sensitivity:
            return "<h2>參數分析</h2><p>沒有可用的參數敏感度數據。</p>"

        analysis = """
        <h2>參數敏感度分析</h2>
        <table>
            <tr><th>參數</th><th>相關係數</th><th>顯著性</th><th>樣本數</th></tr>
        """

        for param, data in sensitivity.items():
            if isinstance(data, dict):
                correlation = data.get('correlation', 0)
                significant = "是" if data.get('significant', False) else "否"
                sample_size = data.get('sample_size', 'N/A')

                analysis += f"""
                <tr>
                    <td>{param}</td>
                    <td>{correlation:.4f if isinstance(correlation, (int, float)) else 'N/A'}</td>
                    <td>{significant}</td>
                    <td>{sample_size}</td>
                </tr>
                """

        analysis += "</table>"
        return analysis

    def _build_performance_analysis(self) -> str:
        """構建效能分析部分"""
        optimal = self.analysis_results.get("optimal_parameters", {})

        if not optimal or "optimal_parameters" not in optimal:
            return "<h2>效能分析</h2><p>沒有找到最佳參數組合。</p>"

        analysis = f"""
        <h2>效能分析</h2>
        <h3>最佳參數組合</h3>
        <div class="recommendation">
            <strong>最佳效能值:</strong> {optimal.get('best_metric_value', 'N/A')}<br>
            <strong>參數組合:</strong>
            <ul>
        """

        for param, value in optimal.get("optimal_parameters", {}).items():
            analysis += f"<li>{param}: {value}</li>"

        analysis += """
            </ul>
        </div>
        """

        return analysis

    def _build_recommendations(self) -> str:
        """構建建議部分"""
        recommendations = self.analysis_results.get("recommendations", [])

        if not recommendations:
            return "<h2>建議</h2><p>沒有特定的調優建議。</p>"

        rec_html = "<h2>調優建議</h2>"

        for i, rec in enumerate(recommendations, 1):
            rec_html += f'<div class="recommendation">{i}. {rec}</div>'

        return rec_html

    def _build_technical_details(self) -> str:
        """構建技術細節部分"""
        return """
        <h2>技術細節</h2>
        <h3>分析方法</h3>
        <ul>
            <li>參數敏感度分析：使用 Pearson 相關係數和 ANOVA F檢驗</li>
            <li>統計顯著性：p < 0.05</li>
            <li>數據清理：移除異常值和無效數據</li>
            <li>效能指標：主要關注平均 FPS</li>
        </ul>

        <h3>數據品質</h3>
        <p>所有分析都基於通過數據驗證的測試結果。</p>
        """

    def generate_markdown_report(self) -> str:
        """生成 Markdown 格式報告"""
        md_content = f"""# BoTSORT 參數測試分析報告

**生成時間**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 執行摘要

- **測試總數**: {self.analysis_results.get("total_tests", 0)}
- **分析完成**: ✅

## 參數敏感度分析

"""

        sensitivity = self.analysis_results.get("parameter_sensitivity", {})
        if sensitivity:
            md_content += "| 參數 | 相關係數 | 顯著性 |\n|------|----------|--------|\n"
            for param, data in sensitivity.items():
                if isinstance(data, dict):
                    correlation = data.get('correlation', 0)
                    significant = "✅" if data.get('significant', False) else "❌"
                    md_content += f"| {param} | {correlation:.4f if isinstance(correlation, (int, float)) else 'N/A'} | {significant} |\n"

        optimal = self.analysis_results.get("optimal_parameters", {})
        if optimal and "optimal_parameters" in optimal:
            md_content += f"""
## 最佳參數組合

**最佳效能**: {optimal.get('best_metric_value', 'N/A')}

**參數設定**:
"""
            for param, value in optimal.get("optimal_parameters", {}).items():
                md_content += f"- {param}: {value}\n"

        recommendations = self.analysis_results.get("recommendations", [])
        if recommendations:
            md_content += "\n## 調優建議\n\n"
            for i, rec in enumerate(recommendations, 1):
                md_content += f"{i}. {rec}\n"

        # 保存檔案
        md_path = os.path.join(self.output_dir, f"analysis_report_{self.timestamp}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return md_path
