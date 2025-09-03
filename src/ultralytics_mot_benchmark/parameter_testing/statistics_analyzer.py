"""
統計分析模組
負責對測試結果進行深入的統計分析
"""

import os
import json
import statistics
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_validator import DataValidator
from loguru import logger


class StatisticsAnalyzer:
    """統計分析器"""

    def __init__(self, results: List[Dict[str, Any]]):
        """
        初始化統計分析器

        Args:
            results: 測試結果列表
        """
        self.results = results
        self.validator = DataValidator()

        # 驗證數據品質
        self.validation_report = self.validator.validate_test_results(results)

        self.df = self._create_dataframe()

        # 清理數據
        if not self.df.empty:
            self.df, self.cleaning_log = self.validator.clean_data(self.df)
        else:
            self.cleaning_log = []

    def _create_dataframe(self) -> pd.DataFrame:
        """將結果轉換為 DataFrame 便於分析"""
        data = []
        for result in self.results:
            row = {}

            # 參數
            if "parameters" in result:
                row.update(result["parameters"])

            # 效能指標
            if "performance_metrics" in result:
                perf = result["performance_metrics"]
                row.update({
                    "avg_fps": perf.get("avg_fps"),
                    "avg_inference_time": perf.get("avg_inference_time"),
                    "frames_processed": perf.get("frames_processed", 0)
                })

            # 追蹤指標
            if "tracking_metrics" in result:
                tracking = result["tracking_metrics"]
                row.update({
                    "total_tracks": tracking.get("total_tracks", 0),
                    "avg_track_length": tracking.get("avg_track_length", 0),
                    "max_track_length": tracking.get("max_track_length", 0),
                    "min_track_length": tracking.get("min_track_length", 0),
                    "frames_processed_tracking": tracking.get("frames_processed", 0),
                    "avg_active_tracks_per_frame": tracking.get("avg_active_tracks_per_frame", 0),
                    # 連續性指標
                    "continuity_ratio": tracking.get("continuity_metrics", {}).get("continuity_ratio", 0),
                    "total_gaps": tracking.get("continuity_metrics", {}).get("total_gaps", 0),
                    "tracks_with_gaps": tracking.get("continuity_metrics", {}).get("tracks_with_gaps", 0),
                    # 片段化指標
                    "fragmentation_ratio": tracking.get("fragmentation_metrics", {}).get("fragmentation_ratio", 0),
                    "cv_track_length": tracking.get("fragmentation_metrics", {}).get("cv_track_length", 0),
                    # 新軌跡頻率
                    "total_new_tracks": tracking.get("new_track_frequency", {}).get("total_new_tracks", 0),
                    "avg_new_tracks_per_frame": tracking.get("new_track_frequency", {}).get("avg_new_tracks_per_frame", 0),
                    # ID 切換
                    "total_id_switches": tracking.get("id_switch_stats", {}).get("total_id_switches", 0),
                    "id_switches_per_frame": tracking.get("id_switch_stats", {}).get("id_switches_per_frame", 0)
                })

            # 執行資訊
            if "execution_info" in result:
                exec_info = result["execution_info"]
                row.update({
                    "duration": exec_info.get("duration", 0),
                    "device": exec_info.get("device", "")
                })

            data.append(row)

        return pd.DataFrame(data)

    def parameter_sensitivity_analysis(self, target_metric: str = "avg_fps") -> Dict[str, Any]:
        """
        參數敏感度分析

        Args:
            target_metric: 目標指標，預設為 avg_fps

        Returns:
            敏感度分析結果
        """
        if self.df.empty or target_metric not in self.df.columns:
            return {}

        # 清理數據：移除 NaN 值
        clean_df = self.df.dropna(subset=[target_metric])
        if clean_df.empty:
            logger.warning(f"所有 {target_metric} 值都是 NaN，無法進行敏感度分析")
            return {}

        sensitivity_results = {}

        # 獲取數值型和分類型參數
        numeric_params = clean_df.select_dtypes(include=[np.number]).columns
        categorical_params = clean_df.select_dtypes(include=['object', 'category']).columns
        all_params = list(numeric_params) + list(categorical_params)
        all_params = [p for p in all_params if p != target_metric and p in clean_df.columns and p != '_repetition']

        for param in all_params:
            # 清理參數數據
            param_clean_df = clean_df.dropna(subset=[param])
            if param_clean_df.empty or param_clean_df[param].nunique() < 2:
                continue

            try:
                # 計算相關係數（只對數值型參數）
                if param in numeric_params:
                    correlation = param_clean_df[param].corr(param_clean_df[target_metric])
                else:
                    # 對於分類變數，使用 ANOVA F 統計量作為相關性的代理指標
                    correlation = None

                # 檢查是否有足夠的數據進行 ANOVA
                groups = [group[target_metric].values for name, group in param_clean_df.groupby(param)]
                groups = [g for g in groups if len(g) > 0]  # 移除空組

                if len(groups) > 1 and all(len(g) >= 2 for g in groups):  # 每組至少2個數據點
                    f_stat, p_value = stats.f_oneway(*groups)
                else:
                    f_stat, p_value = None, None
                    logger.warning(f"參數 {param} 數據不足，無法進行 ANOVA 分析")

                sensitivity_results[param] = {
                    "correlation": float(correlation) if correlation is not None and not np.isnan(correlation) else None,
                    "f_statistic": float(f_stat) if f_stat is not None else None,
                    "p_value": float(p_value) if p_value is not None else None,
                    "significant": bool(p_value is not None and p_value < 0.05),
                    "sample_size": int(len(param_clean_df)),
                    "unique_values": int(param_clean_df[param].nunique())
                }
            except Exception as e:
                logger.error(f"分析參數 {param} 時發生錯誤: {e}")
                continue

        return sensitivity_results

    def advanced_parameter_importance_analysis(self, target_metric: str = "avg_fps") -> Dict[str, Any]:
        """
        使用機器學習方法分析參數重要性

        Args:
            target_metric: 目標指標

        Returns:
            參數重要性分析結果
        """
        if self.df.empty or target_metric not in self.df.columns:
            return {}

        try:
            # 準備數據
            feature_cols = self.df.select_dtypes(include=[np.number]).columns
            feature_cols = [c for c in feature_cols if c != target_metric and c not in ['duration', 'frames_processed']]

            if not feature_cols:
                return {"error": "沒有可用的特徵欄位"}

            X = self.df[feature_cols].fillna(self.df[feature_cols].mean())
            y = self.df[target_metric].fillna(self.df[target_metric].mean())

            if len(X) < 5:  # 至少需要5個樣本
                return {"error": "樣本數量不足，需要至少5個樣本"}

            # 使用 Random Forest 分析特徵重要性
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)

            # 計算交叉驗證分數
            cv_scores = cross_val_score(rf, X, y, cv=min(5, len(X)), scoring='r2')

            importance_results = {
                "model_performance": {
                    "r2_score": float(rf.score(X, y)),
                    "cv_mean_r2": float(cv_scores.mean()),
                    "cv_std_r2": float(cv_scores.std())
                },
                "feature_importance": {},
                "top_features": []
            }

            # 特徵重要性
            for feature, importance in zip(feature_cols, rf.feature_importances_):
                importance_results["feature_importance"][feature] = float(importance)

            # 排序並獲取前3個重要特徵
            sorted_features = sorted(importance_results["feature_importance"].items(),
                                   key=lambda x: x[1], reverse=True)
            importance_results["top_features"] = sorted_features[:3]

            return importance_results

        except ImportError:
            return {"error": "需要安裝 scikit-learn 來使用高級分析功能"}
        except Exception as e:
            return {"error": f"高級分析失敗: {str(e)}"}

    def statistical_significance_test(self, param: str, target_metric: str = "avg_fps") -> Dict[str, Any]:
        """
        對參數進行統計顯著性檢驗

        Args:
            param: 要檢驗的參數
            target_metric: 目標指標

        Returns:
            統計檢驗結果
        """
        if self.df.empty or param not in self.df.columns or target_metric not in self.df.columns:
            return {}

        clean_df = self.df[[param, target_metric]].dropna()
        if clean_df.empty:
            return {}

        groups = [group[target_metric].values for name, group in clean_df.groupby(param)]
        groups = [g for g in groups if len(g) >= 2]  # 至少2個觀測值

        if len(groups) < 2:
            return {"error": "分組數量不足，無法進行統計檢驗"}

        results = {}

        try:
            # ANOVA F檢驗
            f_stat, p_value = stats.f_oneway(*groups)
            results["anova"] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            }

            # Kruskal-Wallis 檢驗（非參數）
            h_stat, p_value_kw = stats.kruskal(*groups)
            results["kruskal_wallis"] = {
                "h_statistic": float(h_stat),
                "p_value": float(p_value_kw),
                "significant": p_value_kw < 0.05
            }

            # 效應量（Eta-squared）
            ss_total = sum([(x - clean_df[target_metric].mean())**2 for x in clean_df[target_metric]])
            ss_between = sum([len(group) * (group.mean() - clean_df[target_metric].mean())**2 for group in groups])
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            results["effect_size"] = {
                "eta_squared": float(eta_squared),
                "interpretation": self._interpret_effect_size(eta_squared)
            }

        except Exception as e:
            results["error"] = f"統計檢驗失敗: {str(e)}"

        return results

    def _interpret_effect_size(self, eta_squared: float) -> str:
        """解釋效應量大小"""
        if eta_squared < 0.01:
            return "極小"
        elif eta_squared < 0.06:
            return "小"
        elif eta_squared < 0.14:
            return "中等"
        else:
            return "大"

    def parameter_interaction_analysis(self, param1: str, param2: str, target_metric: str = "avg_fps") -> Dict[str, Any]:
        """
        參數交互作用分析

        Args:
            param1: 第一個參數
            param2: 第二個參數
            target_metric: 目標指標

        Returns:
            交互作用分析結果
        """
        if not all(col in self.df.columns for col in [param1, param2, target_metric]):
            return {}

        try:
            # 創建交互作用圖數據
            pivot_table = self.df.pivot_table(
                values=target_metric,
                index=param1,
                columns=param2,
                aggfunc='mean'
            )

            # 計算交互作用效果
            interaction_effect = self._calculate_interaction_effect(param1, param2, target_metric)

            return {
                "pivot_table": pivot_table.to_dict(),
                "interaction_effect": interaction_effect,
                "heatmap_data": pivot_table.values.tolist(),
                "param1_values": pivot_table.index.tolist(),
                "param2_values": pivot_table.columns.tolist()
            }
        except Exception as e:
            logger.error(f"分析參數交互作用時發生錯誤: {e}")
            return {}

    def _calculate_interaction_effect(self, param1: str, param2: str, target_metric: str) -> float:
        """計算交互作用效果大小"""
        try:
            # 使用雙因子變異數分析
            formula = f'{target_metric} ~ {param1} * {param2}'
            # 這裡簡化處理，實際應該使用 statsmodels
            # 計算簡單的交互作用指標
            groups = {}
            for p1_val in self.df[param1].unique():
                for p2_val in self.df[param2].unique():
                    mask = (self.df[param1] == p1_val) & (self.df[param2] == p2_val)
                    if mask.any():
                        groups[f"{p1_val}_{p2_val}"] = self.df.loc[mask, target_metric].mean()

            # 計算交互作用的變異係數
            values = list(groups.values())
            if len(values) > 1:
                return statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) != 0 else 0
            return 0
        except:
            return 0

    def find_optimal_parameters(self, target_metric: str = "avg_fps", optimization: str = "maximize") -> Dict[str, Any]:
        """
        尋找最佳參數組合

        Args:
            target_metric: 目標指標
            optimization: "maximize" 或 "minimize"

        Returns:
            最佳參數組合
        """
        if self.df.empty or target_metric not in self.df.columns:
            return {}

        # 過濾出有效的結果
        valid_df = self.df.dropna(subset=[target_metric])

        if valid_df.empty:
            return {}

        # 根據優化方向排序
        if optimization == "maximize":
            best_result = valid_df.loc[valid_df[target_metric].idxmax()]
        else:
            best_result = valid_df.loc[valid_df[target_metric].idxmin()]

        optimal_params = {}
        for col in valid_df.columns:
            if col != target_metric and not col.startswith(('duration', 'device')):
                optimal_params[col] = best_result[col]

        return {
            "optimal_parameters": optimal_params,
            "best_metric_value": best_result[target_metric],
            "optimization_direction": optimization
        }

    def bayesian_optimization_analysis(self, target_metric: str = "avg_fps") -> Dict[str, Any]:
        """
        貝葉斯優化分析，建議下一組測試參數

        Args:
            target_metric: 目標指標

        Returns:
            優化建議
        """
        if self.df.empty or target_metric not in self.df.columns:
            return {}

        try:
            # 簡化的貝葉斯優化建議
            # 實際應用中可以使用 scikit-optimize 或 Optuna

            # 找到目前最佳結果
            best_idx = self.df[target_metric].idxmax()
            best_params = self.df.loc[best_idx].to_dict()

            # 分析參數空間的探索情況
            param_ranges = {}
            numeric_params = self.df.select_dtypes(include=[np.number]).columns
            numeric_params = [p for p in numeric_params if p != target_metric]

            for param in numeric_params:
                param_values = self.df[param].dropna()
                if len(param_values) > 1:
                    param_ranges[param] = {
                        'min': float(param_values.min()),
                        'max': float(param_values.max()),
                        'std': float(param_values.std()),
                        'tested_values': param_values.unique().tolist()
                    }

            # 建議新的測試點（簡化策略）
            suggested_params = {}
            for param, range_info in param_ranges.items():
                current_best = best_params.get(param)
                if current_best is not None:
                    # 在最佳點附近探索
                    std_dev = range_info['std']
                    suggested_params[param] = {
                        'explore_around_best': [
                            max(range_info['min'], current_best - std_dev/2),
                            min(range_info['max'], current_best + std_dev/2)
                        ],
                        'current_best': current_best
                    }

            return {
                'current_best_params': {k: v for k, v in best_params.items() if k in numeric_params},
                'best_metric_value': best_params.get(target_metric),
                'parameter_ranges': param_ranges,
                'suggested_exploration': suggested_params,
                'exploration_completeness': self._assess_exploration_completeness(param_ranges)
            }

        except Exception as e:
            logger.error(f"貝葉斯優化分析時發生錯誤: {e}")
            return {}

    def _assess_exploration_completeness(self, param_ranges: Dict[str, Any]) -> Dict[str, float]:
        """評估參數空間探索的完整性"""
        completeness = {}

        for param, range_info in param_ranges.items():
            tested_values = range_info['tested_values']
            range_span = range_info['max'] - range_info['min']

            if range_span > 0:
                # 簡單的完整性評估：測試點數 / 理論最大點數
                # 假設每個參數應該測試至少5個不同的值
                ideal_test_points = 5
                actual_test_points = len(tested_values)
                completeness[param] = min(1.0, actual_test_points / ideal_test_points)

        return completeness

    def multi_objective_analysis(self, metrics: List[str] = None) -> Dict[str, Any]:
        """
        多目標優化分析

        Args:
            metrics: 目標指標列表，如 ['avg_fps', 'total_tracks']

        Returns:
            多目標分析結果
        """
        if not metrics:
            metrics = ['avg_fps']
            if 'total_tracks' in self.df.columns:
                metrics.append('total_tracks')

        valid_metrics = [m for m in metrics if m in self.df.columns]
        if len(valid_metrics) < 2:
            return {}

        try:
            # 標準化指標（0-1 範圍）
            normalized_df = self.df.copy()
            for metric in valid_metrics:
                metric_values = self.df[metric].dropna()
                if len(metric_values) > 0:
                    min_val = metric_values.min()
                    max_val = metric_values.max()
                    if max_val > min_val:
                        normalized_df[f'{metric}_normalized'] = (
                            (self.df[metric] - min_val) / (max_val - min_val)
                        )
                    else:
                        normalized_df[f'{metric}_normalized'] = 1.0

            # 計算綜合分數（簡單加權平均）
            normalized_cols = [f'{m}_normalized' for m in valid_metrics]
            normalized_df['composite_score'] = normalized_df[normalized_cols].mean(axis=1)

            # 找到帕累托前沿（簡化版本）
            pareto_front = self._find_pareto_front(normalized_df, valid_metrics)

            return {
                'metrics_analyzed': valid_metrics,
                'pareto_front_indices': pareto_front,
                'best_composite_score': {
                    'index': normalized_df['composite_score'].idxmax(),
                    'score': normalized_df['composite_score'].max(),
                    'parameters': normalized_df.loc[normalized_df['composite_score'].idxmax()].to_dict()
                },
                'trade_offs': self._analyze_trade_offs(normalized_df, valid_metrics)
            }

        except Exception as e:
            logger.error(f"多目標分析時發生錯誤: {e}")
            return {}

    def _find_pareto_front(self, df: pd.DataFrame, metrics: List[str]) -> List[int]:
        """找到帕累托前沿（簡化版本）"""
        pareto_front = []

        for i in range(len(df)):
            is_dominated = False
            current_values = [df.iloc[i][metric] for metric in metrics]

            for j in range(len(df)):
                if i != j:
                    other_values = [df.iloc[j][metric] for metric in metrics]

                    # 檢查是否被其他點支配
                    all_better_or_equal = all(other >= current for other, current in zip(other_values, current_values))
                    at_least_one_better = any(other > current for other, current in zip(other_values, current_values))

                    if all_better_or_equal and at_least_one_better:
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_front.append(i)

        return pareto_front

    def _analyze_trade_offs(self, df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """分析指標之間的權衡關係"""
        trade_offs = {}

        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                correlation = df[metric1].corr(df[metric2])
                trade_offs[f'{metric1}_vs_{metric2}'] = {
                    'correlation': float(correlation),
                    'relationship': 'positive' if correlation > 0.3 else 'negative' if correlation < -0.3 else 'weak'
                }

        return trade_offs

    def advanced_statistical_tests(self, target_metric: str = "avg_fps") -> Dict[str, Any]:
        """
        進階統計檢驗

        Args:
            target_metric: 目標指標

        Returns:
            統計檢驗結果
        """
        results = {}

        if self.df.empty or target_metric not in self.df.columns:
            return results

        try:
            target_values = self.df[target_metric].dropna()
            if len(target_values) < 3:
                return results

            # 正態性檢驗
            from scipy.stats import shapiro, normaltest
            shapiro_stat, shapiro_p = shapiro(target_values)
            results['normality_tests'] = {
                'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p, 'is_normal': shapiro_p > 0.05}
            }

            # 如果樣本數量足夠，進行更多檢驗
            if len(target_values) >= 8:
                dagostino_stat, dagostino_p = normaltest(target_values)
                results['normality_tests']['dagostino'] = {
                    'statistic': dagostino_stat, 'p_value': dagostino_p, 'is_normal': dagostino_p > 0.05
                }

            # 參數效果的統計檢驗
            numeric_params = self.df.select_dtypes(include=[np.number]).columns
            numeric_params = [p for p in numeric_params if p != target_metric]

            param_effects = {}
            for param in numeric_params:
                if self.df[param].nunique() > 1:
                    try:
                        # 將連續參數分組進行方差分析
                        param_values = self.df[param].dropna()
                        if len(param_values.unique()) <= 5:  # 離散參數
                            groups = [self.df[self.df[param] == val][target_metric].values
                                    for val in param_values.unique()]
                            groups = [g for g in groups if len(g) > 0]

                            if len(groups) > 1:
                                from scipy.stats import f_oneway, kruskal
                                f_stat, f_p = f_oneway(*groups)
                                h_stat, h_p = kruskal(*groups)

                                param_effects[param] = {
                                    'anova_f_test': {'f_statistic': f_stat, 'p_value': f_p, 'significant': f_p < 0.05},
                                    'kruskal_wallis': {'h_statistic': h_stat, 'p_value': h_p, 'significant': h_p < 0.05}
                                }
                        else:  # 連續參數
                            from scipy.stats import pearsonr
                            corr_coef, corr_p = pearsonr(self.df[param], self.df[target_metric])
                            param_effects[param] = {
                                'pearson_correlation': {
                                    'correlation': corr_coef,
                                    'p_value': corr_p,
                                    'significant': corr_p < 0.05
                                }
                            }
                    except Exception as e:
                        logger.error(f"分析參數 {param} 時發生錯誤: {e}")

            results['parameter_effects'] = param_effects

        except Exception as e:
            logger.error(f"進階統計檢驗時發生錯誤: {e}")

        return results

    def performance_comparison_report(self, group_by: str = None) -> Dict[str, Any]:
        """
        效能對比報告

        Args:
            group_by: 分組依據的參數

        Returns:
            對比報告
        """
        if self.df.empty:
            return {}

        report = {
            "overall_statistics": {},
            "group_comparison": {}
        }

        # 整體統計
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = self.df[col].dropna()
            if not values.empty:
                report["overall_statistics"][col] = {
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "count": int(len(values))
                }

        # 分組比較
        if group_by and group_by in self.df.columns:
            group_stats = self.df.groupby(group_by).agg({
                col: ['mean', 'std', 'count'] for col in numeric_cols if col != group_by
            }).round(4)

            report["group_comparison"] = group_stats.to_dict()

        return report

    def generate_analysis_report(self, output_dir: str = "reports") -> str:
        """
        生成完整的分析報告

        Args:
            output_dir: 輸出目錄

        Returns:
            報告檔案路徑
        """
        os.makedirs(output_dir, exist_ok=True)

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_tests": len(self.results),
            "parameter_sensitivity": self.parameter_sensitivity_analysis(),
            "optimal_parameters": self.find_optimal_parameters(),
            "performance_comparison": self.performance_comparison_report(),
            "recommendations": self._generate_recommendations()
        }

        # 保存 JSON 報告
        report_path = os.path.join(output_dir, f"analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # 生成圖表
        self._generate_visualizations(output_dir)

        return report_path

    def _generate_recommendations(self) -> List[str]:
        """生成分析建議"""
        recommendations = []

        # 基於敏感度分析的建議
        sensitivity = self.parameter_sensitivity_analysis()
        if sensitivity:
            significant_params = [p for p, v in sensitivity.items() if v.get("significant", False)]
            if significant_params:
                recommendations.append(f"關鍵參數: {', '.join(significant_params)} - 這些參數對效能有顯著影響")

        # 基於最佳參數的建議
        optimal = self.find_optimal_parameters()
        if optimal:
            recommendations.append(f"建議參數組合: {optimal['optimal_parameters']}")

        return recommendations

    def _generate_visualizations(self, output_dir: str):
        """生成可視化圖表"""
        try:
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            # 參數敏感度熱力圖
            sensitivity = self.parameter_sensitivity_analysis()
            if sensitivity:
                self._plot_sensitivity_heatmap(sensitivity, output_dir)

            # 效能分佈圖
            self._plot_performance_distribution(output_dir)

            # 參數交互作用熱力圖
            self._plot_parameter_interaction_heatmap(output_dir)

            # 參數效能散點圖
            self._plot_parameter_performance_scatter(output_dir)

            # 效能趨勢圖
            self._plot_performance_trends(output_dir)

        except Exception as e:
            logger.error(f"生成可視化時發生錯誤: {e}")

    def _plot_parameter_interaction_heatmap(self, output_dir: str):
        """繪製參數交互作用熱力圖"""
        if self.df.empty:
            return

        numeric_params = self.df.select_dtypes(include=[np.number]).columns
        numeric_params = [p for p in numeric_params if p not in ['avg_fps', 'duration', 'frames_processed']]

        if len(numeric_params) < 2:
            return

        try:
            # 創建參數相關性矩陣
            corr_matrix = self.df[numeric_params + ['avg_fps']].corr()

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.3f')
            plt.title('參數相關性熱力圖')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'parameter_correlation_heatmap.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"生成參數交互作用熱力圖時發生錯誤: {e}")

    def _plot_parameter_performance_scatter(self, output_dir: str):
        """繪製參數-效能散點圖"""
        if self.df.empty or 'avg_fps' not in self.df.columns:
            return

        numeric_params = self.df.select_dtypes(include=[np.number]).columns
        numeric_params = [p for p in numeric_params if p not in ['avg_fps', 'duration', 'frames_processed']]

        if not numeric_params:
            return

        # 創建子圖
        n_params = len(numeric_params)
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_params == 1:
            axes = [axes]
        elif rows == 1:
            pass
        else:
            axes = axes.flatten()

        for i, param in enumerate(numeric_params):
            if i < len(axes):
                ax = axes[i] if n_params > 1 else axes[0]
                clean_data = self.df[[param, 'avg_fps']].dropna()
                if not clean_data.empty:
                    ax.scatter(clean_data[param], clean_data['avg_fps'], alpha=0.7)
                    ax.set_xlabel(param)
                    ax.set_ylabel('平均 FPS')
                    ax.set_title(f'{param} vs 平均 FPS')
                    ax.grid(True, alpha=0.3)

        # 隱藏多餘的子圖
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_performance_scatter.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_trends(self, output_dir: str):
        """繪製效能趨勢圖"""
        if self.df.empty or 'avg_fps' not in self.df.columns:
            return

        # 假設有測試順序或時間戳
        fps_data = self.df['avg_fps'].dropna()
        if len(fps_data) < 2:
            return

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(fps_data)), fps_data, 'o-', linewidth=2, markersize=6)
        plt.xlabel('測試順序')
        plt.ylabel('平均 FPS')
        plt.title('測試效能趨勢')
        plt.grid(True, alpha=0.3)

        # 添加趨勢線
        if len(fps_data) > 1:
            z = np.polyfit(range(len(fps_data)), fps_data, 1)
            p = np.poly1d(z)
            plt.plot(range(len(fps_data)), p(range(len(fps_data))), "--",
                    alpha=0.7, label=f'趨勢線 (斜率: {z[0]:.4f})')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_trends.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_sensitivity_heatmap(self, sensitivity: Dict[str, Any], output_dir: str):
        """繪製參數敏感度熱力圖"""
        params = list(sensitivity.keys())
        correlations = [v.get("correlation", 0) for v in sensitivity.values()]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(params, correlations)
        plt.xlabel('相關係數')
        plt.ylabel('參數')
        plt.title('參數敏感度分析')
        plt.grid(True, alpha=0.3)

        # 添加數值標籤
        for i, (param, corr) in enumerate(zip(params, correlations)):
            plt.text(corr + 0.01 if corr >= 0 else corr - 0.01,
                    i, f'{corr:.3f}', ha='left' if corr >= 0 else 'right', va='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_distribution(self, output_dir: str):
        """繪製效能分佈圖"""
        if 'avg_fps' not in self.df.columns:
            return

        fps_data = self.df['avg_fps'].dropna()
        if fps_data.empty:
            return

        plt.figure(figsize=(10, 6))
        plt.hist(fps_data, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('平均 FPS')
        plt.ylabel('測試次數')
        plt.title('效能分佈')
        plt.grid(True, alpha=0.3)

        # 添加統計線
        mean_fps = fps_data.mean()
        median_fps = fps_data.median()
        plt.axvline(mean_fps, color='red', linestyle='--', label=f'平均: {mean_fps:.2f}')
        plt.axvline(median_fps, color='green', linestyle='--', label=f'中位數: {median_fps:.2f}')

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
