"""
擴展可視化模組
提供更豐富的圖表和分析視覺化
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly 不可用，將使用基礎可視化功能")


class AdvancedVisualizer:
    """進階可視化器"""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 設置樣式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def create_parameter_interaction_heatmap(self, df: pd.DataFrame,
                                           param1: str, param2: str,
                                           target_metric: str = "avg_fps") -> str:
        """創建參數交互作用熱力圖"""
        try:
            # 創建交互矩陣
            pivot_table = df.pivot_table(
                values=target_metric,
                index=param1,
                columns=param2,
                aggfunc='mean'
            )

            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table,
                       annot=True,
                       fmt='.2f',
                       cmap='RdYlBu_r',
                       center=pivot_table.mean().mean())

            plt.title(f'{param1} vs {param2} 對 {target_metric} 的影響')
            plt.xlabel(param2)
            plt.ylabel(param1)

            filepath = os.path.join(self.output_dir, f'interaction_{param1}_{param2}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"創建交互熱力圖時發生錯誤: {e}")
            return ""

    def create_performance_radar_chart(self, results: List[Dict[str, Any]]) -> str:
        """創建效能雷達圖"""
        if not PLOTLY_AVAILABLE:
            print("  ⚠️  Plotly 不可用，跳過雷達圖生成")
            return ""

        try:
            # 準備數據
            metrics = ['avg_fps', 'avg_inference_time', 'total_tracks', 'avg_track_length']

            # 找出最佳配置
            best_configs = []
            for result in results[:5]:  # 取前5個配置
                config_data = []
                params = result.get('parameters', {})
                perf = result.get('performance_metrics', {})
                track = result.get('tracking_metrics', {})

                # 標準化數據 (0-1)
                fps = perf.get('avg_fps', 0)
                inf_time = 1.0 / (perf.get('avg_inference_time', 1) + 0.001)  # 反轉，越小越好
                tracks = track.get('total_tracks', 0)
                track_len = track.get('avg_track_length', 0)

                config_data = [fps/50, inf_time/50, tracks/100, track_len/20]  # 標準化

                best_configs.append({
                    'name': f"Config {result.get('run_id', 'Unknown')[:8]}",
                    'values': config_data
                })

            # 創建雷達圖
            fig = go.Figure()

            for config in best_configs:
                fig.add_trace(go.Scatterpolar(
                    r=config['values'],
                    theta=metrics,
                    fill='toself',
                    name=config['name']
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="配置效能對比雷達圖"
            )

            filepath = os.path.join(self.output_dir, 'performance_radar.html')
            fig.write_html(filepath)

            return filepath

        except Exception as e:
            print(f"創建雷達圖時發生錯誤: {e}")
            return ""

    def create_parameter_correlation_matrix(self, df: pd.DataFrame) -> str:
        """創建參數相關性矩陣"""
        try:
            # 選擇數值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()

            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            sns.heatmap(corr_matrix,
                       mask=mask,
                       annot=True,
                       fmt='.2f',
                       center=0,
                       cmap='RdBu_r',
                       square=True)

            plt.title('參數相關性矩陣')
            plt.tight_layout()

            filepath = os.path.join(self.output_dir, 'correlation_matrix.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"創建相關性矩陣時發生錯誤: {e}")
            return ""

    def create_parameter_importance_plot(self, sensitivity_results: Dict[str, Any]) -> str:
        """創建參數重要性圖表"""
        try:
            params = list(sensitivity_results.keys())
            correlations = [abs(v.get('correlation', 0)) for v in sensitivity_results.values()]
            p_values = [v.get('p_value', 1) for v in sensitivity_results.values()]

            # 創建重要性分數 (相關性 * 顯著性)
            importance_scores = [corr * (1 - p_val) if p_val is not None else 0
                               for corr, p_val in zip(correlations, p_values)]

            # 排序
            sorted_data = sorted(zip(params, importance_scores, correlations, p_values),
                               key=lambda x: x[1], reverse=True)

            params_sorted = [x[0] for x in sorted_data]
            scores_sorted = [x[1] for x in sorted_data]

            plt.figure(figsize=(12, 8))
            bars = plt.barh(params_sorted, scores_sorted)

            # 根據重要性著色
            colors = ['red' if score > 0.5 else 'orange' if score > 0.2 else 'green'
                     for score in scores_sorted]
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            plt.xlabel('重要性分數')
            plt.ylabel('參數')
            plt.title('參數重要性排序')
            plt.grid(True, alpha=0.3)

            # 添加數值標籤
            for i, score in enumerate(scores_sorted):
                plt.text(score + 0.01, i, f'{score:.3f}',
                        va='center', ha='left')

            plt.tight_layout()

            filepath = os.path.join(self.output_dir, 'parameter_importance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"創建重要性圖表時發生錯誤: {e}")
            return ""

    def create_performance_trend_plot(self, results: List[Dict[str, Any]]) -> str:
        """創建效能趨勢圖"""
        try:
            # 按時間排序結果
            sorted_results = sorted(results,
                                  key=lambda x: x.get('execution_info', {}).get('start_time', ''))

            fps_data = []
            inference_data = []
            labels = []

            for i, result in enumerate(sorted_results):
                perf = result.get('performance_metrics', {})
                fps = perf.get('avg_fps')
                inf_time = perf.get('avg_inference_time')

                if fps is not None:
                    fps_data.append(fps)
                    inference_data.append(inf_time if inf_time else 0)
                    labels.append(f"Test {i+1}")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

            # FPS 趨勢
            ax1.plot(labels, fps_data, 'b-o', linewidth=2, markersize=6)
            ax1.set_ylabel('平均 FPS')
            ax1.set_title('效能趨勢分析')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)

            # 推理時間趨勢
            ax2.plot(labels, inference_data, 'r-s', linewidth=2, markersize=6)
            ax2.set_ylabel('平均推理時間 (秒)')
            ax2.set_xlabel('測試序號')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            filepath = os.path.join(self.output_dir, 'performance_trends.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"創建趨勢圖時發生錯誤: {e}")
            return ""

    def create_interactive_parameter_explorer(self, df: pd.DataFrame) -> str:
        """創建互動式參數探索器"""
        if not PLOTLY_AVAILABLE:
            print("  ⚠️  Plotly 不可用，創建基礎散點圖替代")
            return self._create_basic_scatter_plot(df)

        try:
            # 創建 3D 散點圖
            if len(df) < 3:
                return ""

            param_cols = [col for col in df.columns
                         if col not in ['avg_fps', 'avg_inference_time', 'duration', 'device']]

            if len(param_cols) >= 2:
                fig = px.scatter_3d(df,
                                   x=param_cols[0],
                                   y=param_cols[1] if len(param_cols) > 1 else param_cols[0],
                                   z='avg_fps' if 'avg_fps' in df.columns else df.columns[0],
                                   color='avg_fps' if 'avg_fps' in df.columns else None,
                                   size='avg_inference_time' if 'avg_inference_time' in df.columns else None,
                                   title='參數空間探索')

                fig.update_layout(
                    scene=dict(
                        xaxis_title=param_cols[0],
                        yaxis_title=param_cols[1] if len(param_cols) > 1 else param_cols[0],
                        zaxis_title='avg_fps' if 'avg_fps' in df.columns else df.columns[0]
                    )
                )

                filepath = os.path.join(self.output_dir, 'parameter_explorer.html')
                fig.write_html(filepath)

                return filepath

        except Exception as e:
            print(f"創建互動式探索器時發生錯誤: {e}")
            return ""

        return ""

    def _create_basic_scatter_plot(self, df: pd.DataFrame) -> str:
        """創建基礎散點圖作為替代"""
        try:
            param_cols = [col for col in df.columns
                         if col not in ['avg_fps', 'avg_inference_time', 'duration', 'device']]

            if len(param_cols) >= 2 and 'avg_fps' in df.columns:
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(df[param_cols[0]], df[param_cols[1]],
                                    c=df['avg_fps'], cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, label='avg_fps')
                plt.xlabel(param_cols[0])
                plt.ylabel(param_cols[1])
                plt.title('參數空間探索 (2D)')

                filepath = os.path.join(self.output_dir, 'parameter_explorer_2d.png')
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()

                return filepath
        except:
            pass
        return ""

    def create_comprehensive_dashboard(self, results: List[Dict[str, Any]],
                                     sensitivity_results: Dict[str, Any]) -> str:
        """創建綜合儀表板"""
        if not PLOTLY_AVAILABLE:
            print("  ⚠️  Plotly 不可用，創建基礎儀表板替代")
            return self._create_basic_dashboard(results, sensitivity_results)

        try:
            # 轉換為 DataFrame
            data = []
            for result in results:
                row = {}
                if "parameters" in result:
                    row.update(result["parameters"])
                if "performance_metrics" in result:
                    perf = result["performance_metrics"]
                    row.update({
                        "avg_fps": perf.get("avg_fps"),
                        "avg_inference_time": perf.get("avg_inference_time"),
                        "frames_processed": perf.get("frames_processed", 0)
                    })
                if "tracking_metrics" in result:
                    tracking = result["tracking_metrics"]
                    row.update({
                        "total_tracks": tracking.get("total_tracks", 0),
                        "avg_track_length": tracking.get("avg_track_length", 0)
                    })
                data.append(row)

            df = pd.DataFrame(data)

            # 創建子圖
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('FPS 分佈', '參數相關性', '效能對比', '追蹤統計'),
                specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )

            # FPS 分佈直方圖
            if 'avg_fps' in df.columns:
                fig.add_trace(
                    go.Histogram(x=df['avg_fps'], name="FPS 分佈"),
                    row=1, col=1
                )

            # 效能散點圖
            if 'avg_fps' in df.columns and 'avg_inference_time' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['avg_inference_time'],
                              y=df['avg_fps'],
                              mode='markers',
                              name="FPS vs 推理時間"),
                    row=2, col=1
                )

            # 追蹤統計條形圖
            if 'total_tracks' in df.columns:
                fig.add_trace(
                    go.Bar(x=list(range(len(df))),
                          y=df['total_tracks'],
                          name="總追蹤數"),
                    row=2, col=2
                )

            fig.update_layout(height=800, showlegend=False,
                             title_text="測試結果綜合儀表板")

            filepath = os.path.join(self.output_dir, 'comprehensive_dashboard.html')
            fig.write_html(filepath)

            return filepath

        except Exception as e:
            print(f"創建綜合儀表板時發生錯誤: {e}")
            return ""

    def _create_basic_dashboard(self, results: List[Dict[str, Any]],
                              sensitivity_results: Dict[str, Any]) -> str:
        """創建基礎儀表板作為替代"""
        try:
            # 轉換為 DataFrame
            data = []
            for result in results:
                row = {}
                if "parameters" in result:
                    row.update(result["parameters"])
                if "performance_metrics" in result:
                    perf = result["performance_metrics"]
                    row.update({
                        "avg_fps": perf.get("avg_fps"),
                        "avg_inference_time": perf.get("avg_inference_time")
                    })
                data.append(row)

            df = pd.DataFrame(data)

            if df.empty:
                return ""

            # 創建多子圖
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # FPS 分佈
            if 'avg_fps' in df.columns:
                ax1.hist(df['avg_fps'].dropna(), bins=10, alpha=0.7, edgecolor='black')
                ax1.set_title('FPS 分佈')
                ax1.set_xlabel('平均 FPS')
                ax1.set_ylabel('頻次')

            # 參數相關性
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax2)
                ax2.set_title('參數相關性')

            # 效能散點圖
            if 'avg_fps' in df.columns and 'avg_inference_time' in df.columns:
                ax3.scatter(df['avg_inference_time'], df['avg_fps'], alpha=0.6)
                ax3.set_xlabel('平均推理時間 (秒)')
                ax3.set_ylabel('平均 FPS')
                ax3.set_title('效能對比')

            # 參數重要性
            if sensitivity_results:
                params = list(sensitivity_results.keys())
                correlations = [abs(v.get('correlation', 0)) for v in sensitivity_results.values()]
                ax4.barh(params, correlations)
                ax4.set_title('參數重要性')
                ax4.set_xlabel('相關性 (絕對值)')

            plt.tight_layout()

            filepath = os.path.join(self.output_dir, 'basic_dashboard.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            return filepath

        except Exception as e:
            print(f"創建基礎儀表板時發生錯誤: {e}")
            return ""

    def generate_all_visualizations(self, results: List[Dict[str, Any]],
                                  sensitivity_results: Dict[str, Any]) -> Dict[str, str]:
        """生成所有可視化圖表"""
        generated_files = {}

        try:
            # 轉換為 DataFrame
            data = []
            for result in results:
                row = {}
                if "parameters" in result:
                    row.update(result["parameters"])
                if "performance_metrics" in result:
                    perf = result["performance_metrics"]
                    row.update({
                        "avg_fps": perf.get("avg_fps"),
                        "avg_inference_time": perf.get("avg_inference_time")
                    })
                data.append(row)

            df = pd.DataFrame(data)

            if not df.empty:
                # 生成各種圖表
                generated_files['correlation_matrix'] = self.create_parameter_correlation_matrix(df)
                generated_files['parameter_importance'] = self.create_parameter_importance_plot(sensitivity_results)
                generated_files['performance_trends'] = self.create_performance_trend_plot(results)
                generated_files['interactive_explorer'] = self.create_interactive_parameter_explorer(df)
                generated_files['comprehensive_dashboard'] = self.create_comprehensive_dashboard(results, sensitivity_results)

                # 如果有足夠的參數，創建交互圖
                param_cols = [col for col in df.columns
                             if col not in ['avg_fps', 'avg_inference_time']]
                if len(param_cols) >= 2:
                    generated_files['interaction_heatmap'] = self.create_parameter_interaction_heatmap(
                        df, param_cols[0], param_cols[1]
                    )

                # 創建雷達圖
                generated_files['radar_chart'] = self.create_performance_radar_chart(results)

        except Exception as e:
            print(f"生成可視化時發生錯誤: {e}")

        # 移除空的文件路徑
        return {k: v for k, v in generated_files.items() if v}


if __name__ == "__main__":
    # 測試可視化功能
    visualizer = AdvancedVisualizer("test_reports")
    print("進階可視化模組已就緒")
