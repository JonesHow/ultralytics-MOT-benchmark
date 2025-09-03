"""
軌跡品質綜合評估工具
用於評估追蹤參數的好壞，特別針對ID冗餘和軌跡碎片化問題
"""

import json
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrackingQualityScore:
    """追蹤品質評分"""
    overall_score: float  # 總體評分 (0-100)
    continuity_score: float  # 連續性評分
    fragmentation_score: float  # 碎片化評分
    efficiency_score: float  # 效率評分
    stability_score: float  # 穩定性評分

    def to_dict(self) -> Dict[str, float]:
        return {
            'overall_score': self.overall_score,
            'continuity_score': self.continuity_score,
            'fragmentation_score': self.fragmentation_score,
            'efficiency_score': self.efficiency_score,
            'stability_score': self.stability_score
        }


class TrackingQualityAnalyzer:
    """追蹤品質分析器"""

    def __init__(self):
        self.weights = {
            'continuity': 0.25,      # 軌跡連續性權重
            'fragmentation': 0.25,   # 碎片化程度權重
            'efficiency': 0.25,      # 推理效率權重
            'stability': 0.25        # 身份穩定性權重
        }

    def analyze_tracking_quality(self, metadata: Dict[str, Any]) -> TrackingQualityScore:
        """
        分析追蹤品質並給出綜合評分

        Args:
            metadata: 追蹤結果的metadata字典

        Returns:
            追蹤品質評分
        """
        tracking_metrics = metadata.get('tracking_metrics', {})

        # 計算各項評分
        continuity_score = self._calculate_continuity_score(tracking_metrics)
        fragmentation_score = self._calculate_fragmentation_score(tracking_metrics)
        efficiency_score = self._calculate_efficiency_score(metadata)
        stability_score = self._calculate_stability_score(tracking_metrics)

        # 計算總體評分
        overall_score = (
            continuity_score * self.weights['continuity'] +
            fragmentation_score * self.weights['fragmentation'] +
            efficiency_score * self.weights['efficiency'] +
            stability_score * self.weights['stability']
        )

        return TrackingQualityScore(
            overall_score=overall_score,
            continuity_score=continuity_score,
            fragmentation_score=fragmentation_score,
            efficiency_score=efficiency_score,
            stability_score=stability_score
        )

    def _calculate_continuity_score(self, tracking_metrics: Dict[str, Any]) -> float:
        """計算連續性評分 (0-100)"""
        continuity = tracking_metrics.get('continuity_metrics', {})

        continuity_ratio = continuity.get('continuity_ratio', 0)
        total_gaps = continuity.get('total_gaps', 0)
        tracks_with_gaps = continuity.get('tracks_with_gaps', 0)
        total_tracks = tracking_metrics.get('total_tracks', 1)

        # 連續性評分 = 連續性比率 * 100 - 間隙懲罰
        gap_penalty = min(20, (tracks_with_gaps / total_tracks) * 20) if total_tracks > 0 else 0

        score = continuity_ratio * 100 - gap_penalty
        return max(0, min(100, score))

    def _calculate_fragmentation_score(self, tracking_metrics: Dict[str, Any]) -> float:
        """計算碎片化評分 (0-100，越高越好)"""
        fragmentation = tracking_metrics.get('fragmentation_metrics', {})

        fragmentation_ratio = fragmentation.get('fragmentation_ratio', 1.0)
        cv_track_length = fragmentation.get('cv_track_length', 1.0)
        short_tracks_ratio = fragmentation.get('short_tracks_ratio', 1.0)

        # 碎片化評分 = (1 - 碎片化比率) * 100 - 變異懲罰
        variability_penalty = min(30, cv_track_length * 10)
        short_track_penalty = short_tracks_ratio * 20

        score = (1 - fragmentation_ratio) * 100 - variability_penalty - short_track_penalty
        return max(0, min(100, score))

    def _calculate_efficiency_score(self, metadata: Dict[str, Any]) -> float:
        """計算效率評分 (0-100)"""
        stats = metadata.get('stats', {})
        avg_inference_time = stats.get('avg_inference_time', 0.1)

        # 基準時間：30ms = 100分，100ms = 0分
        if avg_inference_time <= 0.03:  # 30ms
            score = 100
        elif avg_inference_time >= 0.1:  # 100ms
            score = 0
        else:
            # 線性插值
            score = 100 * (0.1 - avg_inference_time) / (0.1 - 0.03)

        return max(0, min(100, score))

    def _calculate_stability_score(self, tracking_metrics: Dict[str, Any]) -> float:
        """計算身份穩定性評分 (0-100)"""
        id_switch_stats = tracking_metrics.get('id_switch_stats', {})

        total_id_switches = id_switch_stats.get('total_id_switches', 0)
        id_switches_per_frame = id_switch_stats.get('id_switches_per_frame', 0)
        total_tracks = tracking_metrics.get('total_tracks', 1)

        # ID切換懲罰
        switch_penalty = min(50, total_id_switches * 5)
        per_frame_penalty = min(30, id_switches_per_frame * 100)

        # 軌跡數量懲罰（過多軌跡表示碎片化嚴重）
        track_penalty = min(20, max(0, (total_tracks - 3) * 2))  # 超過3個軌跡開始懲罰

        score = 100 - switch_penalty - per_frame_penalty - track_penalty
        return max(0, min(100, score))

    def get_quality_interpretation(self, score: TrackingQualityScore) -> Dict[str, Any]:
        """解釋品質評分的含義"""
        def interpret_single_score(score_value: float, score_name: str) -> str:
            if score_value >= 80:
                return f"優秀 ({score_name})"
            elif score_value >= 60:
                return f"良好 ({score_name})"
            elif score_value >= 40:
                return f"一般 ({score_name})"
            elif score_value >= 20:
                return f"較差 ({score_name})"
            else:
                return f"很差 ({score_name})"

        return {
            'overall_quality': interpret_single_score(score.overall_score, "總體品質"),
            'continuity_quality': interpret_single_score(score.continuity_score, "軌跡連續性"),
            'fragmentation_quality': interpret_single_score(score.fragmentation_score, "碎片化控制"),
            'efficiency_quality': interpret_single_score(score.efficiency_score, "推理效率"),
            'stability_quality': interpret_single_score(score.stability_score, "身份穩定性"),
            'recommendations': self._generate_recommendations(score)
        }

    def _generate_recommendations(self, score: TrackingQualityScore) -> List[str]:
        """基於評分生成改進建議"""
        recommendations = []

        if score.continuity_score < 60:
            recommendations.append("軌跡連續性較差，建議增加 track_buffer 參數值")

        if score.fragmentation_score < 60:
            recommendations.append("軌跡碎片化嚴重，建議調整 new_track_thresh 和 track_high_thresh")

        if score.efficiency_score < 60:
            recommendations.append("推理效率較低，考慮使用更小的模型或優化配置")

        if score.stability_score < 60:
            recommendations.append("身份穩定性不足，建議調整 appearance_thresh 和 proximity_thresh")

        if score.overall_score >= 80:
            recommendations.append("參數配置優秀，建議以此為基準進行微調優化")

        return recommendations if recommendations else ["參數配置均衡，建議保持當前設定"]


def analyze_tracking_result(json_path: str) -> Dict[str, Any]:
    """
    分析單個追蹤結果文件

    Args:
        json_path: JSON結果文件路徑

    Returns:
        分析結果字典
    """
    analyzer = TrackingQualityAnalyzer()

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # 計算品質評分
        quality_score = analyzer.analyze_tracking_quality(metadata)

        # 獲取解釋
        interpretation = analyzer.get_quality_interpretation(quality_score)

        # 提取關鍵指標
        tracking_metrics = metadata.get('tracking_metrics', {})
        stats = metadata.get('stats', {})

        result = {
            'file_path': json_path,
            'run_id': metadata.get('run_id', 'unknown'),
            'quality_scores': quality_score.to_dict(),
            'interpretation': interpretation,
            'key_metrics': {
                'total_tracks': tracking_metrics.get('total_tracks', 0),
                'avg_track_length': tracking_metrics.get('avg_track_length', 0),
                'continuity_ratio': tracking_metrics.get('continuity_metrics', {}).get('continuity_ratio', 0),
                'fragmentation_ratio': tracking_metrics.get('fragmentation_metrics', {}).get('fragmentation_ratio', 0),
                'id_switches': tracking_metrics.get('id_switch_stats', {}).get('total_id_switches', 0),
                'avg_fps': 1.0 / stats.get('avg_inference_time', 0.1) if stats.get('avg_inference_time', 0) > 0 else 0
            },
            'parameters': metadata.get('custom_tracker_config', {})
        }

        return result

    except Exception as e:
        return {
            'error': f'分析失敗: {str(e)}',
            'file_path': json_path
        }


def compare_multiple_results(json_paths: List[str]) -> Dict[str, Any]:
    """
    比較多個追蹤結果

    Args:
        json_paths: JSON文件路徑列表

    Returns:
        比較結果
    """
    analyzer = TrackingQualityAnalyzer()
    results = []

    for path in json_paths:
        result = analyze_tracking_result(path)
        if 'error' not in result:
            results.append(result)

    if not results:
        return {'error': '沒有有效的結果可以比較'}

    # 找出最佳結果
    best_result = max(results, key=lambda x: x['quality_scores']['overall_score'])

    # 計算平均值
    avg_scores = {}
    score_keys = ['overall_score', 'continuity_score', 'fragmentation_score', 'efficiency_score', 'stability_score']

    for key in score_keys:
        values = [r['quality_scores'][key] for r in results]
        avg_scores[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values)
        }

    return {
        'total_results': len(results),
        'best_result': best_result,
        'average_scores': avg_scores,
        'all_results': results,
        'ranking': sorted(results, key=lambda x: x['quality_scores']['overall_score'], reverse=True)
    }


if __name__ == "__main__":
    # 示例用法
    import sys

    if len(sys.argv) < 2:
        print("用法: python tracking_quality_analyzer.py <json_file_path>")
        sys.exit(1)

    json_path = sys.argv[1]
    result = analyze_tracking_result(json_path)

    print("=== 追蹤品質分析結果 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
