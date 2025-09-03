"""
測試改進的ID切換檢測算法
模擬用戶描述的場景並驗證檢測效果
"""

import json
from ultralytics_mot_benchmark.analysis.tracking_quality_analyzer import TrackingQualityAnalyzer
from ultralytics_mot_benchmark.analysis.tracking_metrics_analyzer import TrackingMetricsAnalyzer


def simulate_user_scenario():
    """
    模擬用戶描述的ID切換場景：
    1. ID 1出現一下就切換成ID 2（同一個人）
    2. ID 3經過障礙物時瞬間切換成ID 4後又變回ID 3
    3. ID 3和ID 2擦身而過時，ID 3切換成ID 5然後又變成ID 6
    """
    analyzer = TrackingMetricsAnalyzer()

    # 模擬場景1: ID 1 -> ID 2 的快速切換
    print("=== 模擬場景1: ID 1 -> ID 2 的快速切換 ===")

    # Frame 1: ID 1 出現在位置 (100, 100, 150, 200)
    analyzer.update_frame(1, [{
        'track_id': 1,
        'bbox': (100, 100, 150, 200),
        'confidence': 0.9
    }])

    # Frame 2: ID 1 消失，ID 2 出現在相似位置 (105, 105, 155, 205)
    analyzer.update_frame(2, [{
        'track_id': 2,
        'bbox': (105, 105, 155, 205),
        'confidence': 0.9
    }])

    # Frame 3: 繼續ID 2
    analyzer.update_frame(3, [{
        'track_id': 2,
        'bbox': (110, 110, 160, 210),
        'confidence': 0.9
    }])

    # 模擬場景2: ID 3 -> ID 4 -> ID 3 的障礙物切換
    print("\n=== 模擬場景2: ID 3 -> ID 4 -> ID 3 的障礙物切換 ===")

    # Frame 4: ID 3 出現在位置 (200, 100, 250, 200)
    analyzer.update_frame(4, [{
        'track_id': 3,
        'bbox': (200, 100, 250, 200),
        'confidence': 0.9
    }])

    # Frame 5: ID 3 經過障礙物，位置變化較大，系統給了新ID 4 (220, 120, 270, 220)
    analyzer.update_frame(5, [{
        'track_id': 4,
        'bbox': (220, 120, 270, 220),
        'confidence': 0.8
    }])

    # Frame 6: 障礙物過去，ID 4 變回 ID 3 (225, 125, 275, 225)
    analyzer.update_frame(6, [{
        'track_id': 3,
        'bbox': (225, 125, 275, 225),
        'confidence': 0.9
    }])

    # 模擬場景3: 多重ID切換
    print("\n=== 模擬場景3: ID 3 和 ID 2 擦身而過時的多重切換 ===")

    # Frame 7: ID 2 和 ID 3 在接近 (110, 110, 160, 210) 和 (225, 125, 275, 225)
    analyzer.update_frame(7, [
        {'track_id': 2, 'bbox': (110, 110, 160, 210), 'confidence': 0.9},
        {'track_id': 3, 'bbox': (225, 125, 275, 225), 'confidence': 0.9}
    ])

    # Frame 8: 擦身而過，ID 3 被誤認為新軌跡，給了ID 5 (230, 130, 280, 230)
    analyzer.update_frame(8, [
        {'track_id': 2, 'bbox': (115, 115, 165, 215), 'confidence': 0.9},
        {'track_id': 5, 'bbox': (230, 130, 280, 230), 'confidence': 0.7}
    ])

    # Frame 9: 系統重新識別，ID 5 變成 ID 6 (235, 135, 285, 235)
    analyzer.update_frame(9, [
        {'track_id': 2, 'bbox': (120, 120, 170, 220), 'confidence': 0.9},
        {'track_id': 6, 'bbox': (235, 135, 285, 235), 'confidence': 0.8}
    ])

    # 獲取追蹤指標
    metrics = analyzer.get_tracking_metrics()

    print("\n=== 檢測到的ID切換 ===")
    print(f"總ID切換次數: {len(analyzer.id_switches)}")
    for i, switch in enumerate(analyzer.id_switches, 1):
        print(f"切換 {i}: 幀 {switch['frame_id']} - ID {switch['previous_track_id']} -> ID {switch['current_track_id']} "
              ".3f")

    print("\n=== 追蹤指標總結 ===")
    print(f"總軌跡數: {metrics['total_tracks']}")
    print(f"軌跡長度: {metrics['track_lengths']}")
    print(f"ID切換統計: {metrics['id_switch_stats']}")

    return metrics


def analyze_real_data(json_path: str):
    """
    分析真實的追蹤數據
    """
    print(f"\n=== 分析真實數據: {json_path} ===")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 使用改進的品質分析器
        quality_analyzer = TrackingQualityAnalyzer()
        quality_score = quality_analyzer.analyze_tracking_quality(data)

        print("改進後的品質評分:")
        print(f"總體評分: {quality_score.overall_score:.2f}")
        print(f"身份穩定性評分: {quality_score.stability_score:.2f}")
        print(f"軌跡連續性評分: {quality_score.continuity_score:.2f}")
        print(f"碎片化評分: {quality_score.fragmentation_score:.2f}")

        # 顯示建議
        interpretation = quality_analyzer.get_quality_interpretation(quality_score)
        print("\n改進建議:")
        for rec in interpretation['recommendations']:
            print(f"- {rec}")

        return quality_score

    except Exception as e:
        print(f"分析失敗: {e}")
        return None


if __name__ == "__main__":
    print("=== ID切換檢測改進測試 ===\n")

    # 1. 模擬用戶描述的場景
    print("1. 模擬用戶描述的ID切換場景:")
    simulated_metrics = simulate_user_scenario()

    # 2. 分析真實數據
    print("\n2. 分析真實追蹤數據:")
    real_quality = analyze_real_data("outputs/videos/camera-d8accaf0_10s__20250901T090023265980Z_37ad9de4.json")

    print("\n=== 總結 ===")
    print("改進的ID切換檢測算法應該能夠更好地識別用戶描述的問題。")
    print("建議重新運行追蹤測試來驗證改進效果。")
