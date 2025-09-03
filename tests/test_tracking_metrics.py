#!/usr/bin/env python3
"""
測試追蹤指標分析器的腳本
"""

from ultralytics_mot_benchmark.analysis.tracking_metrics_analyzer import TrackingMetricsAnalyzer
import json
from loguru import logger

def test_tracking_metrics_analyzer():
    """測試追蹤指標分析器"""
    logger.info("開始測試追蹤指標分析器...")

    # 創建分析器實例
    analyzer = TrackingMetricsAnalyzer()

    # 模擬一些追蹤數據
    # 模擬場景：3個軌跡，分別持續不同長度，有一些間隙

    # 幀 0: 軌跡 1 和 2 開始
    frame_0_tracks = [
        {'track_id': 1, 'bbox': (100, 100, 150, 200), 'confidence': 0.9},
        {'track_id': 2, 'bbox': (200, 100, 250, 200), 'confidence': 0.8}
    ]
    analyzer.update_frame(0, frame_0_tracks)

    # 幀 1: 軌跡 1 和 2 繼續
    frame_1_tracks = [
        {'track_id': 1, 'bbox': (105, 105, 155, 205), 'confidence': 0.9},
        {'track_id': 2, 'bbox': (205, 105, 255, 205), 'confidence': 0.8}
    ]
    analyzer.update_frame(1, frame_1_tracks)

    # 幀 2: 軌跡 1 繼續，軌跡 2 中斷，軌跡 3 開始
    frame_2_tracks = [
        {'track_id': 1, 'bbox': (110, 110, 160, 210), 'confidence': 0.9},
        {'track_id': 3, 'bbox': (300, 100, 350, 200), 'confidence': 0.7}
    ]
    analyzer.update_frame(2, frame_2_tracks)

    # 幀 3: 軌跡 1 和 3 繼續
    frame_3_tracks = [
        {'track_id': 1, 'bbox': (115, 115, 165, 215), 'confidence': 0.9},
        {'track_id': 3, 'bbox': (305, 105, 355, 205), 'confidence': 0.7}
    ]
    analyzer.update_frame(3, frame_3_tracks)

    # 幀 4: 軌跡 2 重新出現（模擬 ID 切換）
    frame_4_tracks = [
        {'track_id': 1, 'bbox': (120, 120, 170, 220), 'confidence': 0.9},
        {'track_id': 4, 'bbox': (210, 110, 260, 210), 'confidence': 0.8},  # 應該檢測為軌跡 2 的 ID 切換
        {'track_id': 3, 'bbox': (310, 110, 360, 210), 'confidence': 0.7}
    ]
    analyzer.update_frame(4, frame_4_tracks)

    # 獲取指標
    metrics = analyzer.get_tracking_metrics()

    logger.info("\n=== 追蹤指標分析結果 ===")
    logger.info(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 驗證基本指標
    assert metrics['total_tracks'] == 4, f"預期總軌跡數為 4，實際為 {metrics['total_tracks']}"
    assert metrics['frames_processed'] == 4, f"預期處理幀數為 4，實際為 {metrics['frames_processed']}"

    # 驗證連續性指標
    continuity = metrics['continuity_metrics']
    assert 'continuity_ratio' in continuity, "缺少連續性比率指標"
    assert continuity['continuity_ratio'] == 1.0, "在這個測試場景中應該沒有間隙"

    # 驗證片段化指標
    fragmentation = metrics['fragmentation_metrics']
    assert 'fragmentation_ratio' in fragmentation, "缺少片段化比率指標"

    # 驗證新軌跡頻率
    new_track_freq = metrics['new_track_frequency']
    assert 'total_new_tracks' in new_track_freq, "缺少新軌跡總數指標"
    assert new_track_freq['total_new_tracks'] == 0, f"預期新軌跡總數為 0，實際為 {new_track_freq['total_new_tracks']}"  # 第一幀不算新軌跡

    # 驗證 ID 切換
    id_switches = metrics['id_switch_stats']
    assert 'total_id_switches' in id_switches, "缺少 ID 切換總數指標"

    logger.success("✅ 所有測試通過！追蹤指標分析器工作正常。")

    return metrics

if __name__ == "__main__":
    # 配置 loguru
    logger.remove()  # 移除預設配置
    logger.add(
        lambda msg: print(msg, end=""),  # 輸出到控制台
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # 添加檔案日誌
    logger.add(
        "logs/test_tracking_metrics_{time}.log",
        rotation="1 day",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
        retention="7 days"
    )

    logger.info("開始執行追蹤指標分析器測試")
    test_tracking_metrics_analyzer()
    logger.info("測試完成")
