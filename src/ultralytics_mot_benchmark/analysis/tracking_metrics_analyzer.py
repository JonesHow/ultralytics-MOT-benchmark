"""
追蹤指標分析器
負責分析追蹤結果並計算各種追蹤品質指標
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
from dataclasses import dataclass
from loguru import logger


@dataclass
class TrackInfo:
    """軌跡資訊"""
    track_id: int
    start_frame: int
    end_frame: int
    frame_count: int
    consecutive_frames: int
    max_gap: int
    gaps: List[int]  # 軌跡中斷的幀數列表
    bbox_history: List[Tuple[float, float, float, float]]  # (x1, y1, x2, y2)


@dataclass
class FrameTrackData:
    """每幀的追蹤數據"""
    frame_id: int
    track_ids: List[int]
    bboxes: List[Tuple[float, float, float, float]]
    confidences: List[float]


class TrackingMetricsAnalyzer:
    """追蹤指標分析器"""

    def __init__(self, max_track_history: int = 1000):
        """
        初始化追蹤指標分析器

        Args:
            max_track_history: 最大追蹤歷史記錄長度
        """
        self.track_history: Dict[int, TrackInfo] = {}
        self.frame_data: List[FrameTrackData] = []
        self.id_switches: List[Dict[str, Any]] = []
        self.new_tracks_per_frame: List[int] = []
        self.active_tracks_per_frame: List[int] = []

        # 用於計算軌跡連續性
        self.track_continuity_buffer = defaultdict(deque)
        self.max_track_history = max_track_history

        # 追蹤狀態
        self.last_frame_tracks: Dict[int, Tuple[float, float, float, float]] = {}
        self.frame_count = 0
        # 短期消失/重現的暫存，用於跨幀匹配（處理遮擋、瞬斷）
        self._recently_disappeared: Dict[int, Dict[str, Any]] = {}
        self._max_disappear_gap = 3  # 允許最多跨 3 幀內的重現

    def update_frame(self, frame_id: int, track_results: List[Dict[str, Any]]) -> None:
        """
        更新一幀的追蹤結果

        Args:
            frame_id: 幀編號
            track_results: 追蹤結果列表，每個元素包含 track_id, bbox, confidence 等
        """
        self.frame_count = frame_id

        # 解析追蹤結果
        current_track_ids = []
        current_bboxes = []
        current_confidences = []

        for result in track_results:
            if 'track_id' in result and result['track_id'] is not None:
                track_id = int(result['track_id'])
                bbox = tuple(result['bbox'])  # (x1, y1, x2, y2)
                confidence = result.get('confidence', 1.0)

                current_track_ids.append(track_id)
                current_bboxes.append(bbox)
                current_confidences.append(confidence)

                # 更新軌跡歷史
                self._update_track_history(track_id, frame_id, bbox)

        # 記錄幀數據
        frame_track_data = FrameTrackData(
            frame_id=frame_id,
            track_ids=current_track_ids,
            bboxes=current_bboxes,
            confidences=current_confidences
        )
        self.frame_data.append(frame_track_data)

        # 計算新軌跡數量
        new_tracks = self._calculate_new_tracks(current_track_ids)
        self.new_tracks_per_frame.append(new_tracks)

        # 記錄活躍軌跡數量
        self.active_tracks_per_frame.append(len(current_track_ids))

        # 檢測身份切換（基於上一幀與當前幀的關聯匹配 + 短期消失/重現匹配）
        self._detect_id_switches(frame_id, current_track_ids, current_bboxes)

        # 更新上一幀的追蹤狀態
        self.last_frame_tracks = dict(zip(current_track_ids, current_bboxes))

    def _update_track_history(self, track_id: int, frame_id: int, bbox: Tuple[float, float, float, float]) -> None:
        """更新軌跡歷史記錄"""
        if track_id not in self.track_history:
            # 新軌跡
            self.track_history[track_id] = TrackInfo(
                track_id=track_id,
                start_frame=frame_id,
                end_frame=frame_id,
                frame_count=1,
                consecutive_frames=1,
                max_gap=0,
                gaps=[],
                bbox_history=[bbox]
            )
        else:
            # 現有軌跡
            track_info = self.track_history[track_id]

            # 檢查是否有間隙
            gap = frame_id - track_info.end_frame - 1
            if gap > 0:
                track_info.gaps.append(gap)
                track_info.max_gap = max(track_info.max_gap, gap)
                track_info.consecutive_frames = 1  # 重置連續幀數
            else:
                track_info.consecutive_frames += 1

            track_info.end_frame = frame_id
            track_info.frame_count += 1
            track_info.bbox_history.append(bbox)

            # 限制歷史記錄長度
            if len(track_info.bbox_history) > self.max_track_history:
                track_info.bbox_history.pop(0)

    def _calculate_new_tracks(self, current_track_ids: List[int]) -> int:
        """計算當前幀的新軌跡數量"""
        if not self.frame_data:
            return len(current_track_ids)

        previous_track_ids = set(self.frame_data[-1].track_ids)
        current_track_ids_set = set(current_track_ids)

        new_tracks = current_track_ids_set - previous_track_ids
        return len(new_tracks)

    def _detect_id_switches(self, frame_id: int, current_track_ids: List[int],
                           current_bboxes: List[Tuple[float, float, float, float]]) -> None:
        """
        基於上一幀與當前幀的IoU關聯來檢測ID切換。
        僅在空間對應關係改變時計一次，避免逐幀重複累計。
        """
        if not self.last_frame_tracks and not current_track_ids:
            return

        # 取得上一幀的 ID 與 bbox 列表
        last_ids = list(self.last_frame_tracks.keys())
        last_bboxes = [self.last_frame_tracks[i] for i in last_ids]

        # 如果任一側為空，無需處理
        if not last_ids or not current_track_ids:
            # 當前幀無 ID 或上一幀無 ID：更新消失/重現暫存並退出
            self._update_disappear_appear_buffers(frame_id, last_ids, current_track_ids, last_bboxes, current_bboxes)
            return

        # 構建所有候選配對 (last_id, curr_id, iou)
        candidates: List[Tuple[int, int, float]] = []
        for li, lb in zip(last_ids, last_bboxes):
            for ci, cb in zip(current_track_ids, current_bboxes):
                iou = self._calculate_iou(lb, cb)
                if iou > 0:  # 先過濾掉完全無交集者
                    candidates.append((li, ci, iou))

        if not candidates:
            # 沒有空間重疊，嘗試用短期消失/重現匹配檢測切換
            self._update_disappear_appear_buffers(frame_id, last_ids, current_track_ids, last_bboxes, current_bboxes)
            self._detect_switch_by_disappear_appear(frame_id, current_track_ids, current_bboxes)
            return

        # 依 IoU 由高到低排序，做貪婪匹配
        candidates.sort(key=lambda x: x[2], reverse=True)
        matched_last = set()
        matched_curr = set()

        # IoU 閾值：過低的重疊不視為同一實體的延續
        iou_thresh = 0.1

        for li, ci, iou in candidates:
            if iou < iou_thresh:
                break
            if li in matched_last or ci in matched_curr:
                continue

            # 配對確立：上一幀的 li 與當前幀的 ci 視為同一實體的延續
            matched_last.add(li)
            matched_curr.add(ci)

            # 若 ID 不同，判定為一次 ID 切換事件（僅計一次）
            if li != ci:
                id_switch = {
                    'frame_id': frame_id,
                    'previous_track_id': li,
                    'current_track_id': ci,
                    'similarity_score': float(iou),
                    'iou': float(iou),
                    'previous_bbox': self.last_frame_tracks[li],
                    'current_bbox': current_bboxes[current_track_ids.index(ci)],
                    'detection_method': 'association_iou'
                }
                self.id_switches.append(id_switch)
                logger.debug(f"檢測到ID切換(關聯): {id_switch}")

        # 可選：對於上一幀存在但當前缺失，且當前附近新生 ID 的情況，
        # 如需跨幀間隙關聯，可在此擴展（此處先保持簡潔，避免過度計數）。
        self._update_disappear_appear_buffers(frame_id, last_ids, current_track_ids, last_bboxes, current_bboxes)
        self._detect_switch_by_disappear_appear(frame_id, current_track_ids, current_bboxes)

    def _update_disappear_appear_buffers(self, frame_id: int,
                                         last_ids: List[int], current_ids: List[int],
                                         last_bboxes: List[Tuple[float, float, float, float]],
                                         current_bboxes: List[Tuple[float, float, float, float]]) -> None:
        """更新短期消失/重現暫存，並清理過舊資料"""
        prev_set = set(last_ids)
        curr_set = set(current_ids)

        disappeared = list(prev_set - curr_set)
        appeared = list(curr_set - prev_set)

        # 新消失的加入暫存
        for did in disappeared:
            self._recently_disappeared[did] = {
                'frame_id': frame_id - 1,
                'bbox': self.last_frame_tracks.get(did)
            }

        # 清理過舊的暫存
        to_delete = []
        for did, info in self._recently_disappeared.items():
            if frame_id - info['frame_id'] > self._max_disappear_gap:
                to_delete.append(did)
        for did in to_delete:
            self._recently_disappeared.pop(did, None)

    def _detect_switch_by_disappear_appear(self, frame_id: int,
                                           current_ids: List[int],
                                           current_bboxes: List[Tuple[float, float, float, float]]) -> None:
        """
        嘗試將「剛消失的舊ID」與「剛出現的新ID」進行空間鄰近關聯，
        用於處理遮擋或瞬斷造成的 ID 突變但相鄰時空位置仍接近的情形。
        """
        if not self._recently_disappeared:
            return

        # 準備出現集合（只考慮當前幀的新出現 ID）
        prev_set = set(self.last_frame_tracks.keys())
        curr_set = set(current_ids)
        appeared = list(curr_set - prev_set)
        if not appeared:
            return

        # 建立候選：每個 disappeared 對 appeared，使用中心距離 + IoU 作為綜合指標
        candidates: List[Tuple[int, int, float]] = []  # (old_id, new_id, score)
        for old_id, info in self._recently_disappeared.items():
            old_bbox = info.get('bbox')
            if old_bbox is None:
                continue
            for new_id in appeared:
                idx = current_ids.index(new_id)
                new_bbox = current_bboxes[idx]
                iou = self._calculate_iou(old_bbox, new_bbox)
                dist = self._calculate_bbox_distance(old_bbox, new_bbox)
                # 分數：偏向重疊和距離近
                score = iou * 0.7 + (1 - dist) * 0.3
                candidates.append((old_id, new_id, score))

        if not candidates:
            return

        # 按分數從高到低貪婪匹配
        candidates.sort(key=lambda x: x[2], reverse=True)
        used_old = set()
        used_new = set()
        score_thresh = 0.3  # 可調: 降低可更敏感

        for old_id, new_id, score in candidates:
            if score < score_thresh:
                break
            if old_id in used_old or new_id in used_new:
                continue
            used_old.add(old_id)
            used_new.add(new_id)

            # 計一次 ID 切換事件
            idx = current_ids.index(new_id)
            id_switch = {
                'frame_id': frame_id,
                'previous_track_id': int(old_id),
                'current_track_id': int(new_id),
                'similarity_score': float(score),
                'iou': float(self._calculate_iou(self._recently_disappeared[old_id]['bbox'], current_bboxes[idx])),
                'previous_bbox': self._recently_disappeared[old_id]['bbox'],
                'current_bbox': current_bboxes[idx],
                'detection_method': 'disappear_appear'
            }
            self.id_switches.append(id_switch)
            logger.debug(f"檢測到ID切換(遮擋/瞬斷): {id_switch}")

            # 一旦匹配成功，從暫存移除舊ID，避免後續重複計數
            self._recently_disappeared.pop(old_id, None)

    # 保留接口占位，但不再使用過於寬鬆的逐對計分方式，以避免嚴重高估
    def _detect_id_switches_by_motion(self, frame_id: int, current_track_ids: List[int],
                                    current_bboxes: List[Tuple[float, float, float, float]]) -> None:
        return

    def _detect_id_switches_by_trajectory(self, frame_id: int, current_track_ids: List[int],
                                        current_bboxes: List[Tuple[float, float, float, float]]) -> None:
        """
        基於軌跡歷史檢測ID切換
        """
        # 檢查軌跡歷史，找出可能被中斷的軌跡
        for track_id, bbox in zip(current_track_ids, current_bboxes):
            if track_id in self.track_history:
                track_info = self.track_history[track_id]

                # 如果這個軌跡之前有間隙，檢查是否可能是ID切換的結果
                if track_info.gaps and len(track_info.gaps) > 0:
                    # 檢查最近的間隙
                    last_gap = track_info.gaps[-1]
                    if last_gap <= 5:  # 最近5幀內有間隙
                        logger.debug(f"軌跡 {track_id} 最近有間隙，可能存在ID切換")

    def _detect_id_switches_by_appearance(self, frame_id: int, current_track_ids: List[int],
                                        current_bboxes: List[Tuple[float, float, float, float]]) -> None:
        """
        基於外觀相似度檢測ID切換（預留接口）
        """
        # TODO: 如果有ReID特徵，可以在這裡實現基於外觀的ID切換檢測
        pass

    def _calculate_bbox_distance(self, bbox1: Tuple[float, float, float, float],
                               bbox2: Tuple[float, float, float, float]) -> float:
        """計算邊界框中心點距離（歸一化到0-1）"""
        # 計算中心點
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2

        # 計算距離
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

        # 歸一化（假設最大距離為畫面對角線的一半）
        max_distance = 500  # 可以根據實際畫面大小調整
        return min(distance / max_distance, 1.0)

    def _calculate_size_similarity(self, bbox1: Tuple[float, float, float, float],
                                bbox2: Tuple[float, float, float, float]) -> float:
        """計算邊界框大小相似度"""
        # 計算面積
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        if area1 == 0 or area2 == 0:
            return 0.0

        # 計算面積比例
        ratio = min(area1, area2) / max(area1, area2)
        return ratio

    def _calculate_iou(self, bbox1: Tuple[float, float, float, float],
                      bbox2: Tuple[float, float, float, float]) -> float:
        """計算兩個邊界框的 IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 計算交集
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # 計算聯集
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def get_tracking_metrics(self) -> Dict[str, Any]:
        """
        獲取追蹤指標統計結果

        Returns:
            包含各種追蹤指標的字典
        """
        if not self.track_history:
            return self._get_empty_metrics()

        # 基本軌跡統計
        total_tracks = len(self.track_history)
        track_lengths = [info.frame_count for info in self.track_history.values()]
        avg_track_length = np.mean(track_lengths) if track_lengths else 0
        max_track_length = max(track_lengths) if track_lengths else 0
        min_track_length = min(track_lengths) if track_lengths else 0

        # 軌跡連續性分析
        continuity_metrics = self._calculate_continuity_metrics()

        # 軌跡片段化分析
        fragmentation_metrics = self._calculate_fragmentation_metrics()

        # 新軌跡建立頻率
        new_track_frequency = self._calculate_new_track_frequency()

        # ID 切換統計
        id_switch_stats = self._calculate_id_switch_stats()

        return {
            'total_tracks': total_tracks,
            'avg_track_length': float(avg_track_length),
            'max_track_length': int(max_track_length),
            'min_track_length': int(min_track_length),
            'track_lengths': track_lengths,
            'continuity_metrics': continuity_metrics,
            'fragmentation_metrics': fragmentation_metrics,
            'new_track_frequency': new_track_frequency,
            'id_switch_stats': id_switch_stats,
            'frames_processed': self.frame_count,
            'avg_active_tracks_per_frame': np.mean(self.active_tracks_per_frame) if self.active_tracks_per_frame else 0
        }

    def _calculate_continuity_metrics(self) -> Dict[str, Any]:
        """計算軌跡連續性指標"""
        if not self.track_history:
            return {}

        total_gaps = 0
        total_gap_frames = 0
        max_gap = 0
        tracks_with_gaps = 0

        for track_info in self.track_history.values():
            if track_info.gaps:
                tracks_with_gaps += 1
                total_gaps += len(track_info.gaps)
                total_gap_frames += sum(track_info.gaps)
                max_gap = max(max_gap, track_info.max_gap)

        continuity_ratio = 1 - (total_gap_frames / sum(info.frame_count for info in self.track_history.values())) if self.track_history else 0

        return {
            'continuity_ratio': float(continuity_ratio),
            'total_gaps': total_gaps,
            'total_gap_frames': total_gap_frames,
            'max_gap': max_gap,
            'tracks_with_gaps': tracks_with_gaps,
            'tracks_without_gaps': len(self.track_history) - tracks_with_gaps
        }

    def _calculate_fragmentation_metrics(self) -> Dict[str, Any]:
        """計算軌跡片段化指標"""
        if not self.track_history:
            return {}

        # 片段化程度：軌跡數量相對於總幀數的比率
        fragmentation_ratio = len(self.track_history) / self.frame_count if self.frame_count > 0 else 0

        # 平均軌跡長度變異係數
        track_lengths = [info.frame_count for info in self.track_history.values()]
        if len(track_lengths) > 1:
            cv_track_length = np.std(track_lengths) / np.mean(track_lengths)
        else:
            cv_track_length = 0

        return {
            'fragmentation_ratio': float(fragmentation_ratio),
            'cv_track_length': float(cv_track_length),
            'short_tracks_ratio': len([l for l in track_lengths if l < 5]) / len(track_lengths) if track_lengths else 0
        }

    def _calculate_new_track_frequency(self) -> Dict[str, Any]:
        """計算新軌跡建立頻率"""
        if not self.new_tracks_per_frame:
            return {}

        total_new_tracks = sum(self.new_tracks_per_frame)
        avg_new_tracks_per_frame = np.mean(self.new_tracks_per_frame)
        max_new_tracks_per_frame = max(self.new_tracks_per_frame)

        return {
            'total_new_tracks': total_new_tracks,
            'avg_new_tracks_per_frame': float(avg_new_tracks_per_frame),
            'max_new_tracks_per_frame': int(max_new_tracks_per_frame),
            'new_tracks_per_frame': self.new_tracks_per_frame
        }

    def _calculate_id_switch_stats(self) -> Dict[str, Any]:
        """計算 ID 切換統計"""
        return {
            'total_id_switches': len(self.id_switches),
            'id_switches_per_frame': len(self.id_switches) / self.frame_count if self.frame_count > 0 else 0,
            'id_switch_details': self.id_switches[-100:] if len(self.id_switches) > 100 else self.id_switches  # 只保留最近的100個
        }

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """獲取空的指標結果"""
        return {
            'total_tracks': 0,
            'avg_track_length': 0,
            'max_track_length': 0,
            'min_track_length': 0,
            'track_lengths': [],
            'continuity_metrics': {},
            'fragmentation_metrics': {},
            'new_track_frequency': {},
            'id_switch_stats': {},
            'frames_processed': 0,
            'avg_active_tracks_per_frame': 0
        }

    def reset(self) -> None:
        """重置分析器狀態"""
        self.track_history.clear()
        self.frame_data.clear()
        self.id_switches.clear()
        self.new_tracks_per_frame.clear()
        self.active_tracks_per_frame.clear()
        self.last_frame_tracks.clear()
        self.frame_count = 0
