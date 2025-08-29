import os
import time
import json
import ast
import uuid
from datetime import datetime
import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from loguru import logger
import statistics
import torch
import yaml

def check_path(path: str):
    """
    偵測給定的 path 路徑，若沒有會自動創建新的資料夾
    """
    if path:
        os.makedirs(path, exist_ok=True)


def generate_run_id() -> str:
    """產生唯一的 run_id，用於綁定影片與日誌等產物。"""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    short = uuid.uuid4().hex[:8]
    return f"{ts}_{short}"


def parse_overrides(overrides: str | None) -> dict:
    """將 key=value,key2=value2 的字串解析為 dict，值支援 literal_eval。"""
    if not overrides:
        return {}
    result = {}
    for pair in overrides.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            result[k] = ast.literal_eval(v)
        except Exception:
            # 當作字串
            result[k] = v
    return result


def load_track_config(
    config_path: str,
    name: str,
    device: str,
    overrides: dict | None = None,
) -> dict:
    """
    從 JSON 檔載入指定名稱的 track_config；找不到時使用內建預設。
    可用 overrides 覆寫欄位，例如 conf、imgsz、classes、iou 等。
    """
    # 內建預設
    default_presets = {
        "default": {
            "conf": 0.5,
            "verbose": False,
            "tracker": "bytetrack.yaml",
            "imgsz": 640,
            "classes": [0],
            "device": device,
            "persist": True,
            "iou": 0.5,
        },
        "fast": {
            "conf": 0.6,
            "verbose": False,
            "tracker": "bytetrack.yaml",
            "imgsz": 512,
            "classes": [0],
            "device": device,
            "persist": True,
            "iou": 0.45,
        },
        "accurate": {
            "conf": 0.4,
            "verbose": False,
            "tracker": "bytetrack.yaml",
            "imgsz": 960,
            "classes": [0],
            "device": device,
            "persist": True,
            "iou": 0.5,
        },
    }

    cfgs = default_presets
    p = Path(config_path)
    if p.exists():
        try:
            cfgs.update(json.loads(p.read_text()))
        except Exception as e:
            logger.warning(f"讀取 {config_path} 失敗，改用內建預設: {e}")

    if name not in cfgs:
        logger.warning(f"找不到 track_config 名稱 '{name}'，改用 'default'")
        name = "default"

    cfg = dict(cfgs[name])
    # 套用裝置
    cfg["device"] = device
    # 套用覆寫
    if overrides:
        cfg.update(overrides)

    # 檢查是否使用自定義追蹤器配置並記錄到日誌中
    tracker_name = cfg.get("tracker", "")
    if tracker_name and tracker_name not in ["bytetrack.yaml", "botsort.yaml"]:
        logger.info(f"檢測到自定義追蹤器配置: {tracker_name}")
        # 嘗試讀取自定義追蹤器配置文件
        tracker_path = Path(tracker_name)
        if tracker_path.exists():
            try:
                with open(tracker_path, 'r', encoding='utf-8') as f:
                    custom_tracker_config = yaml.safe_load(f)
                logger.info(f"自定義追蹤器配置內容: {json.dumps(custom_tracker_config, ensure_ascii=False, indent=2)}")
            except Exception as e:
                logger.warning(f"讀取自定義追蹤器配置文件 {tracker_name} 失敗: {e}")
        else:
            logger.warning(f"自定義追蹤器配置文件 {tracker_name} 不存在")

    return cfg

def run(
    weights,
    source="test.mp4",
    view_img=False,
    save_img=False,
    save_path="ultralytics_results",
    *,
    run_id: str | None = None,
    track_config_name: str = "default",
    track_configs_path: str = "track_configs.json",
    track_overrides: str | None = None,
):
    """
    Run object detection on a video using YOLOv8

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
    """

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # equipment_model = YOLO(equipment_weight)
    # pose_model = YOLO(pose_weights)
    model = YOLO(weights)

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
    total_frames = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))

    check_path(save_path)

    # 準備 run_id 與輸出檔名
    run_id = run_id or generate_run_id()
    source_stem = Path(source).stem
    video_out_path = Path(save_path) / f"{source_stem}__{run_id}.mp4"
    video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    inference_times = []  # 儲存每個 frame 的推理時間

    label_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                    (255, 255, 0), (0, 255, 255), (255, 0, 255),
                    (192, 192, 192), (128, 0, 0), (128, 128, 0),
                    (0, 128, 0), (128, 0, 128), (0, 128, 128),
                    (0, 0, 128), (72, 61, 139), (47, 79, 79),
                    (0, 206, 209), (148, 0, 211), (255, 20, 147),
                    (255, 165, 0)
                    ]

    # 創建進度條
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")

    logger.info(f"開始處理視頻: {source}")
    logger.info(f"總幀數: {total_frames}")

    # 根據 CUDA 可用性自動選擇推理裝置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Track 設定參數（由管理機制取得）
    overrides = parse_overrides(track_overrides)
    track_config = load_track_config(track_configs_path, track_config_name, device, overrides)

    # 在第一個 frame 前記錄一次 track 設定與環境
    logger.info(f"Track 設定參數: {track_config}")
    logger.info(
        f"torch.cuda.is_available(): {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}"
    )

    while videocapture.isOpened():
        # print(f"Frame: {frame_count}")  # 註解掉原本的輸出，使用進度條代替

        success, frame = videocapture.read()
        if not success:
            break

        boxes_list = []
        class_list = []
        class_num_list = []

        # 記錄推理開始時間
        inference_start_time = time.time()

        results = model.track(frame, **track_config)  # 動態展開參數，偵測 person 類別

        # 記錄推理結束時間並計算耗時
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time
        inference_times.append(inference_time)

        # 記錄每個 frame 的推理時間
        logger.debug(f"Frame {frame_count}: 推理時間 {inference_time:.4f} 秒")

        for person_result in results:
            if person_result.boxes.xyxy.numel() > 0:
                person_bbox = person_result.boxes.xyxy.clone().cpu().numpy()
                boxes_list.extend([(x1, y1, x2, y2) for x1, y1, x2, y2 in person_bbox])
                class_list.extend([results[0].names[int(cls.item())] for cls in person_result.boxes.cls])
                class_num_list.extend([int(cls.item()) for cls in person_result.boxes.cls])

        if boxes_list and class_list:
            for i, (box, cls, cls_num) in enumerate(zip(boxes_list, class_list, class_num_list)):
                x1, y1, x2, y2 = box
                color = label_colors[cls_num]

                # 計算 pixel 大小（檢測框面積）
                pixel_width = int(x2 - x1)
                pixel_height = int(y2 - y1)

                # 取得 track ID（如果有的話）
                track_id = None
                if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None and i < len(results[0].boxes.id):
                    track_id = int(results[0].boxes.id[i].item())

                # 建立 unique ID: <class>_<track_id>_<pixel大小>
                if track_id is not None:
                    unique_id = f"{cls}_{track_id} {pixel_width}x{pixel_height}"
                else:
                    unique_id = f"{cls}_no_id {pixel_width}x{pixel_height}"

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # 顯示包含 pixel 大小的 unique ID
                label = unique_id
                t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
                cv2.rectangle(
                    frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), color, -1
                )
                cv2.putText(
                    frame, label, (int(x1), int(y1) - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA
                )

        if view_img:
            cv2.imshow(Path(source).stem, frame)
        if save_img:
            if frame is not None:
                video_writer.write(frame)
            else:
                print("Warning: trying to write an empty frame")

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        frame_count += 1
        pbar.update(1)  # 更新進度條

    pbar.close()  # 關閉進度條

    # 計算統計數據
    if inference_times:
        # 排除第一幀的推理時間（第一幀通常有初始化開銷）
        inference_times_filtered = inference_times[2:] if len(inference_times) > 1 else inference_times

        avg_inference_time = statistics.mean(inference_times_filtered)
        min_inference_time = min(inference_times_filtered)
        max_inference_time = max(inference_times_filtered)
        median_inference_time = statistics.median(inference_times_filtered)

        logger.info(f"處理完成！統計數據:")
        logger.info(f"總幀數: {frame_count}")
        logger.info(f"第一幀推理時間（已排除）: {inference_times[0]:.4f} 秒" if len(inference_times) > 1 else "")
        logger.info(f"平均推理時間（排除前2幀）: {avg_inference_time:.4f} 秒")
        logger.info(f"最小推理時間: {min_inference_time:.4f} 秒")
        logger.info(f"最大推理時間: {max_inference_time:.4f} 秒")
        logger.info(f"中位數推理時間: {median_inference_time:.4f} 秒")
        logger.info(f"平均 FPS（排除前2幀）: {1/avg_inference_time:.2f}")
    else:
        logger.warning("沒有記錄到推理時間數據")

    video_writer.release()
    videocapture.release()

    # 輸出本次 run 的 metadata，綁定影片與 log 與參數
    meta = {
        "run_id": run_id,
        "weights": str(weights),
        "source": str(source),
        "video_path": str(video_out_path),
        "log_path": None,  # 將在 main 中設置後填入
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "total_frames": total_frames,
        "device": device,
        "track_config": track_config,
        "stats": {
            "frames_processed": frame_count,
            "avg_inference_time": (statistics.mean(inference_times[2:]) if len(inference_times) > 2 else (statistics.mean(inference_times) if inference_times else None)),
        },
    }

    # 在 caller 設定 log 檔後回寫 meta 的 log_path
    return meta

# person_weights = "models/yolov8n.pt"
# person_weights = "models/3_360_person_equipment_yolov8m-seg/model/best.pt"
person_weights = "models/yolo12x.pt"

# source = 'RAW_SHORT_VIDEOS/camera-d8accaf0-2a04-4e83-a43c-511ec5fc0a0e_20250829_0900_s0s_1x_300s.mp4'
source = 'RAW_SHORT_VIDEOS/camera-d8accaf0_60s.mp4'
output_path = 'output_videos'

view_img=False
save_img=True

# 設置 loguru 日誌配置（在 main 中依 run_id 綁定檔名）
logger.remove()  # 先移除預設的控制台輸出，稍後在 main 重新配置

start_time = time.time()

if __name__ == "__main__":
    # 產生唯一 run_id，並依此配置 logger 與輸出
    run_id = generate_run_id()

    # 準備 logs 與輸出資料夾
    log_dir = "logs"
    check_path(log_dir)
    check_path(output_path)

    # 綁定 run_id 到 logger 並設置輸出（檔名帶 run_id 即可和影片綁定）
    logger = logger.bind(run_id=run_id)
    log_file = f"{log_dir}/inference_{run_id}.log"
    logger.add(
        log_file,
        rotation="1 day",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {extra[run_id]} | {message}",
    )
    # 也輸出到控制台，與 tqdm 兼容
    logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO", format="{message}")

    logger.info("開始執行 YOLOv8 視頻推理程序")

    meta = run(
        person_weights,
        source,
        view_img,
        save_img,
        save_path=output_path,
        run_id=run_id,
        track_config_name=os.getenv("TRACK_CONFIG", "default"),
        track_configs_path=os.getenv("TRACK_CONFIGS_PATH", "track_configs.json"),
        track_overrides=os.getenv("TRACK_OVERRIDES"),  # 例: "conf=0.45,imgsz=800,classes=[0]"
    )

    # 回寫 meta 的 log_path 並落盤到與影片相同路徑（相同檔名前綴）
    meta["log_path"] = log_file
    meta_path = Path(meta["video_path"]).with_suffix("")
    meta_json_path = Path(f"{meta_path}.json")
    try:
        meta_json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        logger.info(f"本次 run metadata 已輸出: {meta_json_path}")
    except Exception as e:
        logger.warning(f"寫入 metadata 失敗: {e}")

total_time = time.time() - start_time
logger.info(f"總執行時間: {total_time:.2f} 秒")
print("--- %s seconds ---" % (time.time() - start_time))
